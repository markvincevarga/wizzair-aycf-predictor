from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from features.generate import generate_route_date_samples
from features.mapping import (
    availabilities_windows_to_daily_start,
    datetime_columns_to_dates,
    remove_data_generated,
)


# =============================================================================
# Route Utilities
# =============================================================================


def extract_routes(df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Extract all unique routes from an availabilities DataFrame.

    Args:
        df: Availabilities DataFrame containing at least 'departure_from' and
            'departure_to' columns.

    Returns:
        List of unique (departure_from, departure_to) tuples representing all routes.

    Raises:
        TypeError: If df is not a DataFrame.
        ValueError: If df is missing required columns.

    Example:
        >>> df = pd.DataFrame({
        ...     "departure_from": ["BUD", "BUD", "LTN"],
        ...     "departure_to": ["LTN", "LTN", "BUD"],
        ...     "availability_start": ["2024-01-01", "2024-01-02", "2024-01-01"],
        ... })
        >>> extract_routes(df)
        [('BUD', 'LTN'), ('LTN', 'BUD')]
    """
    if df is None:
        raise TypeError("df must be a pandas DataFrame, got None")

    required = {"departure_from", "departure_to"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    if df.empty:
        return []

    unique_routes = (
        df[["departure_from", "departure_to"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    return list(unique_routes)


def _infer_route_group_cols(df: pd.DataFrame) -> list[str]:
    """
    Infer the columns that uniquely identify a route in our feature frames.

    Base route key: departure_from, departure_to
    If both country columns exist, they are included as well.
    """
    if df is None:
        raise TypeError("df must be a pandas DataFrame, got None")

    group_cols = ["departure_from", "departure_to"]
    if "departure_from_country" in df.columns and "departure_to_country" in df.columns:
        group_cols += ["departure_from_country", "departure_to_country"]

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required route columns: {missing}")

    return group_cols


# =============================================================================
# Date Utilities
# =============================================================================


def _add_date_parts(
    df: pd.DataFrame,
    *,
    date_col: str = "availability_start",
) -> pd.DataFrame:
    """
    Add calendar-derived features from the per-day availability date column.

    Adds:
    - day_of_week: Monday=0 ... Sunday=6
    - week_of_year: ISO week number (1..53)
    - month_of_year: month number (1..12)
    """
    if df is None:
        raise TypeError("df must be a pandas DataFrame, got None")

    # Intentionally mutate the incoming dataframe in-place (no copies).
    if date_col not in df.columns:
        raise ValueError(f"df must contain '{date_col}'")

    # Ensure columns exist (even for empty frames) with a nullable integer dtype.
    if df.empty:
        df["day_of_week"] = pd.Series(dtype="Int64")
        df["week_of_year"] = pd.Series(dtype="Int64")
        df["month_of_year"] = pd.Series(dtype="Int64")
        return df

    s = pd.to_datetime(df[date_col], errors="coerce")
    df["day_of_week"] = s.dt.dayofweek.astype("Int64")
    df["week_of_year"] = s.dt.isocalendar().week.astype("Int64")
    df["month_of_year"] = s.dt.month.astype("Int64")

    return df


def build_route_date_grid(
    routes: list[tuple[str, str]],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Generate all possible availabilities for every route and every day within a timeframe.

    Creates a complete (routes × days) grid where each row represents a potential
    availability for a specific route on a specific day.

    Args:
        routes: List of route pairs as (departure_from, departure_to) tuples.
                Example: [("BUD", "LTN"), ("LTN", "BUD"), ("BUD", "BCN")]
        start: Start of the timeframe (inclusive), naive datetime.
        end: End of the timeframe (inclusive), naive datetime.

    Returns:
        DataFrame with columns:
        - departure_from: Airport code for departure
        - departure_to: Airport code for arrival
        - availability_start: Date of the availability (one row per day)

    Raises:
        TypeError: If routes is not a list or start/end are not datetime objects.
        ValueError: If routes is empty, start > end, or route tuples are malformed.

    Example:
        >>> routes = [("BUD", "LTN"), ("BUD", "BCN")]
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 1, 3)
        >>> df = build_route_date_grid(routes, start, end)
        >>> len(df)  # 2 routes × 3 days = 6 rows
        6
    """ 

    start_date = start.date()
    end_date = end.date()

    if start_date > end_date:
        raise ValueError(f"start must be <= end, got start={start_date}, end={end_date}")

    # Generate daily date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    dates = [d.date() for d in date_range]

    # Build the cross-product of routes × dates
    rows = []
    for departure_from, departure_to in routes:
        for day in dates:
            rows.append({
                "departure_from": departure_from,
                "departure_to": departure_to,
                "availability_start": day,
            })

    df = pd.DataFrame(rows)
    df = _add_date_parts(df)
    
    return df


# =============================================================================
# Feature Builders
# =============================================================================


def add_lagged_feature(
    df: pd.DataFrame,
    *,
    lag: int,
    date_col: str = "availability_start",
    occurs_col: str = "occurs",
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add a lagged `occurs` feature column to an existing (route x day) frame.

    - Mutates `df` in-place by adding the output column.
    - Lag is computed *within each route* after sorting by `date_col`.
    - Requires `df` to contain `date_col` and `occurs_col`.

    Args:
        df: Existing feature dataframe.
        lag: Number of rows (days, if the frame is daily) to lag by. Must be >= 1.
        date_col: Date column to order by (default: availability_start).
        occurs_col: Label/feature column to lag (default: occurs).
        out_col: Optional override for output column name.
                 Defaults to `f"{occurs_col}_lag_{lag}"`.

    Returns:
        The same dataframe instance, with the new column added.
    """
    if df is None:
        raise TypeError("df must be a pandas DataFrame, got None")
    if not isinstance(lag, int):
        raise TypeError("lag must be an int")
    if lag < 1:
        raise ValueError("lag must be >= 1")

    if date_col not in df.columns:
        raise ValueError(f"df must contain '{date_col}'")
    if occurs_col not in df.columns:
        raise ValueError(f"df must contain '{occurs_col}'")

    col_name = out_col or f"{occurs_col}_lag_{lag}"

    group_cols = _infer_route_group_cols(df)

    if df.empty:
        df[col_name] = pd.Series(dtype="Float64")
        return df

    # Compute lag on a sorted view, then align back to original row order.
    sort_cols = group_cols + [date_col]
    sorted_idx = df.sort_values(by=sort_cols, ascending=True, kind="mergesort").index
    df_sorted = df.loc[sorted_idx]
    lagged_sorted = df_sorted.groupby(group_cols, sort=False)[occurs_col].shift(lag)
    df[col_name] = lagged_sorted.reindex(df.index).astype("Float64")

    return df


def add_rolling_mean_feature(
    df: pd.DataFrame,
    *,
    window_days: int,
    date_col: str = "availability_start",
    occurs_col: str = "occurs",
    out_col: Optional[str] = None,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Add a per-route rolling mean of `occurs` over the previous `window_days` days.

    - Mutates `df` in-place by adding the output column.
    - Computed within each route after sorting by `date_col`.
    - Excludes the current day by shifting by 1 before rolling.

    Output column default: `f"{occurs_col}_mean_prev_{window_days}d"`.
    """
    if df is None:
        raise TypeError("df must be a pandas DataFrame, got None")
    if not isinstance(window_days, int):
        raise TypeError("window_days must be an int")
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    if not isinstance(min_periods, int):
        raise TypeError("min_periods must be an int")
    if min_periods < 1:
        raise ValueError("min_periods must be >= 1")

    if date_col not in df.columns:
        raise ValueError(f"df must contain '{date_col}'")
    if occurs_col not in df.columns:
        raise ValueError(f"df must contain '{occurs_col}'")

    col_name = out_col or f"{occurs_col}_mean_prev_{window_days}d"
    group_cols = _infer_route_group_cols(df)

    if df.empty:
        df[col_name] = pd.Series(dtype="Float64")
        return df

    sort_cols = group_cols + [date_col]
    sorted_idx = df.sort_values(by=sort_cols, ascending=True, kind="mergesort").index
    df_sorted = df.loc[sorted_idx]

    rolling_sorted = df_sorted.groupby(group_cols, sort=False, group_keys=False)[
        occurs_col
    ].apply(
        lambda s: s.shift(1).rolling(window=window_days, min_periods=min_periods).mean()
    )

    df[col_name] = rolling_sorted.reindex(df.index).astype("Float64")
    return df


def add_route_count_features(
    df: pd.DataFrame,
    *,
    occurs_col: str = "occurs",
) -> pd.DataFrame:
    """
    Add route-level count and leave-one-out mean encoding features.

    Mutates `df` in-place by adding:
    - route_count: Number of observations per route
    - route_count_log: Log-transformed route count (log1p)
    - route_loo_mean: Leave-one-out mean of occurs per route

    Args:
        df: Existing feature dataframe with route columns and occurs.
        occurs_col: Label column to use for LOO encoding (default: occurs).

    Returns:
        The same dataframe instance, with new columns added.
    """
    if df is None:
        raise TypeError("df must be a pandas DataFrame, got None")

    if occurs_col not in df.columns:
        raise ValueError(f"df must contain '{occurs_col}'")

    group_cols = _infer_route_group_cols(df)

    if df.empty:
        df["route_count"] = pd.Series(dtype="Int64")
        df["route_count_log"] = pd.Series(dtype="Float64")
        df["route_loo_mean"] = pd.Series(dtype="Float64")
        return df

    # Route count: number of observations per route
    route_counts = df.groupby(group_cols).size()
    df["route_count"] = df.set_index(group_cols).index.map(route_counts).astype("Int64")
    df["route_count_log"] = np.log1p(df["route_count"].astype(float))

    # Leave-one-out mean encoding
    route_sums = df.groupby(group_cols)[occurs_col].transform("sum")
    route_loo_mean = (route_sums - df[occurs_col]) / (df["route_count"] - 1)
    global_mean = df[occurs_col].mean()
    df["route_loo_mean"] = route_loo_mean.fillna(global_mean).astype("Float64")

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived feature columns to an existing availabilities feature frame.

    This is intended to be called *after* building the base labeled frame
    (e.g. from `build_labeled_features`), so it can be reused consistently
    across training and prediction pipelines.
    """
    if df is None:
        raise TypeError("df must be a pandas DataFrame, got None")

    # Intentionally mutate the incoming dataframe in-place (no copies).
    _add_date_parts(df, date_col="availability_start")

    # Lagged label features (only when `occurs` exists on the frame).
    if "occurs" in df.columns:
        add_lagged_feature(df, lag=1)
        add_lagged_feature(df, lag=2)
        add_lagged_feature(df, lag=3)
        add_rolling_mean_feature(df, window_days=7)
        add_rolling_mean_feature(df, window_days=14)
        df["occurs_lag_1x2"] = df["occurs_lag_1"] * df["occurs_lag_2"]
        add_route_count_features(df)

    return df


# =============================================================================
# Main Functions
# =============================================================================


def build_labeled_features(
    availabilities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a labeled (route x day) feature frame from an availabilities DataFrame.

    The input dataframe is expected to look like `storage.availabilities.Availabilities.get_all(...)`,
    i.e. include at least:
    - departure_from, departure_to
    - availability_start, availability_end
    - optionally data_generated, id, and country code columns

    Timeline boundaries are determined per-route automatically based on each route's
    first and last observed availability dates.
    """
    if availabilities is None:
        raise TypeError("availabilities must be a pandas DataFrame, got None")

    # Positive (observed) rows: expand windows to daily `availability_start` rows and drop `availability_end`
    cleaned = datetime_columns_to_dates(remove_data_generated(availabilities))
    pos = availabilities_windows_to_daily_start(cleaned)
    pos = pos.copy()
    pos["occurs"] = 1.0

    # Negative sampling frame (same schema as the daily positives; no availability_end)
    # Timeline boundaries are set per-route by generate_route_date_samples.
    neg = generate_route_date_samples(
        availabilities=cleaned.drop(columns=["availability_end"])
        if "availability_end" in cleaned.columns
        else cleaned,
    ).copy()
    neg["occurs"] = 0.0

    # Align columns (in case one side is empty)
    combined = pd.concat([pos, neg], ignore_index=True, sort=False)

    if combined.empty:
        # Ensure `occurs` exists and is float even for empty frames.
        combined["occurs"] = combined.get("occurs", pd.Series(dtype="float")).astype(
            float
        )
        return add_derived_features(combined)

    # Prefer occurs=1.0 when a sampled row collides with an observed row.
    combined["occurs"] = combined["occurs"].astype(float)

    # Sort so occurs=1.0 comes first; mergesort is stable (handy if you later add tie-breaks).
    combined = combined.sort_values(by=["occurs"], ascending=False, kind="mergesort")

    # Drop duplicates ignoring 'occurs' and ignoring 'id' (id differs between observed vs sampled).
    ignore_cols = {"occurs"}
    if "id" in combined.columns:
        ignore_cols.add("id")
    dedupe_subset = [c for c in combined.columns if c not in ignore_cols]

    combined = combined.drop_duplicates(subset=dedupe_subset, keep="first").reset_index(
        drop=True
    )

    return add_derived_features(combined)


def sort_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort a labeled availabilities feature frame by route then date.

    Sorting keys (when present):
    - departure_from, departure_to
    - availability_start, availability_end
    - departure_from_country, departure_to_country (if present)
    """
    if df is None:
        raise TypeError("df must be a pandas DataFrame, got None")

    sort_cols: list[str] = []
    for c in [
        "departure_from",
        "departure_to",
        "departure_from_country",
        "departure_to_country",
        "availability_start",
    ]:
        if c in df.columns:
            sort_cols.append(c)

    if not sort_cols:
        return df.copy()

    return df.sort_values(by=sort_cols, ascending=True, kind="mergesort").reset_index(
        drop=True
    )
