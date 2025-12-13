from __future__ import annotations

from typing import Optional

import pandas as pd

from database import DatabaseWrapper
from features.generate import generate_route_date_samples
from features.mapping import (
    DateLike,
    availabilities_windows_to_daily_start,
    datetime_columns_to_dates,
    remove_data_generated,
)
from storage.availabilities import Availabilities


def _add_availability_date_parts(
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


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived feature columns to an existing availabilities feature frame.

    This is intended to be called *after* building the base labeled frame
    (e.g. from `build_availabilities_occurs_feature_from_df`), so it can be reused
    consistently across training and prediction pipelines.
    """
    if df is None:
        raise TypeError("df must be a pandas DataFrame, got None")

    # Intentionally mutate the incoming dataframe in-place (no copies).
    _add_availability_date_parts(df, date_col="availability_start")

    # Lagged label features (only when `occurs` exists on the frame).
    if "occurs" in df.columns:
        add_lagged_occurs_feature(df, lag=1)
        add_lagged_occurs_feature(df, lag=2)
        add_lagged_occurs_feature(df, lag=3)
        add_rolling_occurs_mean_feature(df, window_days=7)
        add_rolling_occurs_mean_feature(df, window_days=14)

    return df


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


def add_lagged_occurs_feature(
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


def add_rolling_occurs_mean_feature(
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
    ].apply(lambda s: s.shift(1).rolling(window=window_days, min_periods=min_periods).mean())

    df[col_name] = rolling_sorted.reindex(df.index).astype("Float64")
    return df


def build_availabilities_occurs_feature(
    db: DatabaseWrapper,
    *,
    include_country_codes: bool = True,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
) -> pd.DataFrame:
    """
    Build a labeled (route x date) feature frame for AYCF availability occurrence.

    Steps:
    - Load availabilities from the database.
    - Drop the ingestion timestamp column (`data_generated`) and convert datetime columns to `date`.
    - Label observed availabilities with `occurs = 1.0`.
    - Generate a negative frame from the same routes/dates (full daily grid),
      and label those with `occurs = 0.0`.
    - Concatenate and de-duplicate, keeping `occurs = 1.0` when both exist for the same row.

    Notes:
    - De-duplication intentionally ignores `id` (and `occurs`) so that sampled rows can be
      matched against observed rows.
    """
    domain = Availabilities(db)
    raw = domain.get_all(include_country_codes=include_country_codes)
    return build_availabilities_occurs_feature_from_df(
        raw,
        start_date=start_date,
        end_date=end_date,
    )


def sort_availabilities_occurs_feature(df: pd.DataFrame) -> pd.DataFrame:
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


def build_availabilities_occurs_feature_from_df(
    availabilities: pd.DataFrame,
    *,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
) -> pd.DataFrame:
    """
    Same as `build_availabilities_occurs_feature`, but starts from an availabilities DataFrame.

    The input dataframe is expected to look like `storage.availabilities.Availabilities.get_all(...)`,
    i.e. include at least:
    - departure_from, departure_to
    - availability_start, availability_end
    - optionally data_generated, id, and country code columns
    """
    if availabilities is None:
        raise TypeError("availabilities must be a pandas DataFrame, got None")

    # Normalize optional global start/end to plain `date` so comparisons are consistent.
    def _parse_to_date(x: Optional[DateLike]):
        if x is None:
            return None
        return pd.to_datetime(x, errors="raise").date()

    window_start = _parse_to_date(start_date)
    window_end = _parse_to_date(end_date)
    if window_start is not None and window_end is not None and window_start > window_end:
        raise ValueError("start_date must be <= end_date")

    # Positive (observed) rows: expand windows to daily `availability_start` rows and drop `availability_end`
    cleaned = datetime_columns_to_dates(remove_data_generated(availabilities))
    pos = availabilities_windows_to_daily_start(cleaned)
    pos = pos.copy()
    pos["occurs"] = 1.0

    # Apply the global window to positives as well (otherwise start/end only constrain negatives).
    if not pos.empty and "availability_start" in pos.columns:
        if window_start is not None:
            pos = pos[pos["availability_start"] >= window_start]
        if window_end is not None:
            pos = pos[pos["availability_start"] <= window_end]

    # Negative sampling frame (same schema as the daily positives; no availability_end)
    neg = generate_route_date_samples(
        availabilities=cleaned.drop(columns=["availability_end"])
        if "availability_end" in cleaned.columns
        else cleaned,
        start_date=window_start,
        end_date=window_end,
    ).copy()
    neg["occurs"] = 0.0

    # Align columns (in case one side is empty)
    combined = pd.concat([pos, neg], ignore_index=True, sort=False)

    if combined.empty:
        # Ensure `occurs` exists and is float even for empty frames.
        combined["occurs"] = combined.get("occurs", pd.Series(dtype="float")).astype(float)
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

    combined = combined.drop_duplicates(subset=dedupe_subset, keep="first").reset_index(drop=True)

    return add_derived_features(combined)

