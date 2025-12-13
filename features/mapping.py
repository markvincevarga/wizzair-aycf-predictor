from __future__ import annotations

from datetime import date, datetime
from typing import Union

import pandas as pd


DateLike = Union[str, date, datetime, pd.Timestamp]


def remove_data_generated(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the `data_generated` column (if present).

    Intended input: availabilities dataframe already enriched with country codes,
    e.g. from `storage.availabilities.Availabilities.get_all(include_country_codes=True)`.
    """
    if df is None:
        raise TypeError("df must be a pandas DataFrame, got None")

    if "data_generated" not in df.columns:
        return df.copy()

    return df.drop(columns=["data_generated"]).copy()


def datetime_columns_to_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all datetime-like columns in `df` to plain dates.

    - Columns with dtype datetime64[ns] (incl. tz-aware) are converted via `.dt.date`.
    - Object columns containing pandas Timestamps / python datetimes are coerced safely.
    """
    if df is None:
        raise TypeError("df must be a pandas DataFrame, got None")

    out = df.copy()
    for col in out.columns:
        s = out[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            out[col] = s.dt.date
            continue

        # Handle object columns that actually contain datetimes/timestamps
        if s.dtype == "object":
            # Fast-path: if there are no non-null values, nothing to do
            non_null = s.dropna()
            if non_null.empty:
                continue

            sample = non_null.iloc[0]
            if isinstance(sample, (pd.Timestamp, datetime)):
                out[col] = pd.to_datetime(s, errors="coerce").dt.date

    return out


def availabilities_windows_to_daily_start(
    df: pd.DataFrame,
    *,
    start_col: str = "availability_start",
    end_col: str = "availability_end",
) -> pd.DataFrame:
    """
    Expand availability windows into per-day rows, keeping only `availability_start`.

    Input: rows with [start_col, end_col] (date-like), inclusive.
    Output: rows where `start_col` is a single day (per row), and `end_col` is removed.

    This is the feature-level representation we want for training/prediction:
    one row per (route, date) instead of a (route, start/end window).
    """
    if df is None:
        raise TypeError("df must be a pandas DataFrame, got None")
    if df.empty:
        out = df.copy()
        if end_col in out.columns:
            out = out.drop(columns=[end_col])
        return out

    if start_col not in df.columns or end_col not in df.columns:
        raise ValueError(f"df must contain '{start_col}' and '{end_col}'")

    base = df.copy()
    # Normalize to pandas datetime for safe date_range expansion.
    s = pd.to_datetime(base[start_col], errors="coerce")
    e = pd.to_datetime(base[end_col], errors="coerce")
    base = base.assign(_s=s, _e=e).dropna(subset=["_s", "_e"])
    if base.empty:
        out = df.head(0).copy()
        if end_col in out.columns:
            out = out.drop(columns=[end_col])
        return out

    base[start_col] = base.apply(
        lambda r: pd.date_range(start=r["_s"], end=r["_e"], freq="D").date, axis=1
    )
    out = base.explode(start_col, ignore_index=True)

    out = out.drop(columns=["_s", "_e"])
    if end_col in out.columns:
        out = out.drop(columns=[end_col])

    return out
