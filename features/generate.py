from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from features.mapping import DateLike, datetime_columns_to_dates, remove_data_generated


@dataclass(frozen=True)
class RouteWindow:
    departure_from: str
    departure_to: str
    departure_from_country: Optional[str]
    departure_to_country: Optional[str]
    first_date: date
    last_date: date


def generate_route_date_samples(
    availabilities: pd.DataFrame,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
) -> pd.DataFrame:
    """
    Generate a complete (route x date) frame from historical availabilities.

    What it does:
    - Identifies unique routes from the input dataframe.
    - For each route, finds its first and last occurrence using the min of
      `availability_start` and max of `availability_end`.
    - For each route, generates *all possible days* (daily grid) within:
        [max(route_first, start_date), min(route_last, end_date)]
      (where the global start/end are optional).
    - Returns a dataframe shaped like `datetime_columns_to_dates(remove_data_generated(availabilities))`:
      - `data_generated` is not present
      - datetime columns are plain `date`
      - column set & order match the cleaned input (missing values filled with NA)

    Notes:
    - Output is a per-day representation with `availability_start` only
      (and no `availability_end`).
    """
    if availabilities is None:
        raise TypeError("availabilities must be a pandas DataFrame, got None")

    if (
        "departure_from" not in availabilities.columns
        or "departure_to" not in availabilities.columns
    ):
        raise ValueError(
            "availabilities must contain 'departure_from' and 'departure_to' columns"
        )

    # We standardize outputs to a per-day representation with `availability_start`
    # (and no `availability_end`). Inputs may still include `availability_end`
    # for window inference.
    if "availability_start" not in availabilities.columns:
        raise ValueError("availabilities must contain 'availability_start'")

    cleaned_template = datetime_columns_to_dates(remove_data_generated(availabilities))
    if "availability_end" in cleaned_template.columns:
        cleaned_template = cleaned_template.drop(columns=["availability_end"])

    # If there's no data, return an empty frame with the expected schema.
    if availabilities.empty:
        return cleaned_template.head(0).copy()

    # Parse optional global start/end and normalize to date
    def _parse_to_date(x: Optional[DateLike]) -> Optional[date]:
        if x is None:
            return None
        ts = pd.to_datetime(x, errors="raise")
        return ts.date()

    global_start = _parse_to_date(start_date)
    global_end = _parse_to_date(end_date)
    if (
        global_start is not None
        and global_end is not None
        and global_start > global_end
    ):
        raise ValueError("start_date must be <= end_date")

    # Work in date-space to keep route window logic consistent.
    d0 = datetime_columns_to_dates(availabilities).copy()
    start_s = pd.to_datetime(d0["availability_start"], errors="coerce")
    d0["_route_first_date"] = start_s.dt.date

    if "availability_end" in d0.columns:
        end_s = pd.to_datetime(d0["availability_end"], errors="coerce")
        d0["_route_last_date"] = end_s.dt.date
    else:
        d0["_route_last_date"] = start_s.dt.date

    d0 = d0.dropna(subset=["_route_first_date", "_route_last_date"])

    if d0.empty:
        return cleaned_template.head(0).copy()

    has_countries = ("departure_from_country" in d0.columns) and (
        "departure_to_country" in d0.columns
    )
    group_cols = ["departure_from", "departure_to"]
    if has_countries:
        group_cols += ["departure_from_country", "departure_to_country"]

    grouped = (
        d0.groupby(group_cols, dropna=False, sort=False)
        .agg(
            first_date=("_route_first_date", "min"),
            last_date=("_route_last_date", "max"),
        )
        .reset_index()
    )

    # Build per-route date ranges
    rows = []
    for _, r in grouped.iterrows():
        route_first: date = r["first_date"]
        route_last: date = r["last_date"]

        route_start = (
            route_first if global_start is None else max(route_first, global_start)
        )
        route_end = route_last if global_end is None else min(route_last, global_end)
        if route_start > route_end:
            continue

        # Full daily grid, then subsample by fraction.
        dr = pd.date_range(start=route_start, end=route_end, freq="D")
        if dr.empty:
            continue

        # Normalize (ensures consistent Timestamp-like values).
        dr = pd.to_datetime(dr, errors="raise")

        dates = [pd.Timestamp(d).date() for d in dr]
        payload: dict[str, object] = {
            "departure_from": r["departure_from"],
            "departure_to": r["departure_to"],
            "availability_start": dates,
        }
        if has_countries:
            payload["departure_from_country"] = r["departure_from_country"]
            payload["departure_to_country"] = r["departure_to_country"]

        route_df = pd.DataFrame(payload)
        rows.append(route_df)

    if not rows:
        return cleaned_template.head(0).copy()

    sampled = pd.concat(rows, ignore_index=True)

    # Make output match the cleaned input schema (order + any extra columns like `id`)
    # Fill missing columns with NA, drop extras not in template.
    sampled = sampled.reindex(columns=cleaned_template.columns, fill_value=pd.NA)

    # If cleaned input includes `id`, we cannot invent it; keep as NA.
    if "id" in sampled.columns:
        sampled["id"] = pd.NA

    # Ensure date-like columns are plain dates (not timestamps)
    sampled = datetime_columns_to_dates(sampled)

    return sampled
