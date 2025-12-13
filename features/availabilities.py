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

    # Positive (observed) rows: expand windows to daily `availability_start` rows and drop `availability_end`
    cleaned = datetime_columns_to_dates(remove_data_generated(availabilities))
    pos = availabilities_windows_to_daily_start(cleaned)
    pos = pos.copy()
    pos["occurs"] = 1.0

    # Negative sampling frame (same schema as the daily positives; no availability_end)
    neg = generate_route_date_samples(
        availabilities=pos.drop(columns=["occurs"]),
        start_date=start_date,
        end_date=end_date,
    ).copy()
    neg["occurs"] = 0.0

    # Align columns (in case one side is empty)
    combined = pd.concat([pos, neg], ignore_index=True, sort=False)

    if combined.empty:
        # Ensure `occurs` exists and is float even for empty frames.
        combined["occurs"] = combined.get("occurs", pd.Series(dtype="float")).astype(float)
        return combined

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

    return combined

