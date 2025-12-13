from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_country_codes(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper()


def add_holiday_distance_features(
    availabilities: pd.DataFrame,
    holidays: pd.DataFrame,
    *,
    availability_date_col: str = "availability_start",
    departure_country_col: str = "departure_from_country",
    destination_country_col: str = "departure_to_country",
    holidays_country_col: str = "countryIsoCode",
    holidays_start_col: str = "startDate",
    holidays_end_col: str = "endDate",
    holidays_category_col: str = "category",
    holidays_nationwide_col: str = "nationwide",
    categories: tuple[str, ...] = ("Public",),
    country_fallback_map: dict[str, str] | None = None,
    fill_missing_value: int | None = None,
) -> pd.DataFrame:
    """
    Add holiday-distance features for both departure and destination countries (in-place).

    For each row and each side (departure/destination), computes distances to the nearest
    nationwide holiday *interval* for the given country:
    - `<side>_days_until_next_holiday`: days until the next holiday (0 if within a holiday)
    - `<side>_days_since_prev_holiday`: days since the previous holiday ended (0 if within a holiday)
    - `<side>_days_since_next_holiday`: negative of days_until_next_holiday
    - `<side>_days_until_prev_holiday`: negative of days_since_prev_holiday

    Holidays are treated as inclusive intervals [startDate, endDate]. If endDate is missing,
    it is treated as equal to startDate.

    Notes:
    - By default this uses nationwide *Public* holidays (OpenHolidays `category='Public'`).
    - Some countries in availabilities may not exist in OpenHolidays; those rows remain missing
      unless `fill_missing_value` is provided.
    """
    if availabilities is None:
        raise TypeError("availabilities must be a pandas DataFrame, got None")
    if holidays is None:
        raise TypeError("holidays must be a pandas DataFrame, got None")

    required_av = {availability_date_col, departure_country_col, destination_country_col}
    missing_av = required_av - set(availabilities.columns)
    if missing_av:
        raise ValueError(f"availabilities is missing required columns: {sorted(missing_av)}")

    required_h = {holidays_country_col, holidays_start_col, holidays_end_col}
    missing_h = required_h - set(holidays.columns)
    if missing_h:
        raise ValueError(f"holidays is missing required columns: {sorted(missing_h)}")

    # Create output columns even for empty input.
    def _ensure_cols(prefix: str):
        for c in [
            f"{prefix}_days_until_next_holiday",
            f"{prefix}_days_since_prev_holiday",
            f"{prefix}_days_since_next_holiday",
            f"{prefix}_days_until_prev_holiday",
        ]:
            if c not in availabilities.columns:
                availabilities[c] = pd.Series(dtype="Int64")

    _ensure_cols("departure_from")
    _ensure_cols("departure_to")

    if availabilities.empty or holidays.empty:
        if fill_missing_value is not None:
            for col in [
                "departure_from_days_until_next_holiday",
                "departure_from_days_since_prev_holiday",
                "departure_from_days_since_next_holiday",
                "departure_from_days_until_prev_holiday",
                "departure_to_days_until_next_holiday",
                "departure_to_days_since_prev_holiday",
                "departure_to_days_since_next_holiday",
                "departure_to_days_until_prev_holiday",
            ]:
                availabilities[col] = availabilities[col].astype("Int64").fillna(fill_missing_value)
        return availabilities

    # Filter to nationwide + selected categories if available.
    h0 = holidays.copy()
    if holidays_nationwide_col in h0.columns:
        # D1 may return booleans as strings (e.g. "True"), so treat common truthy encodings.
        nw = h0[holidays_nationwide_col]
        nw_norm = nw.astype("string").str.strip().str.lower()
        h0 = h0[nw_norm.isin(["true", "1", "t", "yes"])]
    if holidays_category_col in h0.columns and categories:
        h0 = h0[h0[holidays_category_col].astype("string").isin(list(categories))]

    if h0.empty:
        if fill_missing_value is not None:
            for col in [
                "departure_from_days_until_next_holiday",
                "departure_from_days_since_prev_holiday",
                "departure_from_days_since_next_holiday",
                "departure_from_days_until_prev_holiday",
                "departure_to_days_until_next_holiday",
                "departure_to_days_since_prev_holiday",
                "departure_to_days_since_next_holiday",
                "departure_to_days_until_prev_holiday",
            ]:
                availabilities[col] = availabilities[col].astype("Int64").fillna(fill_missing_value)
        return availabilities

    # Normalize & coerce holiday dates.
    h0 = h0.copy()
    h0[holidays_country_col] = _normalize_country_codes(h0[holidays_country_col])
    h0[holidays_start_col] = pd.to_datetime(h0[holidays_start_col], errors="coerce")
    h0[holidays_end_col] = pd.to_datetime(h0[holidays_end_col], errors="coerce")
    h0[holidays_end_col] = h0[holidays_end_col].fillna(h0[holidays_start_col])
    h0 = h0.dropna(subset=[holidays_country_col, holidays_start_col, holidays_end_col])

    # Build per-country interval arrays (day precision).
    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if not h0.empty:
        h0 = h0.sort_values(by=[holidays_country_col, holidays_start_col], kind="mergesort")
        for cc, g in h0.groupby(holidays_country_col, sort=False):
            starts = pd.to_datetime(g[holidays_start_col], errors="coerce").to_numpy(
                dtype="datetime64[D]"
            )
            ends = pd.to_datetime(g[holidays_end_col], errors="coerce").to_numpy(
                dtype="datetime64[D]"
            )
            if len(starts) == 0:
                continue
            intervals[str(cc)] = (starts, ends)

    avail_dates = pd.to_datetime(availabilities[availability_date_col], errors="coerce").to_numpy(
        dtype="datetime64[D]"
    )

    def _compute_for_side(country_series: pd.Series, prefix: str):
        col_until_next = f"{prefix}_days_until_next_holiday"
        col_since_prev = f"{prefix}_days_since_prev_holiday"
        col_since_next = f"{prefix}_days_since_next_holiday"
        col_until_prev = f"{prefix}_days_until_prev_holiday"

        cc_s = _normalize_country_codes(country_series)
        if country_fallback_map:
            cc_s = cc_s.replace(country_fallback_map)

        out_until_next = pd.Series(pd.NA, index=availabilities.index, dtype="Int64")
        out_since_prev = pd.Series(pd.NA, index=availabilities.index, dtype="Int64")

        for cc, idxs in cc_s.groupby(cc_s, dropna=True).groups.items():
            cc_key = str(cc)
            if cc_key not in intervals:
                continue

            starts, ends = intervals[cc_key]
            d = avail_dates[idxs]

            # Next holiday start (>= date)
            pos_next = np.searchsorted(starts, d, side="left")
            next_ok = pos_next < len(starts)

            # Candidate previous holiday (last start <= date)
            pos_prev = np.searchsorted(starts, d, side="right") - 1
            prev_ok = pos_prev >= 0

            in_holiday = np.zeros(len(d), dtype=bool)
            if np.any(prev_ok):
                # Only evaluate end >= date where prev_ok
                in_holiday[prev_ok] = ends[pos_prev[prev_ok]] >= d[prev_ok]

            # Compute until next
            until_next = np.full(len(d), fill_value=np.nan, dtype="float64")
            if np.any(next_ok):
                dn = (starts[pos_next[next_ok]] - d[next_ok]).astype("timedelta64[D]")
                until_next[next_ok] = dn.astype("int64")

            # Compute since prev (based on prev holiday end)
            since_prev = np.full(len(d), fill_value=np.nan, dtype="float64")
            if np.any(prev_ok):
                dp = (d[prev_ok] - ends[pos_prev[prev_ok]]).astype("timedelta64[D]")
                since_prev[prev_ok] = dp.astype("int64")

            # If within a holiday interval, distances are 0
            until_next[in_holiday] = 0.0
            since_prev[in_holiday] = 0.0

            out_until_next.loc[idxs] = pd.Series(until_next, index=idxs, dtype="Int64")
            out_since_prev.loc[idxs] = pd.Series(since_prev, index=idxs, dtype="Int64")

        availabilities[col_until_next] = out_until_next
        availabilities[col_since_prev] = out_since_prev
        availabilities[col_since_next] = (-out_until_next).astype("Int64")
        availabilities[col_until_prev] = (-out_since_prev).astype("Int64")

        if fill_missing_value is not None:
            availabilities[col_until_next] = (
                availabilities[col_until_next].astype("Int64").fillna(fill_missing_value)
            )
            availabilities[col_since_prev] = (
                availabilities[col_since_prev].astype("Int64").fillna(fill_missing_value)
            )
            availabilities[col_since_next] = (
                availabilities[col_since_next].astype("Int64").fillna(fill_missing_value)
            )
            availabilities[col_until_prev] = (
                availabilities[col_until_prev].astype("Int64").fillna(fill_missing_value)
            )

    _compute_for_side(availabilities[departure_country_col], "departure_from")
    _compute_for_side(availabilities[destination_country_col], "departure_to")

    return availabilities


