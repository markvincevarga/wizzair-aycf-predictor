from __future__ import annotations

import numpy as np
import pandas as pd


def _load_neer_timeseries(
    financials: pd.DataFrame,
    *,
    country_col: str = "REF_AREA",
    date_col: str = "TIME_PERIOD",
    value_col: str = "OBS_VALUE",
) -> pd.DataFrame:
    """
    Normalize NEER time series to a minimal schema:
    [country_col, date_col, value_col], with date_col as pandas datetime64[ns].
    """
    if financials is None:
        raise TypeError("financials must be a pandas DataFrame, got None")

    if financials.empty:
        return pd.DataFrame(
            {
                country_col: pd.Series(dtype="string"),
                date_col: pd.Series(dtype="datetime64[ns]"),
                value_col: pd.Series(dtype="float"),
            }
        )

    missing = [c for c in (country_col, date_col, value_col) if c not in financials.columns]
    if missing:
        raise ValueError(f"financials is missing required columns: {missing}")

    out = financials[[country_col, date_col, value_col]].copy()
    out[country_col] = (
        out[country_col].astype("string").str.strip().str.upper()
    )
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

    # Drop rows without a usable value; for as-of lookups we want the last *observed* value.
    out = out.dropna(subset=[country_col, date_col, value_col])
    out = out.sort_values(by=[country_col, date_col], ascending=True, kind="mergesort").reset_index(
        drop=True
    )
    return out


def add_latest_neer_features(
    availabilities: pd.DataFrame,
    financials: pd.DataFrame,
    *,
    availability_date_col: str = "availability_start",
    departure_country_col: str = "departure_from_country",
    destination_country_col: str = "departure_to_country",
    out_departure_col: str = "departure_from_neer",
    out_destination_col: str = "departure_to_neer",
    out_departure_mean_3w_col: str = "departure_from_neer_mean_3w",
    out_destination_mean_3w_col: str = "departure_to_neer_mean_3w",
    out_departure_mean_6w_col: str = "departure_from_neer_mean_6w",
    out_destination_mean_6w_col: str = "departure_to_neer_mean_6w",
    financial_country_col: str = "REF_AREA",
    financial_date_col: str = "TIME_PERIOD",
    financial_value_col: str = "OBS_VALUE",
    country_fallback_map: dict[str, str] | None = None,
    fill_missing_value: float | None = None,
) -> pd.DataFrame:
    """
    Add NEER features to an existing availabilities dataframe (in-place).

    For each row, for both departure and destination countries:
    - Find the latest financials record where TIME_PERIOD <= availability_start
      (as-of join, direction=backward).
    - Add the corresponding OBS_VALUE as the NEER value.

    This mutates `availabilities` by adding:
    - out_departure_col (default: departure_from_neer)
    - out_destination_col (default: departure_to_neer)
    - out_departure_mean_3w_col (default: departure_from_neer_mean_3w)
    - out_destination_mean_3w_col (default: departure_to_neer_mean_3w)
    - out_departure_mean_6w_col (default: departure_from_neer_mean_6w)
    - out_destination_mean_6w_col (default: departure_to_neer_mean_6w)
    """
    if availabilities is None:
        raise TypeError("availabilities must be a pandas DataFrame, got None")
    if financials is None:
        raise TypeError("financials must be a pandas DataFrame, got None")

    required = {availability_date_col, departure_country_col, destination_country_col}
    missing = required - set(availabilities.columns)
    if missing:
        raise ValueError(f"availabilities is missing required columns: {sorted(missing)}")

    neer = _load_neer_timeseries(
        financials,
        country_col=financial_country_col,
        date_col=financial_date_col,
        value_col=financial_value_col,
    )

    # Ensure output columns exist even for empty frames.
    if availabilities.empty or neer.empty:
        availabilities[out_departure_col] = pd.Series(dtype="Float64")
        availabilities[out_destination_col] = pd.Series(dtype="Float64")
        availabilities[out_departure_mean_3w_col] = pd.Series(dtype="Float64")
        availabilities[out_destination_mean_3w_col] = pd.Series(dtype="Float64")
        availabilities[out_departure_mean_6w_col] = pd.Series(dtype="Float64")
        availabilities[out_destination_mean_6w_col] = pd.Series(dtype="Float64")
        return availabilities

    left_date = pd.to_datetime(availabilities[availability_date_col], errors="coerce")

    # Pre-index NEER by country for fast as-of lookup.
    neer_idx: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cc, g in neer.groupby(financial_country_col, sort=False):
        d = pd.to_datetime(g[financial_date_col], errors="coerce").to_numpy(dtype="datetime64[ns]")
        v = pd.to_numeric(g[financial_value_col], errors="coerce").to_numpy(dtype="float64")
        if len(d) == 0:
            continue
        neer_idx[str(cc)] = (d, v)

    def _asof_neer_for_country(country_series: pd.Series) -> pd.Series:
        out = pd.Series(pd.NA, index=availabilities.index, dtype="Float64")
        cc_s = country_series.astype("string").str.strip().str.upper()

        # Optional mapping for codes not present in BIS series (e.g. euroized countries -> XM).
        # Applied after normalization.
        if country_fallback_map:
            cc_s = cc_s.replace(country_fallback_map)

        for cc, idxs in cc_s.groupby(cc_s, dropna=True).groups.items():
            cc_key = str(cc)
            if cc_key not in neer_idx:
                continue

            right_dates, right_vals = neer_idx[cc_key]
            left_dates = left_date.loc[idxs].to_numpy(dtype="datetime64[ns]")

            pos = np.searchsorted(right_dates, left_dates, side="right") - 1
            ok = pos >= 0
            if not np.any(ok):
                continue

            mapped = np.full(shape=len(left_dates), fill_value=np.nan, dtype="float64")
            mapped[ok] = right_vals[pos[ok]]
            out.loc[idxs] = pd.Series(mapped, index=idxs, dtype="Float64")

        return out

    def _rolling_mean_neer_for_country(
        country_series: pd.Series, window: pd.Timedelta
    ) -> pd.Series:
        """Compute rolling mean NEER over a window ending at availability_start."""
        out = pd.Series(pd.NA, index=availabilities.index, dtype="Float64")
        cc_s = country_series.astype("string").str.strip().str.upper()

        if country_fallback_map:
            cc_s = cc_s.replace(country_fallback_map)

        for cc, idxs in cc_s.groupby(cc_s, dropna=True).groups.items():
            cc_key = str(cc)
            if cc_key not in neer_idx:
                continue

            right_dates, right_vals = neer_idx[cc_key]
            left_dates_arr = left_date.loc[idxs].to_numpy(dtype="datetime64[ns]")
            window_ns = window.value  # Timedelta in nanoseconds

            mapped = np.full(shape=len(left_dates_arr), fill_value=np.nan, dtype="float64")
            for i, ref_date in enumerate(left_dates_arr):
                if pd.isna(ref_date):
                    continue
                start_date = ref_date - window_ns
                # Find observations in (start_date, ref_date]
                mask = (right_dates > start_date) & (right_dates <= ref_date)
                if np.any(mask):
                    mapped[i] = np.nanmean(right_vals[mask])

            out.loc[idxs] = pd.Series(mapped, index=idxs, dtype="Float64")

        return out

    availabilities[out_departure_col] = _asof_neer_for_country(
        availabilities[departure_country_col]
    )
    availabilities[out_destination_col] = _asof_neer_for_country(
        availabilities[destination_country_col]
    )

    # 3-week rolling means
    availabilities[out_departure_mean_3w_col] = _rolling_mean_neer_for_country(
        availabilities[departure_country_col], pd.Timedelta(weeks=3)
    )
    availabilities[out_destination_mean_3w_col] = _rolling_mean_neer_for_country(
        availabilities[destination_country_col], pd.Timedelta(weeks=3)
    )

    # 6-week rolling means
    availabilities[out_departure_mean_6w_col] = _rolling_mean_neer_for_country(
        availabilities[departure_country_col], pd.Timedelta(weeks=6)
    )
    availabilities[out_destination_mean_6w_col] = _rolling_mean_neer_for_country(
        availabilities[destination_country_col], pd.Timedelta(weeks=6)
    )

    if fill_missing_value is not None:
        availabilities[out_departure_col] = (
            availabilities[out_departure_col].astype("Float64").fillna(fill_missing_value)
        )
        availabilities[out_destination_col] = (
            availabilities[out_destination_col].astype("Float64").fillna(fill_missing_value)
        )
        availabilities[out_departure_mean_3w_col] = (
            availabilities[out_departure_mean_3w_col].astype("Float64").fillna(fill_missing_value)
        )
        availabilities[out_destination_mean_3w_col] = (
            availabilities[out_destination_mean_3w_col].astype("Float64").fillna(fill_missing_value)
        )
        availabilities[out_departure_mean_6w_col] = (
            availabilities[out_departure_mean_6w_col].astype("Float64").fillna(fill_missing_value)
        )
        availabilities[out_destination_mean_6w_col] = (
            availabilities[out_destination_mean_6w_col].astype("Float64").fillna(fill_missing_value)
        )

    return availabilities


