from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable, Optional

import pandas as pd
import holidays as pyholidays


def _kosovo_fixed_holidays(year: int) -> dict[date, str]:
    """
    Minimal Kosovo (XK) public holiday set (fixed-date only).
    Variable-date holidays (Easter/Eid) are intentionally excluded.
    """
    return {
        date(year, 1, 1): "New Year's Day",
        date(year, 1, 7): "Orthodox Christmas Day",
        date(year, 2, 17): "Independence Day",
        date(year, 4, 9): "Constitution Day",
        date(year, 5, 1): "Labour Day",
        date(year, 5, 9): "Europe Day",
        date(year, 12, 25): "Christmas Day (Catholic)",
    }


def get_holidays(
    *,
    country_codes: Iterable[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Generate a holiday dataframe using python-holidays for the given country codes and date range.

    Output schema matches what our feature adder expects:
    - countryIsoCode, startDate, endDate, category, name_text, nationwide
    """
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    years = list(range(start_date.year - 1, end_date.year + 2))

    rows: list[dict[str, object]] = []
    for cc0 in country_codes:
        if cc0 is None:
            continue
        cc = str(cc0).strip().upper()
        if not cc:
            continue

        # Kosovo (XK) is not supported by python-holidays; we include a minimal fixed list.
        if cc == "XK":
            for y in years:
                for d, name in _kosovo_fixed_holidays(y).items():
                    if d < start_date or d > end_date:
                        continue
                    rows.append(
                        {
                            "id": f"XK:{d.isoformat()}",
                            "countryIsoCode": "XK",
                            "startDate": pd.Timestamp(d),
                            "endDate": pd.Timestamp(d),
                            "category": "Public",
                            "name_text": name,
                            "nationwide": True,
                        }
                    )
            continue

        try:
            cal = pyholidays.country_holidays(cc, years=years)
        except Exception:
            # Unknown/unsupported country code in python-holidays.
            continue

        for d, name in cal.items():
            if isinstance(d, datetime):
                d0 = d.date()
            else:
                d0 = d
            if d0 < start_date or d0 > end_date:
                continue
            rows.append(
                {
                    "id": f"{cc}:{d0.isoformat()}",
                    "countryIsoCode": cc,
                    "startDate": pd.Timestamp(d0),
                    "endDate": pd.Timestamp(d0),
                    "category": "Public",
                    "name_text": str(name),
                    "nationwide": True,
                }
            )

    if not rows:
        return pd.DataFrame(
            {
                "id": pd.Series(dtype="string"),
                "countryIsoCode": pd.Series(dtype="string"),
                "startDate": pd.Series(dtype="datetime64[ns]"),
                "endDate": pd.Series(dtype="datetime64[ns]"),
                "category": pd.Series(dtype="string"),
                "name_text": pd.Series(dtype="string"),
                "nationwide": pd.Series(dtype="object"),
            }
        )

    df = pd.DataFrame(rows)
    # keep stable order
    df["countryIsoCode"] = df["countryIsoCode"].astype("string").str.strip().str.upper()
    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
    df["endDate"] = pd.to_datetime(df["endDate"], errors="coerce")
    df = df.dropna(subset=["countryIsoCode", "startDate", "endDate"])
    df = df.sort_values(by=["countryIsoCode", "startDate"], kind="mergesort").reset_index(drop=True)
    return df
