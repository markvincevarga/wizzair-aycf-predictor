import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
from config import PROJECT_START_DATE

BASE_URL = "https://openholidaysapi.org"

def _fetch_data(endpoint: str, country_code: str, start_date: str, end_date: str) -> List[dict]:
    try:
        resp = requests.get(
            f"{BASE_URL}/{endpoint}",
            params={
                "countryIsoCode": country_code,
                "validFrom": start_date,
                "validTo": end_date,
                "languageIsoCode": "EN" # Prefer English names
            }
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error fetching {endpoint} for {country_code}: {e}")
    return []

def get_holidays(after: Optional[datetime] = None) -> pd.DataFrame:
    """
    Fetch Public and School holidays for all supported countries.
    Filters for nationwide holidays only.
    
    Args:
        after (datetime, optional): Filter for holidays starting after this date.
                                   If None, uses PROJECT_START_DATE.
    
    Returns:
        pd.DataFrame: Consolidated DataFrame of holidays.
    """
    if after is None:
        after = PROJECT_START_DATE

    # Define the maximum future horizon we care about (e.g., 1 month from today)
    # We don't want to infinitely fetch into the future.
    now = datetime.now()
    max_horizon = now + timedelta(days=30)
    
    # If the database is already up to date beyond our horizon, don't fetch more.
    if after >= max_horizon:
        print(f"Database is up to date until {after.date()}, which is beyond the fetch horizon ({max_horizon.date()}). Skipping fetch.")
        return pd.DataFrame()

    # Determine date range
    start_date = after.strftime("%Y-%m-%d")
    end_date = max_horizon.strftime("%Y-%m-%d")

    print(f"Fetching holidays from {start_date} to {end_date}...")

    # 1. Get Countries
    countries = []
    try:
        resp = requests.get(f"{BASE_URL}/Countries")
        if resp.status_code == 200:
            countries = resp.json()
    except Exception as e:
        print(f"Error fetching countries: {e}")
        return pd.DataFrame()

    all_holidays = []

    for country in countries:
        iso = country['isoCode']
        
        # Public Holidays
        ph_data = _fetch_data("PublicHolidays", iso, start_date, end_date)
        for item in ph_data:
            # Filter for nationwide
            if item.get('nationwide', False):
                item['countryIsoCode'] = iso
                item['category'] = 'Public'
                all_holidays.append(item)
            
        # School Holidays
        sh_data = _fetch_data("SchoolHolidays", iso, start_date, end_date)
        for item in sh_data:
            # Filter for nationwide
            if item.get('nationwide', False):
                item['countryIsoCode'] = iso
                item['category'] = 'School'
                all_holidays.append(item)

    if not all_holidays:
        return pd.DataFrame()

    df = pd.DataFrame(all_holidays)
    
    # Normalize columns
    def get_name(name_list):
        if not isinstance(name_list, list):
            return str(name_list)
        for n in name_list:
            if n.get('language') == 'EN':
                return n.get('text')
        return name_list[0].get('text') if name_list else None

    if 'name' in df.columns:
        df['name_text'] = df['name'].apply(get_name)
        
    # Ensure dates are datetime
    for col in ['startDate', 'endDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            
    if after and 'startDate' in df.columns:
        df = df[df['startDate'] >= after]

    return df
