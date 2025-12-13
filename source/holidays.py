import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List

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
    
    Args:
        after (datetime, optional): Filter for holidays starting after this date.
                                   If None, defaults to current year start.
    
    Returns:
        pd.DataFrame: Consolidated DataFrame of holidays.
    """
    # Determine date range
    if after:
        start_date = after.strftime("%Y-%m-%d")
        # Fetch up to 2 years in advance? Or just 1 year?
        # Availabilities go up to Dec 2025. 
        # Let's fetch 2 years from start date to be safe.
        end_dt = after + timedelta(days=730)
        end_date = end_dt.strftime("%Y-%m-%d")
    else:
        # Default to start of current year
        now = datetime.now()
        start_date = f"{now.year}-01-01"
        end_date = f"{now.year + 2}-12-31"

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
            item['countryIsoCode'] = iso
            item['category'] = 'Public'
            all_holidays.append(item)
            
        # School Holidays
        sh_data = _fetch_data("SchoolHolidays", iso, start_date, end_date)
        for item in sh_data:
            item['countryIsoCode'] = iso
            item['category'] = 'School'
            all_holidays.append(item)

    if not all_holidays:
        return pd.DataFrame()

    df = pd.DataFrame(all_holidays)
    
    # Normalize columns
    # API returns: id, startDate, endDate, type, name (list of dicts), regionalScope, etc.
    # We want to extract English name if possible.
    
    def get_name(name_list):
        if not isinstance(name_list, list):
            return str(name_list)
        for n in name_list:
            if n.get('language') == 'EN':
                return n.get('text')
        # Fallback to first
        return name_list[0].get('text') if name_list else None

    if 'name' in df.columns:
        df['name_text'] = df['name'].apply(get_name)
        
    # Ensure dates are datetime
    for col in ['startDate', 'endDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            
    # Filter strictly by 'after' if needed (API is validFrom/To inclusive)
    if after and 'startDate' in df.columns:
        df = df[df['startDate'] > after]

    return df

