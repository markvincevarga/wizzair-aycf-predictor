import pandas as pd
from datetime import datetime
from datetime import timedelta
from typing import Optional
from config import PROJECT_START_DATE

def get_financials(after: Optional[datetime] = None) -> pd.DataFrame:
    """
    Fetch Nominal Effective Exchange Rate (NEER) data from BIS.
    Filters for:
    - Frequency: Daily (D)
    - Type: Nominal (N)
    - Basket: Broad (B)
    
    Args:
        after (datetime, optional): Filter for data generated after this timestamp. 
                                   If None, uses PROJECT_START_DATE.
                                   Note: BIS API filters by month (YYYY-MM).

    Returns:
        pd.DataFrame: DataFrame containing filtered NEER data.
        'TIME_PERIOD' column is datetime objects.
    """
    if after is None:
        after = PROJECT_START_DATE

    try:
        # The URL structure .../D.N.B implies Frequency=D, Type=N, Basket=B.
        # We keep local filtering for robustness.
        base_url = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/D.N.B"
        params = {"format": "csv"}
        
        # Determine start period
        # For as-of feature joins you often need a small lookback buffer so the
        # first days in a window can still resolve to a prior observation.
        start_period_dt = after - timedelta(days=31)
        start_period = start_period_dt.strftime("%Y-%m-%d")
        params["startPeriod"] = start_period
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{base_url}?{query_string}"
        
        df = pd.read_csv(full_url)
        
        if df.empty:
            return pd.DataFrame()

        # Local filtering for safety
        if 'FREQ' in df.columns:
            df = df[df['FREQ'] == 'D']

        if 'EER_TYPE' in df.columns:
            df = df[df['EER_TYPE'] == 'N']
            
        if 'EER_BASKET' in df.columns:
            df = df[df['EER_BASKET'] == 'B']
            
        date_col = 'TIME_PERIOD'
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')
            
        return df

    except Exception as e:
        print(f"Error fetching financials: {e}")
        return pd.DataFrame()
