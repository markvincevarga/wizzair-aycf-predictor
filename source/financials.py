import pandas as pd
from datetime import datetime
from typing import Optional
from config import PROJECT_START_DATE

def get_financials(after: Optional[datetime] = None) -> pd.DataFrame:
    """
    Fetch Nominal Effective Exchange Rate (NEER) data from BIS.
    Filters for:
    - Frequency: Monthly (M)
    - Type: Real (R)
    - Basket: Narrow (N)
    
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
        # The URL structure .../M.R.N implies Frequency=M, Type=R, Basket=N.
        # We keep local filtering for robustness.
        base_url = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/M.R.N"
        params = {"format": "csv"}
        
        # Determine start period
        # If after is provided, format YYYY-MM
        start_period = after.strftime("%Y-%m")
        params["startPeriod"] = start_period
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{base_url}?{query_string}"
        
        df = pd.read_csv(full_url)
        
        if df.empty:
            return pd.DataFrame()

        # Local filtering for safety
        if 'FREQ' in df.columns:
            df = df[df['FREQ'] == 'M']

        if 'EER_TYPE' in df.columns:
            df = df[df['EER_TYPE'] == 'R']
            
        if 'EER_BASKET' in df.columns:
            df = df[df['EER_BASKET'] == 'N']
            
        date_col = 'TIME_PERIOD'
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m')
            
        return df

    except Exception as e:
        print(f"Error fetching financials: {e}")
        return pd.DataFrame()
