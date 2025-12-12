from typing import Optional
import pandas as pd
from datetime import datetime
from database import DatabaseWrapper

class Availabilities:
    def __init__(self, db: DatabaseWrapper):
        self.db = db

    def availability_start_ge(self, start_date: datetime) -> pd.DataFrame:
        """
        Get availabilities starting on or after the given date.
        
        :param start_date: The start date to filter by.
        :return: DataFrame containing matching availabilities.
        """
        timestamp = start_date.timestamp()
        
        sql = "SELECT * FROM availabilities WHERE availability_start >= ?"
        df = self.db.query(sql, [str(timestamp)])
        
        for col in ['availability_start', 'availability_end', 'data_generated']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit='s')
        
        return df

    def latest_data_generated(self) -> Optional[datetime]:
        """
        Get the latest data_generated date as a naive datetime.
        This represents the timestamp of the latest ingested file.
        
        :return: Naive datetime of the latest data_generated, or None if no data.
        """
        sql = "SELECT MAX(data_generated) as latest_gen FROM availabilities"
        df = self.db.query(sql)
        
        if df.empty or pd.isna(df.iloc[0]['latest_gen']):
            return None
            
        latest_val = df.iloc[0]['latest_gen']
        
        try:
            timestamp = float(latest_val)
            dt = pd.to_datetime(timestamp, unit='s')
            return dt.replace(tzinfo=None)
        except (ValueError, TypeError):
            try:
                dt = pd.to_datetime(latest_val)
                return dt.replace(tzinfo=None)
            except Exception:
                return None

    def latest_availability_start(self) -> Optional[datetime]:
        """
        Get the latest availability start date as a naive datetime.
        
        :return: Naive datetime of the latest availability start, or None if no data.
        """
        sql = "SELECT MAX(availability_start) as latest_start FROM availabilities"
        df = self.db.query(sql)
        
        if df.empty or pd.isna(df.iloc[0]['latest_start']):
            return None
            
        latest_val = df.iloc[0]['latest_start']
        
        try:
            timestamp = float(latest_val)
            dt = pd.to_datetime(timestamp, unit='s')
            return dt.replace(tzinfo=None)
        except (ValueError, TypeError):
            try:
                dt = pd.to_datetime(latest_val)
                return dt.replace(tzinfo=None)
            except Exception:
                return None

    def push_new_rows(self, df: pd.DataFrame):
        """
        Push new availabilities to the database.
        
        :param df: DataFrame containing new availabilities.
        """
        df_to_push = df.copy()
        
        for col in ['availability_start', 'availability_end', 'data_generated']:
            if col in df_to_push.columns:
                if pd.api.types.is_datetime64_any_dtype(df_to_push[col]):
                    df_to_push[col] = df_to_push[col].apply(lambda x: x.timestamp() if pd.notnull(x) else None)
                elif df_to_push[col].dtype == 'object':
                     df_to_push[col] = pd.to_datetime(df_to_push[col]).apply(lambda x: x.timestamp() if pd.notnull(x) else None)

        self.db.push_new_rows("availabilities", df_to_push)
