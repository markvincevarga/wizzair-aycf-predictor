from typing import Optional
import pandas as pd
from datetime import datetime
from database import DatabaseWrapper

class Availabilities:
    def __init__(self, db: DatabaseWrapper):
        self.db = db

    def create_table(self):
        """
        Create the availabilities table and indexes if they do not exist.
        Schema:
        - id: INTEGER PRIMARY KEY AUTOINCREMENT
        - data_generated: INTEGER (Unix timestamp)
        - departure_from: TEXT
        - departure_to: TEXT
        - availability_start: INTEGER (Unix timestamp)
        - availability_end: INTEGER (Unix timestamp)
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS availabilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_generated INTEGER,
            departure_from TEXT,
            departure_to TEXT,
            availability_start INTEGER,
            availability_end INTEGER
        );
        """
        self.db.query(create_sql)
        
        # Create indexes
        # Index on availability_start for efficient querying by date range
        index_sql = "CREATE INDEX IF NOT EXISTS idx_avail_start ON availabilities(availability_start);"
        self.db.query(index_sql)
        
        # Index on data_generated for incremental ingestion
        index_gen_sql = "CREATE INDEX IF NOT EXISTS idx_data_generated ON availabilities(data_generated);"
        self.db.query(index_gen_sql)

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
        try:
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
        except Exception:
            return None

    def latest_availability_start(self) -> Optional[datetime]:
        """
        Get the latest availability start date as a naive datetime.
        
        :return: Naive datetime of the latest availability start, or None if no data.
        """
        sql = "SELECT MAX(availability_start) as latest_start FROM availabilities"
        try:
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

    def remove_duplicates(self):
        """
        Remove duplicate rows from the availabilities table.
        Duplicates are rows with identical data_generated, departure_from, 
        departure_to, availability_start, and availability_end.
        Keeps the row with the smallest id.
        """
        sql = """
        DELETE FROM availabilities
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM availabilities
            GROUP BY data_generated, departure_from, departure_to, availability_start, availability_end
        )
        """
        self.db.query(sql)
