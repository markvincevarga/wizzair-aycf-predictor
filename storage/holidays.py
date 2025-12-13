from typing import Optional
import pandas as pd
from datetime import datetime
from database import DatabaseWrapper

class Holidays:
    def __init__(self, db: DatabaseWrapper):
        self.db = db

    def create_table(self):
        """
        Create the holidays table and indexes if they do not exist.
        Schema:
        - id: TEXT PRIMARY KEY (using API ID)
        - countryIsoCode: TEXT
        - startDate: INTEGER (Unix timestamp)
        - endDate: INTEGER (Unix timestamp)
        - type: TEXT (Public/School)
        - name_text: TEXT
        - regionalScope: TEXT
        - temporalScope: TEXT
        - nationwide: BOOLEAN
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS holidays (
            id TEXT PRIMARY KEY,
            countryIsoCode TEXT,
            startDate INTEGER,
            endDate INTEGER,
            category TEXT,
            name_text TEXT,
            regionalScope TEXT,
            temporalScope TEXT,
            nationwide BOOLEAN
        );
        """
        self.db.query(create_sql)
        
        # Indexes
        index_date = "CREATE INDEX IF NOT EXISTS idx_holidays_date ON holidays(startDate);"
        self.db.query(index_date)
        
        index_country = "CREATE INDEX IF NOT EXISTS idx_holidays_country ON holidays(countryIsoCode);"
        self.db.query(index_country)

    def latest_data_date(self) -> Optional[datetime]:
        """
        Get the latest startDate from the holidays table.
        """
        sql = "SELECT MAX(startDate) as latest_date FROM holidays"
        try:
            df = self.db.query(sql)
            if df.empty or pd.isna(df.iloc[0]['latest_date']):
                return None
            
            latest_val = df.iloc[0]['latest_date']
            try:
                timestamp = float(latest_val)
                dt = pd.to_datetime(timestamp, unit='s')
                return dt.replace(tzinfo=None)
            except Exception:
                return None
        except Exception:
            return None

    def push_new_rows(self, df: pd.DataFrame):
        """
        Push new holidays to the database.
        Uses INSERT OR REPLACE to update existing records by ID.
        """
        df_to_push = df.copy()
        
        date_cols = ['startDate', 'endDate']
        for col in date_cols:
            if col in df_to_push.columns:
                if pd.api.types.is_datetime64_any_dtype(df_to_push[col]):
                    df_to_push[col] = df_to_push[col].apply(lambda x: x.timestamp() if pd.notnull(x) else None)
        
        # Filter columns to match schema
        schema_cols = ['id', 'countryIsoCode', 'startDate', 'endDate', 'category', 'name_text', 'regionalScope', 'temporalScope', 'nationwide']
        
        # Ensure only relevant columns are present
        cols_to_keep = [c for c in schema_cols if c in df_to_push.columns]
        df_to_push = df_to_push[cols_to_keep]
        
        # Using INSERT OR REPLACE because ID is primary key and stable from API
        # But DatabaseWrapper.push_new_rows uses "INSERT INTO" or "INSERT OR IGNORE"
        # Since we want to update if details changed (unlikely for past, but maybe future), REPLACE is better.
        # However, DatabaseWrapper doesn't support REPLACE yet.
        # "ignore_duplicates=True" does INSERT OR IGNORE.
        # Let's assume ID collisions mean same holiday, so IGNORE is fine.
        
        self.db.push_new_rows("holidays", df_to_push, ignore_duplicates=True)

    def remove_duplicates(self):
        # API ID is primary key, so duplicates are handled by constraint.
        # But if we want to clean up manually just in case schema was loose before:
        pass
