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

    def get_all(self) -> pd.DataFrame:
        """
        Get all holidays from the database.
        
        :return: DataFrame containing all holidays with correct datetime types.
        """
        sql = "SELECT * FROM holidays"
        df = self.db.query(sql)
        
        date_cols = ['startDate', 'endDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit='s')
        
        return df

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
        
        # Ensure only nationwide rows are pushed (double safety)
        if 'nationwide' in df_to_push.columns:
            df_to_push = df_to_push[df_to_push['nationwide'] == True]
        
        date_cols = ['startDate', 'endDate']
        for col in date_cols:
            if col in df_to_push.columns:
                if pd.api.types.is_datetime64_any_dtype(df_to_push[col]):
                    df_to_push[col] = df_to_push[col].apply(lambda x: x.timestamp() if pd.notnull(x) else None)
        
        schema_cols = ['id', 'countryIsoCode', 'startDate', 'endDate', 'category', 'name_text', 'regionalScope', 'temporalScope', 'nationwide']
        
        cols_to_keep = [c for c in schema_cols if c in df_to_push.columns]
        df_to_push = df_to_push[cols_to_keep]
        
        self.db.push_new_rows("holidays", df_to_push, ignore_duplicates=True)

    def remove_duplicates(self):
        """
        Remove duplicate rows from the holidays table based on content.
        Rows are considered duplicates if they have the same:
        - countryIsoCode
        - startDate
        - endDate
        - category
        - name_text
        
        Keeps one arbitrary row (min(id)) for each group.
        """
        sql = """
        DELETE FROM holidays
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM holidays
            GROUP BY countryIsoCode, startDate, endDate, category, name_text
        )
        """
        self.db.query(sql)
