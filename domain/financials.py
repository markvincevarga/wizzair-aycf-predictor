from typing import Optional
import pandas as pd
from datetime import datetime
from database import DatabaseWrapper

class Financials:
    def __init__(self, db: DatabaseWrapper):
        self.db = db

    def create_table(self):
        """
        Create the financials table and indexes if they do not exist.
        Schema inferred from BIS source:
        - FREQ: TEXT
        - EER_TYPE: TEXT
        - EER_BASKET: TEXT
        - REF_AREA: TEXT
        - UNIT_MEASURE: INTEGER
        - TIME_FORMAT: INTEGER
        - COLLECTION: TEXT
        - TITLE_TS: TEXT
        - TIME_PERIOD: INTEGER (Unix timestamp)
        - OBS_VALUE: REAL
        - OBS_STATUS: TEXT
        - OBS_CONF: TEXT
        - OBS_PRE_BREAK: REAL
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS financials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            FREQ TEXT,
            EER_TYPE TEXT,
            EER_BASKET TEXT,
            REF_AREA TEXT,
            UNIT_MEASURE INTEGER,
            TIME_FORMAT INTEGER,
            COLLECTION TEXT,
            TITLE_TS TEXT,
            TIME_PERIOD INTEGER,
            OBS_VALUE REAL,
            OBS_STATUS TEXT,
            OBS_CONF TEXT,
            OBS_PRE_BREAK REAL
        );
        """
        self.db.query(create_sql)
        
        # Create index on TIME_PERIOD
        index_sql = "CREATE INDEX IF NOT EXISTS idx_fin_time_period ON financials(TIME_PERIOD);"
        self.db.query(index_sql)
        
        # Create unique index to prevent duplicates
        unique_index_sql = """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_fin_unique ON financials(TIME_PERIOD, REF_AREA, FREQ, EER_TYPE, EER_BASKET);
        """
        self.db.query(unique_index_sql)

    def latest_data_date(self) -> Optional[datetime]:
        """
        Get the latest date from the financials table (TIME_PERIOD).
        
        :return: Naive datetime of the latest record, or None if no data.
        """
        sql = "SELECT MAX(TIME_PERIOD) as latest_date FROM financials"
        try:
            df = self.db.query(sql)
            
            if df.empty or pd.isna(df.iloc[0]['latest_date']):
                return None
                
            latest_val = df.iloc[0]['latest_date']
            
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
            # Table might not exist yet
            return None

    def push_new_rows(self, df: pd.DataFrame):
        """
        Push new financials to the database.
        
        :param df: DataFrame containing new financials.
        """
        df_to_push = df.copy()
        
        date_cols = ['TIME_PERIOD']
        
        for col in date_cols:
            if col in df_to_push.columns:
                if pd.api.types.is_datetime64_any_dtype(df_to_push[col]):
                    df_to_push[col] = df_to_push[col].apply(lambda x: x.timestamp() if pd.notnull(x) else None)
                elif df_to_push[col].dtype == 'object':
                     df_to_push[col] = pd.to_datetime(df_to_push[col]).apply(lambda x: x.timestamp() if pd.notnull(x) else None)

        # Use ignore_duplicates=True to skip rows that violate the unique index
        self.db.push_new_rows("financials", df_to_push, ignore_duplicates=True)

    def remove_duplicates(self):
        """
        Remove duplicate rows from the financials table.
        Duplicates are rows with identical TIME_PERIOD, REF_AREA, FREQ, EER_TYPE, EER_BASKET.
        Keeps the row with the smallest id.
        """
        sql = """
        DELETE FROM financials
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM financials
            GROUP BY TIME_PERIOD, REF_AREA, FREQ, EER_TYPE, EER_BASKET
        )
        """
        self.db.query(sql)
