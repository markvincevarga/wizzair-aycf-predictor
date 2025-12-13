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
        Filters out existing rows based on unique key before pushing.
        
        :param df: DataFrame containing new financials.
        """
        if df.empty:
            return

        df_to_push = df.copy()
        
        # Convert date columns to timestamps
        date_cols = ['TIME_PERIOD']
        for col in date_cols:
            if col in df_to_push.columns:
                if pd.api.types.is_datetime64_any_dtype(df_to_push[col]):
                    df_to_push[col] = df_to_push[col].apply(lambda x: x.timestamp() if pd.notnull(x) else None)
                elif df_to_push[col].dtype == 'object':
                     df_to_push[col] = pd.to_datetime(df_to_push[col]).apply(lambda x: x.timestamp() if pd.notnull(x) else None)

        # Pre-filter existing rows to avoid unnecessary INSERT attempts
        if not df_to_push.empty:
            # Check against DB
            min_time = df_to_push['TIME_PERIOD'].min()
            max_time = df_to_push['TIME_PERIOD'].max()
            
            if pd.notnull(min_time) and pd.notnull(max_time):
                query_sql = """
                SELECT TIME_PERIOD, REF_AREA, FREQ, EER_TYPE, EER_BASKET 
                FROM financials 
                WHERE TIME_PERIOD >= ? AND TIME_PERIOD <= ?
                """
                existing_df = self.db.query(query_sql, [str(min_time), str(max_time)])
                
                if not existing_df.empty:
                    # Filter df_to_push
                    keys_col = ['TIME_PERIOD', 'REF_AREA', 'FREQ', 'EER_TYPE', 'EER_BASKET']
                    
                    # Merge df_to_push with existing_df on keys
                    # Left join, keep only those where existing is null (not found)
                    merged = df_to_push.merge(
                        existing_df, 
                        on=keys_col, 
                        how='left', 
                        indicator=True
                    )
                    
                    # Keep rows that are 'left_only'
                    df_to_push = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
                    
        if df_to_push.empty:
            print("All rows already exist in database. Skipping push.")
            return

        # Use ignore_duplicates=True as a fallback safety net
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
