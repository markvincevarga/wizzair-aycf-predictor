import pandas as pd
from datetime import date, datetime, time
from storage.database import DatabaseWrapper

class Predictions:
    def __init__(self, db: DatabaseWrapper):
        self.db = db

    def create_table(self):
        """
        Create the predictions table and indexes if they do not exist.
        Schema:
        - id: INTEGER PRIMARY KEY AUTOINCREMENT
        - prediction_time: INTEGER (Unix timestamp)
        - availability_start: INTEGER (Unix timestamp)
        - departure_from: TEXT
        - departure_to: TEXT
        - predicted_probability: REAL
        - predicted_available: INTEGER
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_time INTEGER,
            availability_start INTEGER,
            departure_from TEXT,
            departure_to TEXT,
            predicted_probability REAL,
            predicted_available INTEGER
        );
        """
        self.db.query(create_sql)
        
        # Create indexes
        # Index on prediction_time for efficient querying by when prediction was made
        index_pred_time_sql = "CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions(prediction_time);"
        self.db.query(index_pred_time_sql)
        
        # Index on availability_start for efficient querying by date range
        index_avail_start_sql = "CREATE INDEX IF NOT EXISTS idx_pred_avail_start ON predictions(availability_start);"
        self.db.query(index_avail_start_sql)

    def push_new_rows(self, df: pd.DataFrame):
        """
        Push new predictions to the database.
        
        :param df: DataFrame containing new predictions.
        """
        df_to_push = df.copy()
        
        # Ensure timestamps are converted to Unix timestamp
        for col in ['prediction_time', 'availability_start']:
            if col in df_to_push.columns:
                if pd.api.types.is_datetime64_any_dtype(df_to_push[col]):
                    df_to_push[col] = df_to_push[col].apply(lambda x: x.timestamp() if pd.notnull(x) else None)
                elif df_to_push[col].dtype == 'object':
                     df_to_push[col] = pd.to_datetime(df_to_push[col]).apply(lambda x: x.timestamp() if pd.notnull(x) else None)

        self.db.push_new_rows("predictions", df_to_push, ignore_duplicates=False)

    def get_recent_predictions(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Get recent predictions within a specific date range, deduplicated to keep only the latest prediction per route/day.
        
        :param start_date: Start date of the range (inclusive).
        :param end_date: End date of the range (inclusive).
        :return: DataFrame with columns [departure_from, departure_to, availability_start, predicted_probability, predicted_available].
        """
        # Convert dates to timestamps (start of day for start_date, end of day for end_date)
        start_ts = datetime.combine(start_date, time.min).timestamp()
        end_ts = datetime.combine(end_date, time.max).timestamp()
        
        sql = """
        SELECT departure_from, departure_to, availability_start, prediction_time, predicted_probability, predicted_available
        FROM predictions
        WHERE availability_start >= ? AND availability_start <= ?
        """
        
        df = self.db.query(sql, [str(start_ts), str(end_ts)])
        
        if df.empty:
            return pd.DataFrame(columns=['departure_from', 'departure_to', 'availability_start', 'predicted_probability', 'predicted_available'])
            
        # Convert timestamps back to datetime
        df['availability_start'] = pd.to_datetime(df['availability_start'], unit='s')
        
        # Deduplicate: Keep row with max prediction_time for each (departure_from, departure_to, availability_start)
        # Sort by prediction_time descending, then drop duplicates keeping first
        df = df.sort_values('prediction_time', ascending=False)
        df = df.drop_duplicates(subset=['departure_from', 'departure_to', 'availability_start'], keep='first')
        
        # Return relevant columns
        return df[['departure_from', 'departure_to', 'availability_start', 'predicted_probability', 'predicted_available']]

    def get_all_predictions_for_target_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Get all predictions (without deduplication) for targets within a specific date range.
        
        Args:
            start_date: Start date of the target range (inclusive).
            end_date: End date of the target range (inclusive).
            
        Returns:
            DataFrame with columns [departure_from, departure_to, availability_start, 
            prediction_time, predicted_available, predicted_probability].
        """
        start_ts = datetime.combine(start_date, time.min).timestamp()
        end_ts = datetime.combine(end_date, time.max).timestamp()
        
        sql = """
        SELECT departure_from, departure_to, availability_start, prediction_time, predicted_available, predicted_probability
        FROM predictions
        WHERE availability_start >= ? AND availability_start <= ?
        """
        
        df = self.db.query(sql, [str(start_ts), str(end_ts)])
        
        if df.empty:
            return pd.DataFrame(columns=[
                'departure_from', 'departure_to', 'availability_start', 
                'prediction_time', 'predicted_available', 'predicted_probability'
            ])
            
        # Convert timestamps back to datetime
        df['availability_start'] = pd.to_datetime(df['availability_start'], unit='s')
        df['prediction_time'] = pd.to_datetime(df['prediction_time'], unit='s')
        
        return df
