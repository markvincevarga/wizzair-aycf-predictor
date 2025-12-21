import pandas as pd
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

