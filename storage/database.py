import os
from typing import List, Optional
import pandas as pd
from cloudflare import Cloudflare
from tqdm import tqdm

class DatabaseWrapper:
    def __init__(self, database_name: str, account_id: str = None, api_token: str = None):
        """
        Initialize the DatabaseWrapper.
        
        :param database_name: The name of the D1 database to connect to.
        :param account_id: Cloudflare Account ID. If not provided, reads from CLOUDFLARE_ACCOUNT_ID env var,
                           or attempts to fetch it using the API token.
        :param api_token: Cloudflare API Token. If not provided, reads from CLOUDFLARE_API_TOKEN env var (handled by SDK).
        """
        self._client = Cloudflare(api_token=api_token)
        self.account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        
        if not self.account_id:
            self.account_id = self._fetch_account_id()
            
        if not self.account_id:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID is required (env or param) or must be discoverable via API token")
            
        self.database_name = database_name
        self.database_id = self._get_database_id()

    def _fetch_account_id(self) -> Optional[str]:
        """
        Attempt to fetch the account ID associated with the API token.
        Returns the first account ID found, or None.
        """
        try:
            accounts = self._client.accounts.list()
            first_account = next(iter(accounts), None)
            if first_account:
                return first_account.id
        except Exception:
            pass
        return None

    def _get_database_id(self) -> str:
        """
        Retrieve the UUID of the database by name.
        """
        response = self._client.d1.database.list(
            account_id=self.account_id,
            name=self.database_name
        )
        
        for db in response:
            if db.name == self.database_name:
                return db.uuid
        
        raise ValueError(f"Database '{self.database_name}' not found")

    def query(self, sql: str, params: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute a SQL query against the D1 database.
        
        :param sql: The SQL query string.
        :param params: Optional list of parameters for the query.
        :return: A pandas DataFrame containing the query results.
        """
        response = self._client.d1.database.query(
            database_id=self.database_id,
            account_id=self.account_id,
            sql=sql,
            params=params or []
        )
        
        all_rows = []
        for query_result in response:
            if query_result.results:
                all_rows.extend(query_result.results)
                
        return pd.DataFrame(all_rows)

    def _normalize_param(self, val):
        """
        Normalize values for D1 parameter binding.
        """
        if val is None or pd.isna(val):
            return None
        if hasattr(val, "isoformat"):
            return val.isoformat()
        return str(val)

    def _batch(self, statements: List[dict]):
        """
        Execute a D1 batch request (multiple statements in one API call).

        Cloudflare's SDK signature has varied; we try a couple common shapes.
        """
        db = self._client.d1.database
        if not hasattr(db, "batch"):
            raise AttributeError("Cloudflare D1 batch API is not available in this SDK")

        # Most likely signature (kwargs include `statements`)
        try:
            return db.batch(
                database_id=self.database_id,
                account_id=self.account_id,
                statements=statements,
            )
        except TypeError:
            pass

        # Alternative signature (single body payload)
        return db.batch(
            database_id=self.database_id,
            account_id=self.account_id,
            body={"statements": statements},
        )

    def push_new_rows(self, table_name: str, df: pd.DataFrame, ignore_duplicates: bool = False):
        """
        Push new rows to the database table with a progress bar.
        
        :param table_name: The name of the table.
        :param df: The DataFrame containing rows to insert.
        :param ignore_duplicates: If True, uses INSERT OR IGNORE to handle duplicate key errors.
        """
        if df.empty:
            return

        columns = df.columns.tolist()
        num_columns = len(columns)
        if num_columns == 0:
            return

        # We want to upload up to 500 rows per API call.
        # D1 has a limit of 100 parameters per *statement*, so we either:
        # - use the D1 batch API (many single-row inserts per call), or
        # - fall back to multi-row VALUES inserts capped by bind limits.
        upload_batch_size = 500

        if num_columns > 100:
            # Each row would exceed the bind limit even as a single-row insert.
            raise ValueError(
                f"Too many columns ({num_columns}) to insert into D1 (max 100 bind params per statement)."
            )

        records = df.to_dict(orient='records')
        
        # Determine SQL statement type
        insert_clause = "INSERT OR IGNORE INTO" if ignore_duplicates else "INSERT INTO"
        
        # Using tqdm for progress bar
        with tqdm(total=len(records), desc=f"Pushing to {table_name}", unit="rows") as pbar:
            # Prefer D1 batch API: 500 single-row statements per request.
            try:
                row_sql = f"{insert_clause} {table_name} ({', '.join(columns)}) VALUES ({', '.join(['?'] * num_columns)})"

                for i in range(0, len(records), upload_batch_size):
                    batch = records[i : i + upload_batch_size]
                    statements = []

                    for row in batch:
                        params = [self._normalize_param(row[col]) for col in columns]
                        statements.append({"sql": row_sql, "params": params})

                    self._batch(statements)
                    pbar.update(len(batch))

                return
            except Exception:
                # If batch isn't supported (or fails), fall back to the safe multi-row insert approach.
                pass

            # Fallback: multi-row VALUES insert capped by bind limits (100 params/query).
            batch_size = max(1, 100 // num_columns)
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]

                placeholders = []
                all_params = []

                for row in batch:
                    placeholders.append(f"({', '.join(['?'] * num_columns)})")
                    for col in columns:
                        all_params.append(self._normalize_param(row[col]))

                sql = f"{insert_clause} {table_name} ({', '.join(columns)}) VALUES {', '.join(placeholders)}"
                self.query(sql, params=all_params)
                pbar.update(len(batch))

