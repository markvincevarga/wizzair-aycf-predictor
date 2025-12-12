import os
from typing import List, Optional
import pandas as pd
from cloudflare import Cloudflare

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

    def push_new_rows(self, table_name: str, df: pd.DataFrame):
        """
        Push new rows to the database table.
        
        :param table_name: The name of the table.
        :param df: The DataFrame containing rows to insert.
        """
        if df.empty:
            return

        columns = df.columns.tolist()
        num_columns = len(columns)
        if num_columns == 0:
            return
            
        batch_size = max(1, 100 // num_columns)
        
        records = df.to_dict(orient='records')
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            placeholders = []
            all_params = []
            
            for row in batch:
                row_placeholders = []
                for col in columns:
                    val = row[col]
                    row_placeholders.append("?")
                    
                    if val is None or pd.isna(val):
                        all_params.append(None)
                    elif hasattr(val, 'isoformat'):
                        all_params.append(val.isoformat())
                    else:
                        all_params.append(str(val))
                        
                placeholders.append(f"({', '.join(row_placeholders)})")
            
            sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES {', '.join(placeholders)}"
            
            self.query(sql, params=all_params)
