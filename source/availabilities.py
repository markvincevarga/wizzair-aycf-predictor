import io
import os
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import PROJECT_START_DATE

AVAILABILITIES_REPOSITORY_URL = "https://github.com/markvincevarga/wizzair-aycf-availability.git"
AVAILABILITIES_REPOSITORY_DIR = "data"
DEFAULT_TIMEOUT = 10  # seconds


def _get_session() -> requests.Session:
    """
    Create a requests Session with retry logic.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_availabilities(after: Optional[datetime] = None) -> pd.DataFrame:
    """
    Fetch availabilities from the GitHub repository.
    Only downloads files generated after the specified timestamp.
    Defaults to PROJECT_START_DATE if after is None.
    
    Args:
        after (datetime, optional): Filter for files generated after this timestamp. 
                                   If None, uses PROJECT_START_DATE.
    
    Returns:
        pd.DataFrame: Concatenated dataframe of all downloaded availabilities.
        Columns 'availability_start', 'availability_end', 'data_generated' are datetime objects.
    """
    if after is None:
        after = PROJECT_START_DATE

    parsed_url = urlparse(AVAILABILITIES_REPOSITORY_URL)
    path_parts = parsed_url.path.strip("/").replace(".git", "").split("/")
    if len(path_parts) < 2:
        raise ValueError(f"Invalid repository URL: {AVAILABILITIES_REPOSITORY_URL}")
    
    owner = path_parts[-2]
    repo = path_parts[-1]

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{AVAILABILITIES_REPOSITORY_DIR}"

    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    session = _get_session()

    try:
        response = session.get(api_url, headers=headers, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        files_info = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file list from GitHub: {e}")
        return pd.DataFrame()

    dataframes = []
    
    for file_info in files_info:
        if not isinstance(file_info, dict) or file_info.get("type") != "file" or not file_info.get("name", "").endswith(".csv"):
            continue
            
        filename = file_info["name"]
        
        try:
            name_without_ext = filename[:-4]
            clean_timestamp = name_without_ext.replace("_", ":")
            file_date = pd.to_datetime(clean_timestamp)
            
            # Filter strictly > after (or >= if project start is inclusive)
            # The user said "do not retrieve any data from before then".
            # Usually 'after' implies strict >, but project start date is usually inclusive.
            # Let's use >= for project start, > for incremental updates.
            # If after was passed explicitly, we usually want >.
            # If it fell back to PROJECT_START_DATE, we might want >=.
            # To be safe and consistent with "from before then", we exclude < PROJECT_START_DATE.
            # So >= is correct.
            
            # Normalize timezones
            if after.tzinfo is not None and file_date.tzinfo is None:
                file_date = file_date.tz_localize("UTC")
            elif after.tzinfo is None and file_date.tzinfo is not None:
                 file_date = file_date.tz_localize(None)
            
            if file_date <= after:
                continue
        except (ValueError, pd.errors.ParserError):
            print(f"Skipping file with unparseable timestamp: {filename}")
            continue

        download_url = file_info.get("download_url")
        if not download_url:
            continue

        try:
            file_response = session.get(download_url, headers=headers, timeout=DEFAULT_TIMEOUT)
            file_response.raise_for_status()
            
            csv_content = io.StringIO(file_response.text)
            df = pd.read_csv(csv_content)
            
            date_columns = ['availability_start', 'availability_end', 'data_generated']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    if not dataframes:
        return pd.DataFrame()

    return pd.concat(dataframes, ignore_index=True)
