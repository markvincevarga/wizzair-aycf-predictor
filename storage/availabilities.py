from typing import Optional
import pandas as pd
from datetime import datetime, date, time
from storage.database import DatabaseWrapper

CITY_TO_COUNTRY_ISO: dict[str, str] = {
    "Aalesund": "NO",
    "Aberdeen": "GB",
    "Abu Dhabi": "AE",
    "Agadir": "MA",
    "Alexandria": "EG",
    "Alghero": "IT",
    "Alicante": "ES",
    "Almaty": "KZ",
    "Amman": "JO",
    "Ancona": "IT",
    "Antalya": "TR",
    "Asyut": "EG",
    "Athens": "GR",
    "Bacau": "RO",
    "Baku": "AZ",
    "Banja Luka": "BA",
    "Barcelona": "ES",
    "Bari": "IT",
    "Basel/Mulhouse": "FR",  # Airport is physically in France
    "Beirut": "LB",
    "Belgrade": "RS",
    "Bergen": "NO",
    "Berlin": "DE",
    "Bilbao": "ES",
    "Billund": "DK",
    "Birmingham": "GB",
    "Bishkek": "KG",
    "Bologna": "IT",
    "Bordeaux": "FR",
    "Brasov": "RO",
    "Bratislava": "SK",
    "Brussels": "BE",
    "Bucharest": "RO",
    "Budapest": "HU",
    "Burgas": "BG",
    "Castellon": "ES",
    "Catania": "IT",
    "Chania": "GR",
    "Chisinau": "MD",
    "Cluj": "RO",
    "Comiso": "IT",
    "Constanta": "RO",
    "Copenhagen": "DK",
    "Craiova": "RO",
    "Dalaman": "TR",
    "Dammam": "SA",
    "Debrecen": "HU",
    "Dortmund": "DE",
    "Dubai": "AE",
    "Dubrovnik": "HR",
    "Eindhoven": "NL",
    "Faro": "PT",
    "Frankfurt": "DE",
    "Friedrichshafen": "DE",
    "Fuerteventura": "ES",
    "Gabala": "AZ",
    "Gdansk": "PL",
    "Genoa": "IT",
    "Girona": "ES",
    "Giza": "EG",
    "Glasgow": "GB",
    "Gothenburg": "SE",
    "Gran Canaria": "ES",
    "Gyumri": "AM",
    "Hamburg": "DE",
    "Haugesund": "NO",
    "Heraklion": "GR",
    "Hurghada": "EG",
    "Iasi": "RO",
    "Ibiza": "ES",
    "Istanbul": "TR",
    "Jeddah": "SA",
    "Karlsruhe/Baden-Baden": "DE",
    "Katowice": "PL",
    "Kaunas": "LT",
    "Kerkyra": "GR",
    "Klaipeda/Palanga": "LT",
    "Kosice": "SK",
    "Krakow": "PL",
    "Kutaisi": "GE",
    "Lamezia Terme": "IT",
    "Larnaca": "CY",
    "Leeds/Bradford": "GB",
    "Leipzig/Halle": "DE",
    "Lisbon": "PT",
    "Liverpool": "GB",
    "Ljubljana": "SI",
    "London": "GB",
    "Lublin": "PL",
    "Lyon": "FR",
    "Maastricht": "NL",
    "Madeira": "PT",
    "Madinah": "SA",
    "Madrid": "ES",
    "Malaga": "ES",
    "Male": "MV",
    "Malmo": "SE",
    "Malta": "MT",
    "Marrakech": "MA",
    "Marsa Alam": "EG",
    "Memmingen": "DE",
    "Milan": "IT",
    "Mykonos": "GR",
    "Naples": "IT",
    "Nice": "FR",
    "Nis": "RS",
    "Nur-Sultan": "KZ",
    "Nuremberg": "DE",
    "Ohrid": "MK",
    "Olbia": "IT",
    "Oslo": "NO",
    "Palma De Mallorca": "ES",
    "Paphos": "CY",
    "Paris": "FR",
    "Perugia": "IT",
    "Pescara": "IT",
    "Pisa": "IT",
    "Plovdiv": "BG",
    "Podgorica": "ME",
    "Poprad/Tatry": "SK",
    "Porto": "PT",
    "Poznan": "PL",
    "Prague": "CZ",
    "Pristina": "XK",  # Temporary country code for Kosovo used by most aviation data
    "Radom": "PL",
    "Reykjavik": "IS",
    "Rhodes": "GR",
    "Riga": "LV",
    "Rimini": "IT",
    "Riyadh": "SA",
    "Rome": "IT",
    "Rzeszow": "PL",
    "Salalah": "OM",
    "Salerno": "IT",
    "Salzburg": "AT",
    "Samarkand": "UZ",
    "Santander": "ES",
    "Santorini": "GR",
    "Sarajevo": "BA",
    "Satu Mare": "RO",
    "Sevilla": "ES",
    "Sharm el-Sheikh": "EG",
    "Sibiu": "RO",
    "Skiathos": "GR",
    "Skopje": "MK",
    "Sofia": "BG",
    "Sohag": "EG",
    "Split": "HR",
    "Stavanger": "NO",
    "Stockholm": "SE",
    "Stuttgart": "DE",
    "Suceava": "RO",
    "Szczecin": "PL",
    "Szczytno": "PL",
    "Tallinn": "EE",
    "Targu-Mures": "RO",
    "Tashkent": "UZ",
    "Tel Aviv": "IL",
    "Tenerife": "ES",
    "Thessaloniki": "GR",
    "Timisoara": "RO",
    "Tirana": "AL",
    "Trieste": "IT",
    "Tromso": "NO",
    "Trondheim": "NO",
    "Turin": "IT",
    "Turkistan": "KZ",
    "Turku": "FI",
    "Tuzla": "BA",
    "Valencia": "ES",
    "Varna": "BG",
    "Venice": "IT",
    "Verona": "IT",
    "Vienna": "AT",
    "Vilnius": "LT",
    "Warsaw": "PL",
    "Wroclaw": "PL",
    "Yerevan": "AM",
    "Zakinthos Island": "GR",
    "Zaragoza": "ES",
}


def _add_country_codes_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if df.empty:
        df["departure_from_country"] = pd.Series(dtype="string")
        df["departure_to_country"] = pd.Series(dtype="string")
        return df

    mapping: dict[str, str] = {}
    mapping.update({k.strip(): v for k, v in CITY_TO_COUNTRY_ISO.items()})

    if "departure_from" in df.columns:
        s_from = df["departure_from"].astype("string").str.strip()
        df["departure_from_country"] = s_from.map(mapping)

    if "departure_to" in df.columns:
        s_to = df["departure_to"].astype("string").str.strip()
        df["departure_to_country"] = s_to.map(mapping)

    return df

class Availabilities:
    def __init__(self, db: DatabaseWrapper):
        self.db = db

    @property
    def routes(self) -> pd.DataFrame:
        """
        Distinct route list currently present in the database.

        Returns a dataframe with:
        - departure_from, departure_to
        - departure_from_country, departure_to_country
        """
        df = self.db.query("SELECT DISTINCT departure_from, departure_to FROM availabilities")
        return _add_country_codes_columns(df)

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
        
        # Create unique index to prevent duplicates
        unique_index_sql = """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_avail_unique ON availabilities(data_generated, departure_from, departure_to, availability_start, availability_end);
        """
        self.db.query(unique_index_sql)

    def get_all(
        self,
        include_country_codes: bool = False,
    ) -> pd.DataFrame:
        """
        Get all availabilities from the database.
        
        :return: DataFrame containing all availabilities with correct datetime types.
        """
        sql = "SELECT * FROM availabilities"
        df = self.db.query(sql)
        
        for col in ['availability_start', 'availability_end', 'data_generated']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit='s')

        if include_country_codes:
            df = _add_country_codes_columns(df)

        return df

    def availability_start_ge(
        self,
        start_date: datetime,
        include_country_codes: bool = False,
    ) -> pd.DataFrame:
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

        if include_country_codes:
            df = _add_country_codes_columns(df)

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

        self.db.push_new_rows("availabilities", df_to_push, ignore_duplicates=True)

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

    def get_recent_availabilities(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Get availabilities within a specific date range.
        
        :param start_date: Start date of the range (inclusive).
        :param end_date: End date of the range (inclusive).
        :return: DataFrame with columns [departure_from, departure_to, availability_start, is_available].
        """
        start_ts = datetime.combine(start_date, time.min).timestamp()
        end_ts = datetime.combine(end_date, time.max).timestamp()
        
        sql = """
        SELECT departure_from, departure_to, availability_start
        FROM availabilities
        WHERE availability_start >= ? AND availability_start <= ?
        """
        
        df = self.db.query(sql, [str(start_ts), str(end_ts)])
        
        if df.empty:
            return pd.DataFrame(columns=['departure_from', 'departure_to', 'availability_start', 'is_available'])
            
        df['availability_start'] = pd.to_datetime(df['availability_start'], unit='s')
        df['is_available'] = True
        
        # Deduplicate just in case multiple records for same route/day exist (though unexpected with unique index)
        df = df.drop_duplicates(subset=['departure_from', 'departure_to', 'availability_start'])
        
        return df[['departure_from', 'departure_to', 'availability_start', 'is_available']]
