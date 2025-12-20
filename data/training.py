import os
import pandas as pd

from storage.database import DatabaseWrapper
from storage.availabilities import Availabilities
from storage.financials import Financials
from features.availabilities import build_labeled_features, sort_features
from features.holidays import add_holiday_distance_features
from features.country_codes import add_country_codes_columns
from features.financial import add_latest_neer_features

DEFAULT_CACHE_DIR = ".cache/aycf"


def get(force_rebuild: bool = False, cache_dir: str = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Load training data from cache or build it from database."""
    cache_path = os.path.join(cache_dir, "training_data.csv")

    if not force_rebuild and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    os.makedirs(cache_dir, exist_ok=True)

    db = DatabaseWrapper(database_name="wizz-aycf")
    availabilities = Availabilities(db).get_all()

    df = build_labeled_features(availabilities)
    df = sort_features(df)
    df = add_country_codes_columns(df)
    df = add_holiday_distance_features(df)

    financials = Financials(db).get_all()
    df = add_latest_neer_features(df, financials)

    df.to_csv(cache_path, index=False)
    return df
