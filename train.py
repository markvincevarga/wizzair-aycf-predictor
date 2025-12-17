# %%
import os
from database import DatabaseWrapper
from storage.availabilities import Availabilities
from features.availabilities import (
    build_labeled_features,
    sort_features,
)
from features.holidays import add_holiday_distance_features
from features.country_codes import add_country_codes_columns
from features.financial import add_latest_neer_features
from storage.financials import Financials
import pandas as pd

# Check whether training data is cached
CACHE_DIR = ".cache/aycf"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAINING_DATA_CACHE = os.path.join(CACHE_DIR, "training_data.csv")

training_data_cached = os.path.exists(TRAINING_DATA_CACHE)
if training_data_cached:
    df = pd.read_csv(TRAINING_DATA_CACHE)
else:
    df = None
# %%
if not training_data_cached:
    DB_NAME = "wizz-aycf"
    db = DatabaseWrapper(database_name=DB_NAME)
    availabilities = Availabilities(db).get_all()

    df = build_labeled_features(availabilities)
    df = sort_features(df) # just so it's nicer
    df.head()
# %%
if not training_data_cached:
    df = add_country_codes_columns(df)  # necessary for holidays
    df = add_holiday_distance_features(df)
    df.tail()
# %%
if not training_data_cached:
    financials = Financials(db).get_all()
    df = add_latest_neer_features(df, financials)

if not training_data_cached:
    df.to_csv(TRAINING_DATA_CACHE, index=False)
df.tail()
# %%
# For running outside of a notebook
print(df.tail())
