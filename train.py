# %%
from database import DatabaseWrapper
from storage.availabilities import Availabilities
from features.availabilities import (
    build_availabilities_occurs_feature_from_df,
    sort_availabilities_occurs_feature,
)
from features.holidays import add_holiday_distance_features
from features.country_codes import add_country_codes_columns
from features.financial import add_latest_neer_features
from storage.financials import Financials
# %%
DB_NAME = "wizz-aycf"
db = DatabaseWrapper(database_name=DB_NAME)
availabilities = Availabilities(db).get_all()

df = build_availabilities_occurs_feature_from_df(availabilities)
df = sort_availabilities_occurs_feature(df) # just so it's nicer
df.head()
# %%
df = add_country_codes_columns(df)  # necessary for holidays
df = add_holiday_distance_features(df)
df.tail()
# %%
financials = Financials(db).get_all()
df = add_latest_neer_features(df, financials)
df.tail()
# %%
# For running outside of a notebook
print(df.tail())
