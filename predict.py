# %%
from features.availabilities import (
    build_labeled_features,
    extract_routes,
    sort_features,
    build_route_date_grid,
)
from storage.availabilities import Availabilities
from storage.database import DatabaseWrapper
from datetime import timedelta
import pandas as pd
from features.financial import add_latest_neer_features
from features.holidays import add_holiday_distance_features
from features.country_codes import add_country_codes_columns
from storage.financials import Financials
# %%
# Get historical data from db
DB_NAME = "wizz-aycf"
db = DatabaseWrapper(database_name=DB_NAME)

# Only need the last 14 days to build rolling average + lagged features
availabilities_start = Availabilities(db).latest_availability_start() - timedelta(days=14)
availabilities = Availabilities(db).availability_start_ge(start_date=availabilities_start)

historical_availabilities = build_labeled_features(availabilities)
historical_availabilities = sort_features(historical_availabilities)
historical_availabilities.tail()
# %%
# Generate future possibilities
start = Availabilities(db).latest_availability_start() + timedelta(days=1)
end = start + timedelta(days=7)
routes = extract_routes(historical_availabilities)
future_availabilities = build_route_date_grid(routes, start, end)
future_availabilities.tail()

# %%
# Concatenate and add generated features
all_availabilities = pd.concat([historical_availabilities, future_availabilities], ignore_index=True)
all_availabilities = add_country_codes_columns(all_availabilities)
all_availabilities = add_holiday_distance_features(all_availabilities)

financials = Financials(db).get_all()
all_availabilities = add_latest_neer_features(all_availabilities, financials)
all_availabilities.head()
 # %%
print(all_availabilities.head())
print(all_availabilities.tail())
