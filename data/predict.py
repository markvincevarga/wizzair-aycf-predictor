import pandas as pd
import numpy as np
from datetime import timedelta, datetime, date
from typing import Optional, Union

from storage.database import DatabaseWrapper
from storage.availabilities import Availabilities
from storage.financials import Financials
from features.availabilities import (
    build_labeled_features,
    build_route_date_grid,
    sort_features,
    add_derived_features,
)
from features.holidays import add_holiday_distance_features
from features.country_codes import add_country_codes_columns
from features.financial import add_latest_neer_features
from features.encoding import add_label_encoded_columns


class PredictionSession:
    """
    Manages state for sequential, day-by-day prediction.
    Keeps track of history and updates it with predictions to allow
    generating features for subsequent days (autoregressive).
    """

    def __init__(
        self,
        db_name: str,
        history_days: int = 30,
    ):
        """
        Initialize the session by loading initial history and route metadata.
        
        Args:
            db_name: Database name.
            history_days: Number of days of history to keep in memory.
        """
        self.db_name = db_name
        self.history_days = history_days
        
        self.db = DatabaseWrapper(database_name=db_name)
        self.avail = Availabilities(self.db)
        
        # 1. Fetch all routes for consistent encoding
        # Store as list of tuples for grid generation
        self.routes_df = self.avail.routes
        self.all_routes_tuples = list(
            self.routes_df[["departure_from", "departure_to"]].itertuples(index=False, name=None)
        )
        
        # Cache encoded categories
        self.all_from_cities = set(self.routes_df["departure_from"].unique())
        self.all_to_cities = set(self.routes_df["departure_to"].unique())
        
        # 2. Fetch Initial History
        latest_start = self.avail.latest_availability_start()
        if not latest_start:
            raise ValueError("No availability data found in database.")
        
        self.current_sim_date = latest_start.date()
        
        start_history = latest_start - timedelta(days=history_days)
        history_raw = self.avail.availability_start_ge(start_history)
        
        # Build labeled features for history (occurs=0 or 1)
        self.history_features = build_labeled_features(history_raw)
        
        # Prune future leakage from history (e.g. from availability_end > availability_start)
        # We only want history up to current_sim_date.
        history_cutoff = pd.Timestamp(self.current_sim_date)
        self.history_features = self.history_features[
            pd.to_datetime(self.history_features["availability_start"]) <= history_cutoff
        ]
        
        # We need financials for feature generation
        self.financials = Financials(self.db).get_all()

    def next_day(self, target_date: Optional[date] = None) -> pd.DataFrame:
        """
        Generate the feature dataframe for the next day (or specific date).
        
        Args:
            target_date: specific date to generate for. If None, defaults to 
                         current_sim_date + 1 day.
                         
        Returns:
            DataFrame containing features for the target date.
            Columns match training data format.
        """
        if target_date is None:
            target_date = self.current_sim_date + timedelta(days=1)
            
        # 1. Generate grid for the single target day
        # Note: build_route_date_grid expects datetime objects for start/end
        start_dt = datetime.combine(target_date, datetime.min.time())
        future_grid = build_route_date_grid(self.all_routes_tuples, start_dt, start_dt)
        future_grid["occurs"] = np.nan
        
        # 2. Combine with History
        # We append the new day to history to compute rolling/lagged features
        combined = pd.concat([self.history_features, future_grid], ignore_index=True, sort=False)
        combined = sort_features(combined)
        
        # 3. Apply Feature Pipeline
        # This re-calculates lags/rolling means using the joined history
        combined = add_derived_features(combined)
        combined = add_country_codes_columns(combined)
        combined = add_holiday_distance_features(combined)
        combined = add_latest_neer_features(combined, self.financials)
        
        # 4. Label Encoding
        known_categories = {
            "departure_from": self.all_from_cities,
            "departure_to": self.all_to_cities,
        }
        combined = add_label_encoded_columns(
            combined, 
            ["departure_from", "departure_to"],
            known_categories=known_categories
        )
        
        # 5. Extract just the target day
        target_ts = pd.Timestamp(target_date)
        mask = pd.to_datetime(combined["availability_start"]) == target_ts
        target_features = combined[mask].copy()
        
        return target_features

    def update(self, date: date, predictions: Union[pd.DataFrame, pd.Series, np.ndarray]):
        """
        Update the history with predictions for a specific date.
        This enables the next day's features to rely on these predicted values.
        
        Args:
            date: The date these predictions correspond to.
            predictions: The predicted 'occurs' values (0 or 1, or probability).
                        Must align with the rows generated by `next_day(date)`.
        """
        # We need to reconstruct the grid for this date to append it to history
        # with the *predicted* occurs values.
        
        # Regenerate the grid structure (same order as next_day)
        start_dt = datetime.combine(date, datetime.min.time())
        day_grid = build_route_date_grid(self.all_routes_tuples, start_dt, start_dt)
        
        # Assign predictions
        # If predictions is a DataFrame/Series with index alignment, great.
        # If it's an array, we assume it matches the sort order of `build_route_date_grid`.
        # build_route_date_grid output is deterministic (route list order * dates).
        
        # Ensure we are working with 0/1 for history
        if hasattr(predictions, "values"):
            vals = predictions.values
        else:
            vals = np.array(predictions)
            
        # Convert probabilities to binary if necessary (assuming threshold 0.5)
        # or keep as float if downstream features handle float probabilities 
        # (the current implementation of rolling means handles floats fine).
        day_grid["occurs"] = vals
        
        # Append to history
        self.history_features = pd.concat([self.history_features, day_grid], ignore_index=True)
        self.history_features = sort_features(self.history_features)
        
        # Prune history if it gets too large?
        # Ideally we keep 'history_days', but growing it slightly is fine.
        # Pruning might be expensive if done every step.
        
        # Update simulation clock
        self.current_sim_date = date
