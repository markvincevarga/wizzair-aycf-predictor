#!/usr/bin/env python3
"""
Predict CLI tool for WizzAir AYCF availability predictions.

Loads a trained XGBoost model and generates predictions with certainty scores
for future route-date combinations, storing results in the database.
"""

import joblib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

import config
from features.availabilities import (
    build_labeled_features,
    extract_routes,
    sort_features,
    build_route_date_grid,
)
from features.country_codes import add_country_codes_columns
from features.financial import add_latest_neer_features
from features.holidays import add_holiday_distance_features
from storage.availabilities import Availabilities
from storage.financials import Financials
from storage.database import DatabaseWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(model_path: Optional[Path] = None) -> object:
    """Load trained XGBoost model from file."""
    if model_path is None:
        model_path = config.ARTIFACTS_DIR / "xgboost_classifier.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)


def prepare_prediction_data(db: DatabaseWrapper, days_ahead: int = 7) -> pd.DataFrame:
    """
    Prepare feature data for prediction including historical context and future dates.

    Args:
        db: Database connection
        days_ahead: Number of future days to predict

    Returns:
        DataFrame with features ready for prediction
    """
    # Get latest availability start date
    latest_start = Availabilities(db).latest_availability_start()
    if latest_start is None:
        raise ValueError("No historical availability data found")

    # Get historical data for feature engineering (last 14 days needed for rolling features)
    availabilities_start = latest_start - timedelta(days=14)
    availabilities = Availabilities(db).availability_start_ge(
        start_date=availabilities_start
    )

    historical_availabilities = build_labeled_features(availabilities)
    historical_availabilities = sort_features(historical_availabilities)

    # Generate future route-date combinations
    start = latest_start + timedelta(days=1)
    end = start + timedelta(days=days_ahead)
    routes = extract_routes(historical_availabilities)
    future_availabilities = build_route_date_grid(routes, start, end)

    # Combine and engineer features
    all_availabilities = pd.concat(
        [historical_availabilities, future_availabilities], ignore_index=True
    )
    all_availabilities = add_country_codes_columns(all_availabilities)
    all_availabilities = add_holiday_distance_features(all_availabilities)

    financials = Financials(db).get_all()
    all_availabilities = add_latest_neer_features(all_availabilities, financials)

    return all_availabilities


def generate_predictions(model, feature_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions with certainty scores using the trained model.

    Args:
        model: Trained XGBoost classifier
        feature_data: DataFrame with engineered features

    Returns:
        DataFrame with predictions and certainty scores
    """
    # Filter to only future dates (where occurs is NaN)
    future_mask = feature_data["occurs"].isna()
    prediction_data = feature_data[future_mask].copy()

    if prediction_data.empty:
        logger.info("No future data available for prediction")
        return pd.DataFrame()

    # Prepare features (drop columns not used in training)
    DROP_COLS = [
        "id",
        "availability_start",
        "departure_from",
        "departure_from_country",
        "departure_to",
        "departure_to_country",
        "occurs",
    ]

    X_pred = prediction_data.drop(columns=DROP_COLS, errors="ignore")

    # Generate predictions and probabilities
    predictions = model.predict(X_pred)
    probabilities = model.predict_proba(X_pred)[
        :, 1
    ]  # Probability of class 1 (available)

    # Create results DataFrame
    results = pd.DataFrame(
        {
            "availability_start": prediction_data["availability_start"],
            "departure_from": prediction_data["departure_from"],
            "departure_to": prediction_data["departure_to"],
            "predicted_occurs": predictions.astype(int),
            "certainty": probabilities,
            "prediction_date": datetime.utcnow(),
        }
    )

    logger.info(f"Generated {len(results)} predictions")
    return results


def store_predictions(db: DatabaseWrapper, predictions: pd.DataFrame) -> None:
    """
    Store predictions in the database.

    Args:
        db: Database connection
        predictions: DataFrame with predictions to store
    """
    if predictions.empty:
        logger.info("No predictions to store")
        return

    # Convert timestamps to Unix timestamps for D1 storage
    predictions["availability_start"] = (
        predictions["availability_start"].astype(int) // 10**9
    )
    predictions["prediction_date"] = predictions["prediction_date"].astype(int) // 10**9

    # Store in predictions table
    db.push_new_rows("predictions", predictions, ignore_duplicates=True)
    logger.info(f"Stored {len(predictions)} predictions in database")


app = typer.Typer()


@app.command()
def predict(
    db: str = typer.Option("wizz-aycf", help="Database name"),
    model: Optional[str] = typer.Option(None, help="Path to trained model file"),
    days: int = typer.Option(7, help="Number of days ahead to predict"),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
) -> None:
    """
    Generate availability predictions and store them in the database.

    Loads a trained XGBoost model and generates predictions for future
    route-date combinations with certainty scores.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize database connection
        database = DatabaseWrapper(database_name=db)
        logger.info(f"Connected to database: {db}")

        # Load trained model
        model_path = Path(model) if model else None
        trained_model = load_model(model_path)

        # Prepare prediction data
        logger.info("Preparing prediction data...")
        feature_data = prepare_prediction_data(database, days_ahead=days)

        # Generate predictions
        logger.info("Generating predictions...")
        predictions = generate_predictions(trained_model, feature_data)

        # Store predictions
        logger.info("Storing predictions...")
        store_predictions(database, predictions)

        logger.info("Prediction workflow completed successfully")

        # Display summary
        if not predictions.empty:
            print(
                f"\nGenerated {len(predictions)} predictions for the next {days} days"
            )
            print(f"Average certainty: {predictions['certainty'].mean():.3f}")
            print(
                f"Predicted availability rate: {predictions['predicted_occurs'].mean():.3f}"
            )

    except Exception as e:
        logger.error(f"Prediction workflow failed: {e}")
        raise typer.Exit(str(e))


if __name__ == "__main__":
    app()
