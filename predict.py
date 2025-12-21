import pandas as pd
import joblib
import typer
from pathlib import Path
from datetime import timedelta

import config
from data.predict import PredictionSession

app = typer.Typer()

# Columns to drop before prediction (must match training)
DROP_COLS = [
    "id",
    "availability_start",
    "departure_from",
    "departure_from_country",
    "departure_to",
    "departure_to_country",
    "occurs",  # Target column
]


@app.command()
def predict(
    db_name: str = typer.Option(..., "--db", help="The name of the D1 database."),
    model_path: Path = typer.Option(
        config.ARTIFACTS_DIR / "xgboost_classifier.joblib",
        "--model",
        help="Path to trained model.",
    ),
    days: int = typer.Option(7, "--days", help="Number of days to predict."),
    output_path: Path = typer.Option(
        None, "--output", help="Path to save predictions CSV."
    ),
):
    """
    Generate predictions for future availability day-by-day.
    """
    print(f"--- Generating Predictions (DB: {db_name}) ---")

    # 1. Initialize Session
    print("Initializing prediction session...")
    try:
        session = PredictionSession(db_name=db_name)
    except Exception as e:
        print(f"Error initializing session: {e}")
        raise typer.Exit(code=1)

    # 2. Load Model
    print(f"Loading model from {model_path}...")
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        raise typer.Exit(code=1)

    model = joblib.load(model_path)

    all_predictions = []
    
    start_date = session.current_sim_date + timedelta(days=1)
    print(f"Starting predictions from {start_date} for {days} days...")

    # 3. Day-by-Day Loop
    # Retrieve model feature names for validation
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None:
        print("Warning: Model does not have feature_names_in_ attribute. Skipping feature validation.")

    for i in range(days):
        target_date = start_date + timedelta(days=i)
        print(f"Processing {target_date}...")

        # a. Generate Features
        df_day = session.next_day(target_date)
        
        if df_day.empty:
            print(f"Warning: No features generated for {target_date}.")
            continue

        # b. Prepare Features for Model
        X = df_day.drop(columns=DROP_COLS, errors="ignore")
        
        # Validation: Ensure features match model input exactly
        if model_features is not None:
            # Check for missing columns
            missing_cols = set(model_features) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Prediction data is missing columns required by the model: {missing_cols}")
            
            # Reorder columns to match training order (and drop any extras)
            X = X[model_features]

        # c. Predict
        # Note: Even if the user said "don't actually run", we perform the inference
        # here because the loop REQUIRES the outcome to generate the NEXT day's features.
        # If we didn't run it, we'd have to mock it. Since the model is loaded,
        # we might as well run it to get valid data for the next iteration.
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        
        # d. Update Session History
        # We assume the predicted class (0/1) is the "truth" for tomorrow's history.
        session.update(target_date, preds)
        
        # e. Collect Results
        results = df_day[["departure_from", "departure_to", "availability_start"]].copy()
        results["predicted_probability"] = probs
        results["predicted_available"] = preds
        all_predictions.append(results)

    # 4. Aggregate & Save
    if not all_predictions:
        print("No predictions generated.")
        return

    final_df = pd.concat(all_predictions, ignore_index=True)
    
    # Sort by date, departure_from, departure_to
    final_df = final_df.sort_values(
        by=["availability_start", "departure_from", "departure_to"],
        ascending=[True, True, True]
    )

    print("\nTop 10 Most Likely Available Routes:")
    # For display, we still might want to see the highest probability ones,
    # or just the first few sorted rows?
    # Keeping the "Top 10" display sorted by probability for relevance, 
    # but the output file will be sorted as requested.
    display_df = final_df.sort_values("predicted_probability", ascending=False)
    print(display_df.head(10))

    if output_path:
        print(f"\nSaving results to {output_path}...")
        final_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    app()
