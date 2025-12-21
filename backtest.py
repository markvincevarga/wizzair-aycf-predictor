import sys
import json
import pandas as pd
import typer
from typing import List, Dict, Any, Optional, Callable
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import data.training
from train import train_model
from predict import generate_predictions
from pathlib import Path

app = typer.Typer()

def run_backtest(
    db_name: str,
    cutoff_dates: List[str],
    days: int,
    bucket_name: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    predictions_file: Path = Path("artifacts/backtest_predictions.csv"),
    comparison_file: Path = Path("artifacts/backtest_comparison.csv"),
    force_rebuild: bool = True,
    status_callback: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Run backtest for multiple cutoff dates and return collected metrics.
    """
    results = []

    for cutoff_date in cutoff_dates:
        print(f"\n\n{'='*40}")
        print(f"Running Backtest for Cutoff: {cutoff_date}")
        print(f"{'='*40}")

        # 1. Training Step
        if status_callback:
            status_callback(f"Training ({cutoff_date})")
        print("\n=== Step 1: Training Model ===")
        train_model(
            db_name=db_name,
            bucket=bucket_name,
            cutoff_date=cutoff_date,
            force_rebuild=force_rebuild,
            model_params=model_params
        )

        # 2. Prediction Step
        if status_callback:
            status_callback(f"Predicting ({cutoff_date})")
        print("\n=== Step 2: Generating Predictions ===")
        generate_predictions(
            db_name=db_name,
            start_date=cutoff_date,
            days=days,
            output_path=predictions_file
        )

        # 3. Evaluation Step
        if status_callback:
            status_callback(f"Evaluating ({cutoff_date})")
        print("\n=== Step 3: Evaluating Performance ===")
        
        # Load predictions
        print(f"Loading predictions from {predictions_file}...")
        try:
            preds_df = pd.read_csv(predictions_file)
        except FileNotFoundError:
            print(f"Error: {predictions_file} not found. Prediction step might have failed.")
            continue

        if preds_df.empty:
            print("Predictions file is empty. Skipping this date.")
            continue

        # Load ground truth
        print("Loading ground truth data...")
        ground_truth_df = data.training.get(db_name=db_name, force_rebuild=False)

        # Prepare for merge
        # Ensure date columns are datetime for correct merging
        preds_df["availability_start"] = pd.to_datetime(preds_df["availability_start"])
        ground_truth_df["availability_start"] = pd.to_datetime(ground_truth_df["availability_start"])

        # Filter ground truth to relevant dates/routes
        # We only care about rows that exist in the prediction set
        # Merge on keys
        merge_keys = ["departure_from", "departure_to", "availability_start"]
        
        print("Merging predictions with ground truth...")
        comparison_df = pd.merge(
            preds_df, 
            ground_truth_df[merge_keys + ["occurs"]], 
            on=merge_keys, 
            how="inner",
            suffixes=("_pred", "_true")
        )
        
        if comparison_df.empty:
            print("Warning: No overlapping records found between predictions and ground truth.")
            print("Predictions range:", preds_df["availability_start"].min(), "to", preds_df["availability_start"].max())
            print("Ground truth range:", ground_truth_df["availability_start"].min(), "to", ground_truth_df["availability_start"].max())
            continue

        # Calculate metrics
        y_true = comparison_df["occurs"].astype(int)
        y_pred = comparison_df["predicted_available"].astype(int)
        y_prob = comparison_df["predicted_probability"]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_auc = 0.0  # Handle case with single class

        metrics = {
            "cutoff_date": cutoff_date,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": roc_auc,
            "samples": len(comparison_df)
        }
        results.append(metrics)

        print("\n" + "-"*30)
        print(f"   RESULTS FOR {cutoff_date}")
        print("-" * 30)
        print(f"Date Range: {comparison_df['availability_start'].min().date()} to {comparison_df['availability_start'].max().date()}")
        print(f"Samples:    {len(comparison_df)}")
        print("-" * 30)
        print(f"Accuracy:   {accuracy:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
        print(f"F1 Score:   {f1:.4f}")
        print(f"AUC-ROC:    {roc_auc:.4f}")
        print("-" * 30)
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["Not Available", "Available"], zero_division=0))

        # Save comparison
        comparison_df.to_csv(comparison_file, index=False)
        print(f"\nDetailed comparison saved to: {comparison_file}")

    return results

@app.command()
def backtest(
    db_name: str = typer.Option("wizz-aycf", "--db", help="Name of the database"),
    bucket_name: str = typer.Option("wizz-aycf", "--bucket", help="Name of the S3 bucket"),
    cutoff_date: str = typer.Option("2025-08-05", "--cutoff-date", help="Cutoff date for training (YYYY-MM-DD)"),
    prediction_days: int = typer.Option(14, "--days", help="Number of days to predict"),
    predictions_file: str = typer.Option("artifacts/backtest_predictions.csv", "--predictions-file", help="Path to save prediction results"),
    comparison_file: str = typer.Option("artifacts/backtest_comparison.csv", "--comparison-file", help="Path to save comparison results"),
    outfile: Optional[str] = typer.Option(None, "--outfile", help="Path to save collected stats JSON"),
):
    """
    Run a full backtest: Train model up to cutoff, predict subsequent days, and evaluate against ground truth.
    """
    
    results = run_backtest(
        db_name=db_name,
        cutoff_dates=[cutoff_date],
        days=prediction_days,
        bucket_name=bucket_name,
        predictions_file=Path(predictions_file),
        comparison_file=Path(comparison_file),
    )

    if not results:
        print("\nNo results collected. Backtest might have failed.")
        sys.exit(1)

    # Since CLI is for single date, just show/save that
    if outfile:
        outfile_path = Path(outfile)
        outfile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(outfile_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nStats saved to: {outfile_path}")

if __name__ == "__main__":
    app()
