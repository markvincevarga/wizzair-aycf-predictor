"""Evaluation utilities for comparing predictions with actual availability data."""

import pandas as pd
from typing import Dict, Any

from storage.database import DatabaseWrapper
from storage.availabilities import Availabilities


def evaluate_predictions(
    preds_df: pd.DataFrame,
    db_name: str,
) -> pd.DataFrame:
    """
    Compare predictions with actual availability data from the database.

    Uses the same logic as the UI performance view: creates a set of
    (departure_from, departure_to, date) tuples from actual availabilities,
    and marks each prediction as correct (1) or incorrect (0) based on
    whether the route-date combination exists in that set.

    Args:
        preds_df: DataFrame with predictions containing at least:
            - departure_from: str
            - departure_to: str
            - availability_start: datetime-like
            - predicted_available: int (0 or 1)
            - predicted_probability: float
        db_name: Name of the database to fetch actual availabilities from.

    Returns:
        DataFrame with all prediction columns plus:
            - target_date: date extracted from availability_start
            - occurs: int (1 if actually available, 0 otherwise)
    """
    if preds_df.empty:
        return preds_df.copy()

    # Ensure datetime type
    preds_df = preds_df.copy()
    preds_df["availability_start"] = pd.to_datetime(preds_df["availability_start"])

    # Get prediction date range
    pred_start = preds_df["availability_start"].min().date()
    pred_end = preds_df["availability_start"].max().date()

    # Fetch actual availabilities
    db = DatabaseWrapper(database_name=db_name)
    avail_repo = Availabilities(db)
    df_avail = avail_repo.get_recent_availabilities(pred_start, pred_end)

    # Create date column for lookup
    preds_df["target_date"] = preds_df["availability_start"].dt.date

    if df_avail.empty:
        # No availabilities found - all predictions are treated as False Positives
        preds_df["occurs"] = 0
        return preds_df

    df_avail["target_date"] = df_avail["availability_start"].dt.date

    # Create actuals set for fast lookup
    actuals_set = set(
        zip(
            df_avail["departure_from"],
            df_avail["departure_to"],
            df_avail["target_date"],
        )
    )

    # Add actual outcome to predictions
    # If (route, date) is in actuals_set -> 1 (Available), else -> 0 (Unavailable)
    preds_df["occurs"] = preds_df.apply(
        lambda row: 1
        if (row["departure_from"], row["departure_to"], row["target_date"])
        in actuals_set
        else 0,
        axis=1,
    )

    return preds_df


def compute_metrics(
    comparison_df: pd.DataFrame,
    true_col: str = "occurs",
    pred_col: str = "predicted_available",
    prob_col: str = "predicted_probability",
) -> Dict[str, Any]:
    """
    Compute classification metrics from a comparison DataFrame.

    Args:
        comparison_df: DataFrame with actual and predicted values.
        true_col: Column name for actual values (0/1).
        pred_col: Column name for predicted values (0/1).
        prob_col: Column name for prediction probabilities.

    Returns:
        Dictionary with metrics: accuracy, precision, recall, f1, auc_roc, samples.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    if comparison_df.empty:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc_roc": 0.0,
            "samples": 0,
        }

    y_true = comparison_df[true_col].astype(int)
    y_pred = comparison_df[pred_col].astype(int)
    y_prob = comparison_df[prob_col]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0  # Handle case with single class

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": roc_auc,
        "samples": len(comparison_df),
    }
