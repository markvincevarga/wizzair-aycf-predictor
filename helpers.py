from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import config

# Subdirectory for plots within artifacts
PLOTS_SUBDIR = "plots"


def is_notebook() -> bool:
    """Check if code is running in a Jupyter notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        if shell_name == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell_name == "TerminalInteractiveShell":
            return False  # Terminal IPython
        else:
            return False
    except (ImportError, NameError):
        return False


def show_or_save_plot(name: str, always_store: bool = False) -> None:
    """
    Show or save a matplotlib plot depending on the execution environment.

    Args:
        name: The name for the plot file (without extension).
        always_store: If True, always save the plot to disk regardless of environment.
    """
    if always_store or not is_notebook():
        plots_dir = config.ARTIFACTS_DIR / PLOTS_SUBDIR
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_path = plots_dir / f"{name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")

    if is_notebook():
        plt.show()
    else:
        plt.close()


def collect_model_stats(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    train_size: int,
    test_size: int,
    train_class_dist: dict,
    test_class_dist: dict,
) -> dict:
    """
    Collect model performance statistics into a dictionary.

    Args:
        y_test: True labels for test set.
        y_pred: Predicted labels for test set.
        y_pred_proba: Predicted probabilities for positive class.
        train_size: Number of samples in training set.
        test_size: Number of samples in test set.
        train_class_dist: Class distribution in training set.
        test_class_dist: Class distribution in test set.

    Returns:
        Dictionary containing all model statistics.
    """
    cm = confusion_matrix(y_test, y_pred)

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "train_size": train_size,
            "test_size": test_size,
            "train_class_distribution": train_class_dist,
            "test_class_distribution": test_class_dist,
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "auc_roc": float(roc_auc_score(y_test, y_pred_proba)),
        },
        "confusion_matrix": cm.tolist(),
    }
