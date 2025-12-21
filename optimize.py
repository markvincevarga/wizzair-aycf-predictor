import optuna
import typer
import statistics
from pathlib import Path
from tqdm import tqdm
from backtest import run_backtest

app = typer.Typer()

@app.command()
def optimize(
    db_name: str = typer.Option("wizz", "--db", help="Name of the database"),
    trials: int = typer.Option(20, "--trials", help="Number of Optuna trials"),
    days: int = typer.Option(7, "--days", help="Prediction horizon for backtests"),
    storage: str = typer.Option(None, "--storage", help="Optuna storage URL (e.g., sqlite:///db.sqlite3)"),
):
    """
    Optimize XGBoost hyperparameters using Optuna and backtesting.
    """
    
    # Dates for backtesting
    cutoff_dates = ["2025-09-10", "2025-11-10", "2025-12-01"]

    # Silence Optuna logging to avoid interfering with tqdm
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Initialize progress bar
    pbar = tqdm(total=trials, desc="Optimization Progress", unit="trial")

    def objective(trial):
        # Hyperparameter search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42,
            "eval_metric": "logloss",
        }

        tqdm.write(f"\nTrial {trial.number + 1}/{trials}: Testing params: {params}")

        # Define status callback
        def status_update(status: str):
            pbar.set_description(f"Trial {trial.number + 1}/{trials} - {status}")

        # Run backtest for all dates
        # Using force_rebuild=False to use cached data if available (filled by previous runs or manually)
        # Note: The first run might need force_rebuild=True if data is stale, but we assume
        # the user handles data freshness or we accept cached data for speed during optimization.
        # Actually, let's force rebuild only on the VERY first trial of the script execution to be safe?
        # Or better yet, just trust the cache or let the user run fill.py/train.py once before.
        # The plan says "This allows the optimizer to use force_rebuild=False".
        
        try:
            results = run_backtest(
                db_name=db_name,
                cutoff_dates=cutoff_dates,
                days=days,
                model_params=params,
                predictions_file=Path(f"artifacts/optuna_preds_{trial.number}.csv"),
                comparison_file=Path(f"artifacts/optuna_comp_{trial.number}.csv"),
                force_rebuild=False,
                status_callback=status_update
            )
        except Exception as e:
            tqdm.write(f"Trial {trial.number + 1} failed: {e}")
            pbar.update(1)
            # Return a bad score if backtest fails
            return 0.0

        if not results:
            pbar.update(1)
            return 0.0

        f1_scores = [r["f1"] for r in results]
        mean_f1 = statistics.mean(f1_scores)
        
        tqdm.write(f"Trial {trial.number + 1} Results: F1 Scores: {f1_scores}, Mean F1: {mean_f1}")
        pbar.update(1)
        return mean_f1

    print("Starting Optuna optimization...")
    study = optuna.create_study(direction="maximize", storage=storage)
    try:
        study.optimize(objective, n_trials=trials)
    finally:
        pbar.close()

    print("\n" + "="*40)
    print("Optimization Completed")
    print("="*40)
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Value (Mean F1): {study.best_value}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("="*40)

if __name__ == "__main__":
    app()

