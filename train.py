# %%
import pandas as pd

import data.training
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

import config
from helpers import show_or_save_plot

df = data.training.get()
df.tail()
# %%
# Separate features and target

# Initially we remove the departure and destination altogether, later
# versions may use use them as features or we may have one model for each
training_df = df.drop(
    columns=["id", "availability_start",  "departure_from", "departure_from_country", "departure_to", "departure_to_country"], errors="ignore"
)
# training_df = training_df.drop(columns=["occurs_lag_1", "occurs_lag_2", "occurs_lag_3", "occurs_mean_prev_7d", "occurs_mean_prev_14d"], errors="ignore")
X = training_df.drop(columns=["occurs"], errors="ignore")
y = training_df["occurs"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
X_train.head()
# %%
xgb_regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_regressor.fit(X_train, y_train)

y_pred = xgb_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

# %%
# display feature importance
feature_importance = xgb_regressor.feature_importances_
feature_names = X.columns

sorted_idx = feature_importance.argsort()
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
show_or_save_plot("xgboost_feature_importance")

# %%
# Train a dedicated XGBoost regressor for every sufficiently large route slice.
ROUTE_SAMPLE_MIN = 10
per_route_models_dir = config.ARTIFACTS_DIR / "per_route_models"
per_route_models_dir.mkdir(parents=True, exist_ok=True)

required_route_columns = {"departure_from", "departure_to"}
missing_columns = required_route_columns - set(df.columns)
if missing_columns:
    missing_str = ", ".join(sorted(missing_columns))
    raise ValueError(f"Missing required route columns: {missing_str}")


def _format_route_segment(value: str | float | None) -> str:
    """Return a filesystem-friendly representation of a route segment."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    return str(value).strip().lower().replace("/", "-") or "unknown"


grouped_routes = df.groupby(["departure_from", "departure_to"], dropna=False)
print(f"Found {len(grouped_routes)} unique routes to evaluate for dedicated models.")
# %%
route_metrics: list[dict[str, float | int | str]] = []
for (departure_from, departure_to), route_df in grouped_routes:
    route_sample_count = len(route_df)
    if route_sample_count < ROUTE_SAMPLE_MIN:
        print(
            f"Skipping {departure_from}->{departure_to} "
            f"({route_sample_count} < {ROUTE_SAMPLE_MIN})"
        )
        continue

    print(f"Training model for route {departure_from}->{departure_to} ")
    route_features = route_df.drop(
        columns=[
            "id",
            "availability_start",
            "occurs",
            "departure_from",
            "departure_from_country",
            "departure_to",
            "departure_to_country",
        ],
        errors="ignore",
    )
    X_route_train, X_route_test, y_route_train, y_route_test = train_test_split(
        route_features,
        route_df["occurs"],
        test_size=0.2,
        random_state=42,
    )
    route_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    route_model.fit(X_route_train, y_route_train)
    route_preds = route_model.predict(X_route_test)
    route_mse = mean_squared_error(y_route_test, route_preds)
    route_r2 = r2_score(y_route_test, route_preds)

    route_id = (
        f"{_format_route_segment(departure_from)}__"
        f"{_format_route_segment(departure_to)}"
    )
    model_path = per_route_models_dir / f"{route_id}.json"
    # Fixes a issue with sklearn compatibility
    route_model._estimator_type = "regressor"
    route_model.save_model(str(model_path))

    route_metrics.append(
        {
            "departure_from": departure_from,
            "departure_to": departure_to,
            "samples": route_sample_count,
            "mse": float(route_mse),
            "r2": float(route_r2),
            "model_path": model_path,
        }
    )
    print(
        f"Trained {route_id} | samples={route_sample_count} "
        f"mse={route_mse:.4f} r2={route_r2:.4f}"
    )

per_route_results = pd.DataFrame(route_metrics)
if per_route_results.empty:
    print(
        "No per-route models were trained; lower ROUTE_SAMPLE_MIN or gather more data."
    )
else:
    per_route_results = per_route_results.sort_values(
        by=["r2", "samples"], ascending=[False, False]
    )
per_route_results.head()

# %%
