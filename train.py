# %%
import data.training
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

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
plt.show()

# %%
