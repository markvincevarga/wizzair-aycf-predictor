# %%
import joblib

import data.training
from data.split import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

import config
from helpers import show_or_save_plot

df = data.training.get()
df.tail()
# %%
# Separate features and target

# Initially we remove the departure and destination altogether, later
# versions may use use them as features or we may have one model for each
DROP_COLS = [
    "id",
    "availability_start",
    "departure_from",
    "departure_from_country",
    "departure_to",
    "departure_to_country",
]

# Time-based train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df)
X_train = X_train.drop(columns=DROP_COLS, errors="ignore")
X_test = X_test.drop(columns=DROP_COLS, errors="ignore")
X = df.drop(columns=DROP_COLS + ["occurs"], errors="ignore")

# Convert target to integer for classification
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Class distribution in train: {y_train.value_counts().to_dict()}")
print(f"Class distribution in test: {y_test.value_counts().to_dict()}")
X_train.head()

# %%
# Train XGBoost Classifier with tuned hyperparameters
# (optimized via Optuna with TimeSeriesSplit cross-validation)
xgb_classifier = XGBClassifier(
    n_estimators=403,
    max_depth=10,
    learning_rate=0.1013574857037285,
    min_child_weight=1,
    subsample=0.8500168201866831,
    colsample_bytree=0.7255486664643275,
    gamma=0.10597988439650874,
    reg_alpha=0.020124256179981054,
    reg_lambda=0.0474101828395834,
    random_state=42,
    eval_metric="logloss",
)
xgb_classifier.fit(X_train, y_train)

# Predictions
y_pred = xgb_classifier.predict(X_test)
y_pred_proba = xgb_classifier.predict_proba(X_test)[:, 1]

# Classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print("\n=== Classification Metrics ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc_roc:.4f}")

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Not Available", "Available"]))

# %%
# Save the trained model
model_path = config.ARTIFACTS_DIR / "xgboost_classifier.joblib"
joblib.dump(xgb_classifier, model_path)
print(f"\nModel saved to: {model_path}")

# %%
# Display feature importance
feature_importance = xgb_classifier.feature_importances_
feature_names = X.columns

sorted_idx = feature_importance.argsort()
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("XGBoost Classifier Feature Importance")
plt.tight_layout()
show_or_save_plot("xgboost_feature_importance")

# %%
