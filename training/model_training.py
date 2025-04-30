import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import shap
import os

# Load preprocessed data and preprocessor
data = pd.read_csv("../preprocessed/campaign_data_preprocessed.csv")
preprocessor = joblib.load("../preprocessed/preprocessor.pkl")

# Consistent with preprocessing.py and app UI
drop_cols = ['click_through_rate']
X = data.drop(columns=drop_cols)
y = data["click_through_rate"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost regressor
model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Meta model RMSE: {rmse:.4f}")

# Save model locally
joblib.dump(model, "xgb_meta_model.pkl")
print("XGBoost meta model saved.")

# Start MLflow logging
mlflow.set_experiment("CTR_Prediction")
with mlflow.start_run():
    # Log metrics and params
    mlflow.log_metric("rmse", rmse)
    mlflow.log_params({
        "model_type": "XGBoost",
        "n_estimators": 100,
        "max_depth": 4,
        "pipeline": "standard_scaler + onehot_encoder"
    })

    # Log model
    input_example = X_test.iloc[:1]
    signature = infer_signature(X_test, y_pred)
    mlflow.xgboost.log_model(model, artifact_path="xgb_model",
                             input_example=input_example,
                             signature=signature)

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, ax=ax)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    mlflow.log_artifact("feature_importance.png")

    # SHAP summary plot
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()
    mlflow.log_artifact("shap_summary.png")

print("Training and logging complete.")
