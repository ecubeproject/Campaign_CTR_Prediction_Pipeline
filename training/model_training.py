import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import shap

# Absolute paths so this script runs the same way regardless of cwd
# (matches preprocessing.py's approach).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")
PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "preprocessed")
APP_DIR = os.path.join(PROJECT_ROOT, "Streamlit_CTR_app")

# Load preprocessed data and preprocessor
data = pd.read_csv(os.path.join(PREPROCESSED_DIR, "campaign_data_preprocessed.csv"))
preprocessor = joblib.load(os.path.join(PREPROCESSED_DIR, "preprocessor.pkl"))

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
r2 = r2_score(y_test, y_pred)
print(f"Meta model RMSE: {rmse:.4f}")
print(f"Meta model R2: {r2:.4f}")

# Save model in XGBoost's native format -- this is what app.py loads via
# XGBRegressor().load_model(), so save directly here rather than via
# joblib.dump (which produces a format load_model() can't read) and rather
# than depending on a separate export step that has to dig the artifact
# back out of a randomly-generated MLflow run directory.
training_model_path = os.path.join(TRAINING_DIR, "xgb_meta_model.json")
model.save_model(training_model_path)
print(f"XGBoost meta model saved to: {training_model_path}")

os.makedirs(APP_DIR, exist_ok=True)
app_model_path = os.path.join(APP_DIR, "xgb_meta_model.json")
model.save_model(app_model_path)
joblib.dump(preprocessor, os.path.join(APP_DIR, "preprocessor.pkl"))
print(f"Deployment copies written to: {APP_DIR}")

# Start MLflow logging
mlflow.set_experiment("CTR_Prediction")
with mlflow.start_run():
    # Log metrics and params
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
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
    feature_importance_path = os.path.join(TRAINING_DIR, "feature_importance.png")
    plt.savefig(feature_importance_path)
    plt.close()
    mlflow.log_artifact(feature_importance_path)

    # SHAP summary plot
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    shap_summary_path = os.path.join(TRAINING_DIR, "shap_summary.png")
    plt.savefig(shap_summary_path)
    plt.close()
    mlflow.log_artifact(shap_summary_path)

print("Training and logging complete.")
