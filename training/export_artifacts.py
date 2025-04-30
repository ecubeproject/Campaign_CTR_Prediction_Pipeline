import mlflow
import joblib
import shutil
import os

# Load model from MLflow registry or path
model_path = "training/mlruns/820660959291570411/f8ae0b4152ca40ffa2114ee502c3b70d/artifacts/xgb_model/model.xgb"
preprocessor_path = "preprocessed/preprocessor.pkl"

# Export model
shutil.copy(model_path, "Streamlit_CTR_app/xgb_meta_model.pkl")
print("Model exported to Streamlit_CTR_app/xgb_meta_model.pkl")

# Export preprocessor
shutil.copy(preprocessor_path, "Streamlit_CTR_app/preprocessor.pkl")
print("Preprocessor exported to Streamlit_CTR_app/preprocessor.pkl")
