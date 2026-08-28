import os

import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.utils.validation import check_is_fitted

# Resolve artifact paths relative to this file, not the process CWD.
# Streamlit Community Cloud runs `streamlit run` from the repo root, so bare
# relative paths like "xgb_meta_model.json" would not be found; the Docker
# image happened to work only because its WORKDIR is this folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    mdl = xgb.XGBRegressor()
    mdl.load_model(os.path.join(BASE_DIR, "xgb_meta_model.json"))
    pre = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
    check_is_fitted(pre)
    return mdl, pre


model, preprocessor = load_artifacts()

# Streamlit App
st.set_page_config(page_title="Campaign CTR Predictor", page_icon="📈", layout="centered")
st.title("📈 Campaign CTR Predictor")
st.markdown(
    "Estimate a marketing campaign's **click-through rate (CTR)** from "
    "pre-campaign planning attributes only — product, audience, channel, "
    "budget and duration. Built with an XGBoost model and a scikit-learn "
    "preprocessing pipeline. See the linked repo for methodology and data notes."
)

# Input fields
with st.form("campaign_form"):
    col1, col2 = st.columns(2)

    with col1:
        product_type = st.selectbox("Product Type", ["Books", "Laptops", "Shoes", "Furniture", "Smartphones"])
        audience_age = st.selectbox("Audience Age", ["18-24", "25-34", "35-44", "45-54", "55+"])
        audience_gender = st.selectbox("Audience Gender", ["Male", "Female", "Other"])
        location = st.selectbox("Location", ["California", "Texas", "Florida", "New York", "Illinois"])
        channel = st.selectbox("Marketing Channel", ["Facebook", "Instagram", "Google Ads", "LinkedIn", "YouTube"])

    with col2:
        duration_days = st.number_input("Duration (in days)", min_value=1, max_value=60, value=10)
        budget_usd = st.number_input("Budget (in USD)", min_value=100, max_value=100000, value=10000)

    submit = st.form_submit_button("Predict CTR")

if submit:
    # Only pre-campaign planning attributes are used -- clicks / impressions /
    # conversions are outcomes of a campaign, not knowable in advance, so they
    # are never model inputs (using them here would let the model "predict"
    # CTR by just recovering clicks / impressions, which is circular).
    input_dict = {
        "product_type": product_type,
        "audience_age": audience_age,
        "audience_gender": audience_gender,
        "location": location,
        "channel": channel,
        "duration_days": duration_days,
        "budget_usd": budget_usd,
    }

    input_data = pd.DataFrame([input_dict])

    expected_cols = set(preprocessor.feature_names_in_)
    missing_cols = expected_cols - set(input_data.columns)

    if missing_cols:
        st.error(f"Input data is missing required columns for preprocessing: {missing_cols}")
    else:
        try:
            X_processed = preprocessor.transform(input_data)
            ctr_pred = float(model.predict(X_processed)[0])
            st.metric("Predicted CTR", f"{ctr_pred:.2%}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.caption("Auto-deployed from GitHub via Cloud Run CI/CD.")  # cicd-test-1787950226
