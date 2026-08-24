
import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np
import os
from sklearn.utils.validation import check_is_fitted

# Load model and preprocessor
model = xgb.XGBRegressor()
model.load_model("xgb_meta_model.json")
preprocessor = joblib.load("preprocessor.pkl")
check_is_fitted(preprocessor)

# Streamlit App
st.set_page_config(page_title="CTR Predictor", layout="centered")
st.title("Campaign CTR Predictor (XGBoost + Streamlit)")
st.markdown("Enter campaign details below to predict **Click-Through Rate (CTR)**.")

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
    # Build input as dictionary. Only pre-campaign planning attributes are
    # used -- clicks/impressions/conversions are outcomes of a campaign,
    # not knowable in advance, so they are never model inputs (using them
    # here would let the model "predict" CTR by just recovering
    # clicks / impressions, which is circular and not real prediction).
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

    # Validate input columns
    expected_cols = set(preprocessor.feature_names_in_)
    print(expected_cols)
    received_cols = set(input_data.columns)
    missing_cols = expected_cols - received_cols

    if missing_cols:
        st.error(f"Input data is missing required columns for preprocessing: {missing_cols}")
    else:
        try:
            X_processed = preprocessor.transform(input_data)
            ctr_pred = model.predict(X_processed)[0]
            st.success(f"**Predicted CTR:** {ctr_pred:.2%}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
