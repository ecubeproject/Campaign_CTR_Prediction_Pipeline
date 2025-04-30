### 📘 README.md

# Recommender System for Campaign CTR Prediction

This project is an end-to-end machine learning pipeline to predict the **Click-Through Rate (CTR)** of marketing campaigns using a hybrid recommendation approach. It combines XGBoost regression with a Streamlit UI and is deployed on **Google Cloud Run**.

---

## 📊 Business Objective
Given historical campaign data, predict the CTR to help marketing teams optimize campaign performance and ROI.

---

## 🔧 Tech Stack
- **Python 3.9**
- **scikit-learn**, **XGBoost**, **Streamlit**, **MLflow**
- **Google Cloud Run**, **Cloud Build**, **Artifact Registry**
- **EDA & Visualization**: matplotlib, seaborn

---

## 🏗️ Project Structure

```
.
├── data/                          # Raw campaign data
├── eda/                           # Exploratory data analysis
├── preprocessing/                # Preprocessing script
├── preprocessed/                 # Preprocessed data and preprocessor.pkl
├── training/                     # Model training and artifacts
├── Streamlit_CTR_app/            # Streamlit UI with Docker support
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview
```

---

## 🚀 Steps to Run Locally

```bash
# 1. Clone repo & setup environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r Streamlit_CTR_app/requirements.txt

# 2. Preprocess the data
python preprocessing/preprocessing.py

# 3. Train model
python training/model_training.py

# 4. Export model & preprocessor
python training/export_artifacts.py

# 5. Launch Streamlit app
cd Streamlit_CTR_app
streamlit run app.py
```

---

## ☁️ GCP Deployment

```bash
# Build Docker & push to Artifact Registry
cd Streamlit_CTR_app

# Enable Cloud Build + Artifact Registry first

# Tag and push image
gcloud builds submit --tag us-central1-docker.pkg.dev/<PROJECT_ID>/vertex-models/ctr-predictor

# Deploy to Cloud Run
gcloud run deploy ctr-streamlit-ui \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 📷 Screenshots
![UI](screenshots/app_ui.png)

---

## 📬 Contact
For questions, contact `aimldstejas@gmail.com` or raise an issue.

