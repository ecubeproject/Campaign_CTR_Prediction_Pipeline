# Campaign CTR Prediction Pipeline

An end-to-end ML pipeline that predicts marketing campaign **Click-Through Rate (CTR)** from pre-campaign planning attributes (product, audience, budget, channel, duration, location), using XGBoost, tracked with MLflow, reading training data from **BigQuery**, containerized with Docker, and deployed as a Streamlit app on **GCP Cloud Run** via Cloud Build + Artifact Registry.

**This is primarily an MLOps / cloud-deployment skills demo, not a claim of a production-grade CTR model.** The training data is synthetic — see "About the Data" below for exactly what that means and why.

**Live demo:** https://ctr-streamlit-ui-zlnzszgezq-uc.a.run.app
(Runs on Cloud Run's scale-to-zero tier — the first request after inactivity may take a few seconds to cold-start. Deployed automatically from `main` by GitHub Actions; see `.github/workflows/deploy-cloudrun.yml`.)

---

## What This Demonstrates

- A modular ML pipeline: data generation → BigQuery → preprocessing → training (MLflow-tracked) → SHAP explainability → Streamlit UI → Docker → Cloud Run.
- Reading training data from a live BigQuery table rather than a static file.
- Deploying a containerized app to Cloud Run via Cloud Build + Artifact Registry.
- A real bug caught and fixed during deployment: an unpinned `scikit-learn` version in the Docker image silently produced wrong predictions because the container's Python version couldn't install the same sklearn version used to pickle the preprocessor (see "Lessons Learned" below) — worth reading if you're deploying scikit-learn pickles in containers.

## About the Data

`data/campaign_data.csv` (and its BigQuery copy) is **synthetically generated** by `generator/data_generator.py`, not real campaign data. Real relationships were deliberately encoded into the generator so the model has genuine signal to learn from, rather than random noise:

- **Channel**: search-intent channels (Google Ads) convert better than social/display.
- **Product × audience-age affinity**: e.g., smartphones/shoes skew younger, furniture/books skew older.
- **Budget**: diminishing-returns effect (`log1p` curve) on CTR.
- **Duration**: mild ad-fatigue decay as a campaign runs longer.
- **Location**: small effect for more ad-saturated markets.
- **Gender**: deliberately left with **no** effect — not every real-world feature carries signal, and no relationship was invented for it.

An earlier version of this dataset had the target (`click_through_rate`) generated fully at random, with no relationship to any feature, and also included `clicks`/`impressions` as **model inputs** — which leaked the target (`clicks ≈ impressions × CTR`) and made the app circular (asking users to guess clicks before predicting click-through rate). Both issues are fixed in the current version: the target now has real, encoded structure, and only pre-campaign-known attributes are used as features.

## Model Performance

Trained on an 80/20 split of the 10,000-row synthetic dataset:

| Metric | Value |
|---|---|
| RMSE | 0.0063 |
| R² | 0.862 |

(From `training/model_training.py`'s held-out test split — re-run it to reproduce.)

## Tech Stack

| Layer | Technologies |
|---|---|
| Data | Google BigQuery (training source), local CSV (fallback) |
| Modeling | XGBoost, scikit-learn (preprocessing pipeline) |
| Experiment tracking | MLflow |
| Explainability | SHAP |
| App | Streamlit |
| Deployment | Docker, Google Cloud Run, Cloud Build, Artifact Registry |

## Repository Structure

```
.
├── generator/
│   └── data_generator.py          # Generates synthetic campaign data with real signal
├── data/
│   └── campaign_data.csv          # Generated dataset (also loaded into BigQuery)
├── eda/
│   └── eda.py                     # Exploratory analysis (CTR by product/channel, correlations)
├── preprocessing/
│   └── preprocessing.py           # Reads from BigQuery (or local CSV fallback), fits preprocessor
├── preprocessed/
│   ├── campaign_data_preprocessed.csv
│   └── preprocessor.pkl
├── training/
│   ├── model_training.py          # Trains XGBoost, logs to MLflow, exports deployment artifacts
│   ├── xgb_meta_model.json        # Trained model (XGBoost native format)
│   ├── feature_importance.png
│   └── shap_summary.png
├── Streamlit_CTR_app/
│   ├── app.py                     # Streamlit UI
│   ├── xgb_meta_model.json        # Deployment copy of the trained model
│   ├── preprocessor.pkl           # Deployment copy of the fitted preprocessor
│   ├── requirements.txt           # App-only deps (pinned to match training versions)
│   └── Dockerfile
├── requirements.txt                # Full pipeline deps (generation/preprocessing/training)
├── LICENSE
└── README.md
```

## Steps to Run Locally

```bash
# 1. Set up environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate the synthetic dataset
python generator/data_generator.py

# 3. (Optional) Load it into BigQuery -- skip this and step 4's env var to use
#    the local CSV instead
bq mk --dataset --location=US YOUR_PROJECT_ID:ctr_prediction
bq load --source_format=CSV --skip_leading_rows=1 --autodetect \
  YOUR_PROJECT_ID:ctr_prediction.campaign_data data/campaign_data.csv

# 4. Preprocess (reads from BigQuery if BQ_PROJECT_ID is set, else the local CSV)
export BQ_PROJECT_ID=YOUR_PROJECT_ID   # optional
python preprocessing/preprocessing.py

# 5. Train (writes deployment-ready artifacts directly into Streamlit_CTR_app/)
python training/model_training.py

# 6. Launch the app
cd Streamlit_CTR_app
pip install -r requirements.txt
streamlit run app.py
```

## GCP Deployment

```bash
# One-time setup
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com bigquery.googleapis.com

# Deploy (builds via Cloud Build using Streamlit_CTR_app/Dockerfile, pushes
# to Artifact Registry, deploys to Cloud Run -- all in one command)
gcloud run deploy ctr-streamlit-ui \
  --source Streamlit_CTR_app \
  --region us-central1 \
  --allow-unauthenticated
```

This repo's live deployment runs on Cloud Run's scale-to-zero tier with a monthly billing budget alert configured — realistic cost for light/demo traffic is close to $0/month (Cloud Run, BigQuery, and Artifact Registry all have generous always-free tiers that comfortably cover a 10K-row dataset and occasional requests).

## Lessons Learned

**Pin ML library versions in Docker images, especially scikit-learn.** This repo's Cloud Run deployment initially predicted CTR values as high as 50%+ on inputs that should have predicted ~6% — with no errors or warnings anywhere. Root cause: the preprocessor was pickled locally with `scikit-learn==1.9.0` (which requires Python ≥3.11), but the Dockerfile used `python:3.9` with an unpinned `scikit-learn` in requirements.txt. `pip install` silently resolved to an older, Python-3.9-compatible sklearn version, which loaded the pickle without complaint but transformed inputs incorrectly — scikit-learn only guarantees pickle compatibility within the same version. Fixed by bumping the base image to `python:3.11-slim` and pinning `scikit-learn`/`xgboost` in `Streamlit_CTR_app/requirements.txt` to the exact versions used in training. Caught by manually testing the deployed app in a browser rather than assuming a successful `gcloud run deploy` meant a correct one.

## License

MIT — see [LICENSE](LICENSE).
