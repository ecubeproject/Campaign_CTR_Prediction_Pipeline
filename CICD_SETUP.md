# CI/CD setup — GitHub Actions → Cloud Run (keyless / WIF)

`.github/workflows/deploy-cloudrun.yml` auto-deploys `Streamlit_CTR_app/` to the
`ctr-streamlit-ui` Cloud Run service on every push to `main`. The deploy job
**skips cleanly** until the three GitHub repo variables below exist.

Run these once (paste into a shell with `gcloud` + `gh` authenticated as an
owner of `ctr-prediction-portfolio` / admin of this repo). ~3 minutes.

```bash
PROJECT=ctr-prediction-portfolio
PNUM=177064880623
REPO=ecubeproject/Campaign_CTR_Prediction_Pipeline
SA=github-deployer
SA_EMAIL="$SA@$PROJECT.iam.gserviceaccount.com"

# APIs (already enabled 2026-08-29, harmless to repeat)
gcloud services enable iamcredentials.googleapis.com sts.googleapis.com iam.googleapis.com \
  run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project $PROJECT

# 1. deployer service account + least-privilege roles
gcloud iam service-accounts create $SA --project $PROJECT \
  --display-name "GitHub Actions Cloud Run deployer"

for ROLE in roles/run.admin roles/cloudbuild.builds.editor roles/artifactregistry.admin \
            roles/storage.admin roles/iam.serviceAccountUser roles/logging.viewer; do
  gcloud projects add-iam-policy-binding $PROJECT \
    --member "serviceAccount:$SA_EMAIL" --role $ROLE --condition None
done

# 2. Workload Identity Federation pool + provider, scoped to this repo only
gcloud iam workload-identity-pools create github-pool --project $PROJECT \
  --location global --display-name "GitHub Actions pool"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --project $PROJECT --location global --workload-identity-pool github-pool \
  --display-name "GitHub OIDC" \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition "assertion.repository=='$REPO'" \
  --issuer-uri "https://token.actions.githubusercontent.com"

# 3. let this repo's workflows impersonate the deployer SA
gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL --project $PROJECT \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/$PNUM/locations/global/workloadIdentityPools/github-pool/attribute.repository/$REPO"

# 4. GitHub repo variables (plain vars, not secrets — WIF needs no key)
WIF="projects/$PNUM/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
gh variable set GCP_PROJECT         --repo $REPO --body "$PROJECT"
gh variable set GCP_SERVICE_ACCOUNT --repo $REPO --body "$SA_EMAIL"
gh variable set GCP_WIF_PROVIDER    --repo $REPO --body "$WIF"

# 5. trigger a run and watch it
gh workflow run deploy-cloudrun.yml --repo $REPO
sleep 5 && gh run watch --repo $REPO "$(gh run list --repo $REPO --workflow deploy-cloudrun.yml -L1 --json databaseId -q '.[0].databaseId')"
```

## Teardown

```bash
gcloud iam workload-identity-pools delete github-pool --project $PROJECT --location global
gcloud iam service-accounts delete $SA_EMAIL --project $PROJECT
gh variable delete GCP_PROJECT GCP_SERVICE_ACCOUNT GCP_WIF_PROVIDER --repo $REPO
```
