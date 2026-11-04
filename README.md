# ChurnGuard: Customer Churn Prediction API

[![CI](https://github.com/dushanth-001/churnguard/actions/workflows/ci.yml/badge.svg)](https://github.com/dushanth-001/churnguard/actions/workflows/ci.yml)

A machine learning service that scores how likely a telecom customer is to churn, so a retention team can decide who to contact first. The project covers the full path from raw data to a tested, containerised and cloud-deployed API.

## Business framing

- **Who uses it:** an internal retention team, not the customers themselves.
- **Decision it supports:** which customers get a retention call or offer.
- **Why recall:** a missed churner costs more than an unnecessary offer, so the model is tuned for recall rather than accuracy. The dataset is imbalanced (about 26.5% churn), so accuracy alone is misleading: a model that always predicts "no churn" is already about 73.5% accurate.

## Results (held-out test set)

Measured with `python src/evaluate.py` on 1057 customers (evaluated on 2026-10-29), decision threshold 0.5.

| Metric | Value |
|---|---|
| Recall | 79.0% |
| Precision | 51.1% |
| F1 | 0.62 |
| Accuracy | 74.4% |
| ROC AUC | 0.841 |

Of the 281 customers who actually churned in the test set, the model flagged 222 and missed 59. It also flagged 212 customers who did not churn (false alarms), which is the price paid for higher recall.

Model: XGBoost. Hyperparameters were chosen with randomized search scored on recall, and class imbalance is handled with `scale_pos_weight`.

### How the model was chosen

Validation-set results from the exploration phase (the exploration notebook is not included in this repository):

| Model | Recall | Precision |
|---|---|---|
| Logistic Regression (baseline) | 0.59 | 0.65 |
| Logistic Regression + class weight | 0.81 | 0.51 |
| Random Forest (tuned) | 0.85 | 0.51 |
| XGBoost (tuned) | 0.86 | 0.52 |

SHAP analysis during exploration pointed to contract type, tenure, fibre-optic internet, electronic-check payment and monthly charges as the strongest churn drivers.

## Architecture

```mermaid
flowchart LR
    A[Raw CSV data] --> B[Cleaning and feature engineering]
    B --> C[XGBoost model and scaler]
    C --> D[FastAPI service]
    D --> E[Docker image]
    E --> F[GitHub Container Registry]
    F --> G[Azure Container Apps]
    H[GitHub Actions CI] -->|tests, docker build, publish| F
```

## Run it locally

```bash
docker run -p 8000:8000 -e API_KEY=choose-a-secret ghcr.io/dushanth-001/churnguard:latest
```

Open http://localhost:8000/docs for the interactive API page.

## API

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /` | none | health check |
| `POST /predict` | `X-API-Key` header | churn prediction for one customer |

`/predict` rejects requests without the correct key. If the `API_KEY` environment variable is not set, it rejects every request (it fails closed). Inputs are validated: binary features must be 0 or 1.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: choose-a-secret" \
  -d @examples/sample_request.json
```

Example response (values for the sample request):

```json
{"churn_prediction": 1, "churn_probability": 0.848}
```

## Project structure
app.py                               FastAPI application
src/
├── data_preprocessing.py            Cleaning and feature engineering
├── train.py                         Training and tuning functions
├── evaluate.py                      Test-set evaluation, writes metrics.json
├── predict.py                       Model loading and prediction helpers
└── logger.py                        Logging setup
models/                              Trained model and scaler
tests/                               Unit tests and API tests
examples/                            Sample API request
.github/workflows/                   CI pipeline
Dockerfile
config.yaml                          Settings reference
metrics.json                         Latest test-set metrics


## Testing and CI

Every push to `main` runs three jobs in GitHub Actions:

1. `test`: installs pinned dependencies in a clean environment and runs pytest.
2. `docker`: builds the image, starts the container and checks that the API responds.
3. `publish`: pushes the image to GitHub Container Registry (only after the first two pass).

## Monitoring and drift detection

- Every prediction is logged as one JSON line (prediction, probability, latency and four numeric inputs). No customer identifiers are logged.
- `GET /stats` (API key required) returns in-memory request counts, error count, predicted churn rate and latency. These counters reset when the container restarts or scales to zero.
- `python src/drift.py --logs <exported log file>` compares logged inputs and predicted probabilities with a reference sample from the validation set (`models/reference.csv`), using the Population Stability Index (PSI) and a KS test. A feature with PSI of 0.25 or more is flagged as drifted, and at least 100 logged predictions are required.
- This is a script run on exported logs, not an automatic alert. It is unit-tested on synthetic data and has not been run on real production traffic.

## Retraining pipeline

`.github/workflows/retrain.yml` is triggered manually (Actions, Run workflow, optionally with a CSV URL that has the same columns as the training data). It runs `src/retrain.py`, which:

1. trains a candidate XGBoost model on the data (same hyperparameters, scaler fitted on the training split only);
2. scores the candidate and the current model on the same held-out test split;
3. applies a promotion gate: the candidate is rejected if recall or ROC AUC drops by more than 0.02, and nothing is promoted if the two models are equivalent (within 0.001);
4. if the candidate passes, pushes a `retrain/run-<id>` branch with the new model, scaler, drift reference sample and metrics. A person reviews and merges the pull request, which triggers CI and image publishing.

On the public static dataset the candidate equals the current model, so the expected result is `no_change`. The pipeline has not been run on genuinely new data, and deploying a merged model to Azure is still a manual step.

## Limitations

- The data is the public IBM Telco dataset, not real company data.
- The decision threshold is fixed at 0.5. With real cost figures for a missed churner and a retention offer, the threshold should be chosen from those costs.
- The API expects already-encoded features, and `tenure`, `MonthlyCharges` and `TotalCharges` must already be standardised. A production version would accept raw customer fields and apply the saved scaler inside the service.
- Retraining is a manually triggered workflow with a promotion gate. It has not been exercised on genuinely new data.
- The demo deployment runs on a free-tier plan that scales to zero, so the first request after idle time is slow.

## Roadmap

- Automated drift alerts and scheduled retraining on real incoming data
- A prediction endpoint that accepts raw customer fields
- Threshold selection from business cost data
