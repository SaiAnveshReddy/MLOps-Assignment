# CardioRisk-MLOps

Heart disease risk classifier with a full MLOps lifecycle - EDA, feature
engineering, MLflow-tracked model training, CI/CD, containerized serving,
Kubernetes deployment, and Prometheus/Grafana monitoring.

**Author:** J Sai Anvesh Reddy (BITS ID: 2025AE05969)
**Course:** AIMLCZG523 - MLOps, Assignment 1, BITS Pilani WILP

Full write-up: [`report/report.pdf`](report/report.pdf)

## Project layout

```
data/raw/            Cleveland heart disease dataset (303 rows, as downloaded from UCI)
data/processed/      Cleaned CSV produced by heart_mlops.data
notebooks/           EDA notebook (Task 1)
src/heart_mlops/      Data cleaning, feature engineering, training, inference
src/api/              FastAPI serving app, logging, Prometheus metrics
tests/                Pytest unit tests (data, features, inference, API)
models/              Saved model + preprocessing pipeline + MLflow plots
docker/              Dockerfile for the serving API
k8s/                 Kubernetes Deployment + Service manifests (Minikube)
monitoring/          Prometheus config + Grafana provisioning/dashboard
.github/workflows/   CI pipeline (lint, test, train)
report/              10-page project report (Markdown source + PDF)
screenshots/         Deployment/CI/monitoring screenshots for the report
```

## 1. Setup

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; use .venv\Scripts\activate on cmd/PowerShell
pip install -r requirements.txt
pip install -e .
```

The raw dataset already ships in `data/raw/` (no download step needed to run
the project). To fetch a fresh copy instead:

```bash
curl -o data/raw/processed.cleveland.data \
  https://archive.ics.uci.edu/static/public/45/data/processed.cleveland.data
```

## 2. Data cleaning + EDA (Task 1)

```bash
python -m heart_mlops.data          # writes data/processed/heart_cleaned.csv
jupyter notebook notebooks/01_eda.ipynb
```

## 3. Train + track experiments (Tasks 2 & 3)

```bash
python -m heart_mlops.train
mlflow ui --backend-store-uri file:./mlruns   # open http://localhost:5000
```

This trains Logistic Regression and Random Forest with grid-search +
5-fold stratified CV, logs every run to MLflow, and saves the best
pipeline to `models/heart_risk_model.pkl` (see `models/model_metadata.json`
for the winning model, params, and test metrics).

## 4. Tests + lint (Task 5)

```bash
flake8 src tests --max-line-length=100
pytest tests/ -v
```

CI runs the same two commands plus `python -m heart_mlops.train` on every
push/PR via `.github/workflows/ci.yml`.

## 5. Run the API locally (Task 6)

```bash
uvicorn api.main:app --app-dir src --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age":57,"sex":1,"cp":4,"trestbps":140,"chol":289,"fbs":0,"restecg":2,
  "thalach":145,"exang":1,"oldpeak":2.0,"slope":2,"ca":1,"thal":7
}'
```

## 6. Docker (Task 6)

```bash
docker build -f docker/Dockerfile -t cardiorisk-api:local .
docker run -p 8000:8000 cardiorisk-api:local
```

## 7. Kubernetes / Minikube (Task 7)

See [`k8s/README.md`](k8s/README.md) for the full step-by-step (build inside
Minikube's Docker daemon, apply manifests, expose via `minikube service`).

## 8. Monitoring (Task 8)

```bash
docker compose up --build
# API:        http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin) -> "CardioRisk API - Monitoring" dashboard
```

Request logs are written as structured JSON to stdout
(`src/api/logging_config.py`), which `docker logs`/`kubectl logs` capture
directly.

## Design choices worth knowing about

- **Target**: UCI's 0-4 severity column is collapsed to binary (0 = no
  disease, 1 = disease present).
- **Missing values**: `ca`/`thal` (6 of 303 rows) are mode-imputed rather
  than dropped, to keep the full sample for cross-validation.
- **Engineered feature**: `hr_reserve = (220 - age) - thalach`, kept for
  interpretability (see `notebooks/01_eda.ipynb`, section 6, for the honest
  correlation comparison against `thalach` alone).
- **Random seed**: `5969` (last 4 digits of my BITS ID) used everywhere for
  reproducibility.

Full rationale, metrics, architecture diagram, and screenshots are in
[`report/report.pdf`](report/report.pdf).
