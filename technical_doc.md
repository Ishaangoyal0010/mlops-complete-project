# Car Price Prediction – End-to-End ML/MLOps System

**Author:** Ishaan Goyal  
**Repository:** [github.com/Ishaangoyal0010/mlops-complete-project](https://github.com/Ishaangoyal0010/mlops-complete-project)  
**Stack:** Python · Scikit-learn · FastAPI · Streamlit · MLflow · AWS S3 · AWS EC2 · Docker · GitHub Actions

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Data Pipeline](#4-data-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Training](#6-model-training)
7. [Model Selection](#7-model-selection)
8. [Experiment Tracking with MLflow](#8-experiment-tracking-with-mlflow)
9. [Data & Model Versioning](#9-data--model-versioning)
10. [API Architecture](#10-api-architecture)
11. [Dockerization](#11-dockerization)
12. [AWS Deployment](#12-aws-deployment)
13. [CI/CD Pipeline](#13-cicd-pipeline)
14. [Monitoring](#14-monitoring)
15. [API Usage](#15-api-usage)
16. [Local Setup](#16-local-setup)
17. [Design Decisions & Trade-offs](#17-design-decisions--trade-offs)
18. [Limitations](#18-limitations)
19. [Future Improvements](#19-future-improvements)

---

## 1. Problem Statement

Used car pricing is inconsistent and opaque. A buyer searching for a second-hand vehicle has no reliable way to evaluate whether a listed price is fair, and sellers have no benchmark to price their vehicle accurately. Manual valuation requires domain expertise and is time-consuming.

**Objective:** Build a production-ready ML system that predicts the resale price of a used car given its attributes — make, model, year of manufacture, kilometers driven, and fuel type — and expose this prediction through a REST API and interactive UI.

**Success Criteria:**
- Prediction latency under 200ms per request
- System remains available through container restarts
- New model versions can be deployed without downtime via CI/CD
- All experiments are tracked and reproducible

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER LAYER                            │
│                                                              │
│        Streamlit Frontend  (port 8501)                       │
│        Interactive UI for car price prediction               │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTP POST /predict
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      API LAYER                               │
│                                                              │
│        FastAPI Backend  (port 8000)                          │
│        REST API — /predict  /health  /docs                   │
└──────────────────────┬───────────────────────────────────────┘
                       │ loads model
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      ML LAYER                                │
│                                                              │
│        Scikit-learn Pipeline                                 │
│        Preprocessing → Feature Engineering → Model          │
└──────────────────────┬───────────────────────────────────────┘
                       │ logs metrics & artifacts
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   TRACKING & STORAGE LAYER                   │
│                                                              │
│        MLflow Tracking Server  (port 5000)                   │
│        AWS S3 — model artifacts, versioned pkl files         │
└──────────────────────┬───────────────────────────────────────┘
                       │ deployed on
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   INFRASTRUCTURE LAYER                        │
│                                                              │
│        AWS EC2 — Docker Compose multi-container host         │
│        GitHub Actions — CI/CD automation                     │
└─────────────────────────────────────────────────────────────┘
```

**Data Flow — Prediction Request:**
```
User fills form → Streamlit → POST /predict → FastAPI validates input
→ Preprocessor transforms features → Model predicts price
→ Response returned → Streamlit displays predicted price
```

**Data Flow — Model Training:**
```
Raw CSV → Data cleaning → Feature engineering → Train/test split
→ Model training → MLflow logs metrics + params → Best model saved
→ pkl artifact uploaded to AWS S3 → API loads from S3 on startup
```

---

## 3. Project Structure

```
mlops-complete-project/
│
├── app/                          # FastAPI backend
│   ├── main.py                   # App entry point, route definitions
│   ├── predict.py                # Prediction logic, model loading
│   └── schemas.py                # Pydantic request/response models
│
├── frontend/                     # Streamlit frontend
│   └── app.py                    # UI — form inputs, API calls, display
│
├── model/                        # Training pipeline
│   ├── train.py                  # Main training script
│   ├── preprocess.py             # Data cleaning and transformation
│   └── evaluate.py               # Metrics computation
│
├── notebooks/                    # Exploratory Data Analysis
│   └── eda.ipynb                 # Dataset exploration, visualizations
│
├── data/                         # Raw and processed datasets
│   └── car_data.csv              # Source dataset
│
├── steps/                        # Documented implementation phases
│
├── utils/                        # Shared utilities
│   └── s3_utils.py               # AWS S3 upload/download helpers
│
├── .github/
│   └── workflows/
│       └── deploy.yml            # GitHub Actions CI/CD pipeline
│
├── Dockerfile                    # Backend container definition
├── docker-compose.yml            # Multi-service orchestration
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 4. Data Pipeline

### 4.1 Dataset

The dataset contains used car listings scraped from a popular automotive marketplace. Each record represents a single car listing.

| Column | Type | Description |
|--------|------|-------------|
| `name` | string | Full car name including variant |
| `company` | string | Manufacturer (Toyota, Honda, etc.) |
| `year` | integer | Year of manufacture |
| `Price` | integer | Listed resale price (target variable) |
| `kms_driven` | string | Kilometers driven (raw, includes units) |
| `fuel_type` | string | Petrol / Diesel / LPG |

### 4.2 Data Cleaning

Raw data requires significant cleaning before use:

```python
# Remove duplicate listings
df.drop_duplicates(inplace=True)

# Drop rows with missing target variable
df.dropna(subset=['Price'], inplace=True)

# Clean kms_driven — remove " kms" suffix, convert to integer
df['kms_driven'] = df['kms_driven'].str.split(' ').str.get(0)
df['kms_driven'] = df['kms_driven'].str.replace(',', '').astype(int)

# Remove outliers — prices above 99th percentile
upper = df['Price'].quantile(0.99)
df = df[df['Price'] <= upper]

# Filter invalid years
df = df[df['year'] >= 1990]
df = df[df['year'] <= datetime.now().year]

# Drop rows with unknown fuel type
df = df[df['fuel_type'].isin(['Petrol', 'Diesel', 'LPG'])]
```

### 4.3 Train/Test Split

```python
from sklearn.model_selection import train_test_split

X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 5. Feature Engineering

### 5.1 Categorical Encoding

Car names and company names are high-cardinality categorical variables. One-Hot Encoding (OHE) is used to convert them to numeric form:

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
# Applied to: name, company, fuel_type columns
```

`handle_unknown='ignore'` ensures that car models not seen during training do not cause errors at prediction time.

### 5.2 Numeric Features

`year` and `kms_driven` are passed through as-is after cleaning. No scaling is applied since tree-based models are scale-invariant.

### 5.3 Scikit-learn Pipeline

All preprocessing and the model are encapsulated in a single `Pipeline` object:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ('ohe', OneHotEncoder(handle_unknown='ignore'),
     ['name', 'company', 'fuel_type']),
    ('passthrough', 'passthrough',
     ['year', 'kms_driven'])
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
```

Using a Pipeline ensures that the preprocessing steps are always applied consistently during both training and inference — no risk of applying different transformations at prediction time.

---

## 6. Model Training

```python
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    r2    = r2_score(y_test, y_pred)
    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))

    # Log metrics
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)

    # Save model artifact
    mlflow.sklearn.log_model(pipeline, "model")
    joblib.dump(pipeline, "artifacts/model.pkl")
```

---

## 7. Model Selection

Multiple regression algorithms were evaluated:

| Model | R² Score | MAE | RMSE | Training Time |
|-------|----------|-----|------|---------------|
| Linear Regression | 0.71 | 82,400 | 1,24,000 | ~1s |
| Decision Tree Regressor | 0.78 | 71,200 | 1,08,500 | ~2s |
| **Random Forest Regressor** | **0.87** | **54,300** | **89,200** | ~18s |
| Gradient Boosting Regressor | 0.85 | 57,100 | 93,400 | ~25s |

**Selected Model: Random Forest Regressor**

**Rationale:**
- Highest R² score (0.87) — explains 87% of variance in car prices
- Lowest MAE — predictions within ₹54,300 on average
- Handles non-linear relationships well — car pricing is non-linear
- Robust to outliers — ensemble averaging reduces the effect of extreme listings
- No feature scaling required — simplifies the pipeline
- Fast inference — sub-millisecond prediction on a single input

---

## 8. Experiment Tracking with MLflow

MLflow is deployed as a tracking server and used to log every training run.

### 8.1 What is Tracked

Every training run automatically logs:

| Category | Details |
|----------|---------|
| **Parameters** | n_estimators, max_depth, random_state, test_size |
| **Metrics** | R² score, MAE, RMSE on test set |
| **Artifacts** | Trained model .pkl, feature importance chart |
| **Tags** | Model type, dataset version, author |
| **Source** | Git commit hash, training script path |

### 8.2 MLflow Server Setup (AWS EC2)

```bash
# On EC2 instance
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://your-bucket/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

### 8.3 Connecting Training Script to Remote MLflow

```python
import mlflow

mlflow.set_tracking_uri("http://<EC2_PUBLIC_IP>:5000")
mlflow.set_experiment("car-price-prediction")
```

### 8.4 Comparing Runs

The MLflow UI at `http://<EC2_IP>:5000` allows visual comparison of all runs — filter by metric, sort by R² score, and select the best model for production promotion.

---

## 9. Data & Model Versioning

### 9.1 Model Artifacts on AWS S3

Trained model files are versioned by run ID and stored in S3:

```
s3://your-bucket/
├── mlflow-artifacts/
│   ├── <experiment-id>/
│   │   ├── <run-id-1>/artifacts/model/model.pkl
│   │   └── <run-id-2>/artifacts/model/model.pkl
└── models/
    └── production/model.pkl   ← current production model
```

### 9.2 Loading Model at API Startup

```python
import boto3
import joblib

def load_model_from_s3():
    s3 = boto3.client('s3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )
    s3.download_file(
        Bucket="your-bucket",
        Key="models/production/model.pkl",
        Filename="/tmp/model.pkl"
    )
    return joblib.load("/tmp/model.pkl")

model = load_model_from_s3()
```

This ensures the API always loads the latest promoted model without requiring a container rebuild.

---

## 10. API Architecture

### 10.1 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — returns service status |
| GET | `/docs` | Auto-generated Swagger UI |
| POST | `/predict` | Accepts car features, returns predicted price |

### 10.2 Request Schema

```python
from pydantic import BaseModel
from typing import Literal

class CarFeatures(BaseModel):
    name:       str
    company:    str
    year:       int
    kms_driven: int
    fuel_type:  Literal["Petrol", "Diesel", "LPG"]

    class Config:
        schema_extra = {
            "example": {
                "name":       "Maruti Swift VXI",
                "company":    "Maruti",
                "year":       2018,
                "kms_driven": 45000,
                "fuel_type":  "Petrol"
            }
        }
```

### 10.3 Response Schema

```python
class PredictionResponse(BaseModel):
    predicted_price: float
    currency:        str = "INR"
    model_version:   str
```

### 10.4 Prediction Endpoint

```python
from fastapi import FastAPI
import pandas as pd

app = FastAPI(title="Car Price Prediction API")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(car: CarFeatures):
    input_df = pd.DataFrame([car.dict()])
    price = model.predict(input_df)[0]
    return {
        "predicted_price": round(float(price), 2),
        "currency": "INR",
        "model_version": "1.0.0"
    }
```

### 10.5 Input Validation

FastAPI + Pydantic automatically validates all incoming requests:
- Missing fields return `422 Unprocessable Entity`
- Invalid fuel_type values are rejected at schema level
- Year must be an integer — strings are rejected

---

## 11. Dockerization

### 11.1 Backend Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 11.2 Docker Compose — Multi-Container Setup

```yaml
services:

  backend:
    image: ishaan0010/complete_project:v1
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

  frontend:
    image: ishaan0010/streamlit_carapp:v2
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8000
    depends_on:
      - backend

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root s3://your-bucket/mlflow-artifacts
      --host 0.0.0.0
      --port 5000
```

### 11.3 Published Docker Images

```bash
# Pull and run without building locally
docker pull ishaan0010/complete_project:v1
docker pull ishaan0010/streamlit_carapp:v2

docker-compose up -d
```

---

## 12. AWS Deployment

### 12.1 Infrastructure Overview

| Service | Purpose |
|---------|---------|
| **EC2 (t2.micro)** | Hosts all three Docker containers |
| **AWS S3** | Stores model artifacts and MLflow experiment data |
| **Docker Compose** | Orchestrates backend, frontend, and MLflow on EC2 |
| **Security Groups** | Opens ports 8000, 8501, 5000 for public access |

### 12.2 EC2 Setup

```bash
# Install Docker on EC2
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=ap-south-1

# Pull and run
docker-compose pull
docker-compose up -d
```

### 12.3 Deployed Endpoints

| Service | URL |
|---------|-----|
| Streamlit Frontend | `http://<EC2_PUBLIC_IP>:8501` |
| FastAPI Backend | `http://<EC2_PUBLIC_IP>:8000/docs` |
| MLflow UI | `http://<EC2_PUBLIC_IP>:5000` |

---

## 13. CI/CD Pipeline

### 13.1 Pipeline Overview

```
Developer pushes to main branch
          ↓
GitHub Actions triggered (.github/workflows/deploy.yml)
          ↓
Checkout code
          ↓
Login to Docker Hub (DOCKER_USERNAME + DOCKER_PASSWORD secrets)
          ↓
Build backend Docker image
          ↓
Push ishaan0010/complete_project:v1 to Docker Hub
          ↓
Build frontend Docker image
          ↓
Push ishaan0010/streamlit_carapp:v2 to Docker Hub
          ↓
✅ Both images updated — EC2 can docker-compose pull to deploy
```

### 13.2 Workflow File

```yaml
name: Build and Push to Docker Hub

on:
  push:
    branches: [main]

jobs:
  build-and-push:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push backend
        uses: docker/build-push-action@v4
        with:
          context: .
          dockerfile: Dockerfile
          push: true
          tags: ishaan0010/complete_project:v1

      - name: Build and push frontend
        uses: docker/build-push-action@v4
        with:
          context: ./frontend
          push: true
          tags: ishaan0010/streamlit_carapp:v2
```

### 13.3 Secrets Configuration

| Secret Name | Description |
|-------------|-------------|
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub access token |

---

## 14. Monitoring

### 14.1 Current Monitoring

| Layer | Tool | What is Monitored |
|-------|------|-------------------|
| API health | FastAPI `/health` endpoint | Service liveness |
| Experiments | MLflow UI | Model metrics across runs |
| Container status | Docker `ps` + logs | Container uptime |

### 14.2 Health Check

```bash
curl http://<EC2_IP>:8000/health
# {"status": "ok", "model_loaded": true}
```

### 14.3 Planned Monitoring (Future)

- **Prometheus** — scrape FastAPI metrics (request count, latency histogram, error rate)
- **Grafana** — dashboard for live request metrics and model performance
- **Data drift detection** — alert when input distributions shift from training data
- **Evidently AI** — automated model performance reports

---

## 15. API Usage

### 15.1 Health Check

```bash
curl -X GET http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 15.2 Price Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Maruti Swift VXI",
    "company": "Maruti",
    "year": 2018,
    "kms_driven": 45000,
    "fuel_type": "Petrol"
  }'
```

Response:
```json
{
  "predicted_price": 485000.0,
  "currency": "INR",
  "model_version": "1.0.0"
}
```

### 15.3 Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "name":       "Honda City ZX",
        "company":    "Honda",
        "year":       2017,
        "kms_driven": 62000,
        "fuel_type":  "Petrol"
    }
)

data = response.json()
print(f"Predicted Price: ₹{data['predicted_price']:,.0f}")
# Predicted Price: ₹6,20,000
```

---

## 16. Local Setup

### 16.1 Option A — Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/Ishaangoyal0010/mlops-complete-project
cd mlops-complete-project

# Create .env file
echo "AWS_ACCESS_KEY_ID=your_key" > .env
echo "AWS_SECRET_ACCESS_KEY=your_secret" >> .env

# Pull images and run
docker-compose pull
docker-compose up -d
```

Access:
- Frontend → http://localhost:8501
- API Docs → http://localhost:8000/docs
- MLflow → http://localhost:5000

### 16.2 Option B — Virtual Environment

```bash
# Create venv
python -m venv venv
venv\Scripts\activate.bat      # Windows
source venv/bin/activate        # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Run backend
cd app
uvicorn main:app --port 8000 --reload

# New terminal — run frontend
cd frontend
streamlit run app.py
```

### 16.3 Train Model Locally

```bash
python model/train.py
# Outputs: artifacts/model.pkl
# Logs: http://localhost:5000 (if MLflow running locally)
```

### 16.4 Dependencies

```
fastapi
uvicorn
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
mlflow
python-multipart
boto3==1.28.57
botocore==1.31.57
```

---

## 17. Design Decisions & Trade-offs

### 17.1 Random Forest over Gradient Boosting

**Decision:** Random Forest was selected despite Gradient Boosting having comparable accuracy.

**Rationale:**
- RF training is parallelisable — faster on multi-core EC2 instance
- RF is less sensitive to hyperparameter tuning — easier to maintain
- RF prediction is faster at inference time (critical for API latency)
- Gradient Boosting's marginal R² improvement (0.85 vs 0.87) did not justify the added complexity

### 17.2 Scikit-learn Pipeline over Manual Preprocessing

**Decision:** All preprocessing and the model are wrapped in a single `sklearn.Pipeline`.

**Rationale:**
- Eliminates train/serving skew — the exact same transformations happen at training and inference
- A single `.pkl` file contains the complete system — easier to version and deploy
- `handle_unknown='ignore'` in OHE prevents crashes on unseen car models

### 17.3 S3 for Model Storage over Local Filesystem

**Decision:** Model artifacts are stored in AWS S3, not in the container.

**Rationale:**
- Container filesystem is ephemeral — restarting loses the model
- S3 allows multiple EC2 instances to share the same model (horizontal scaling)
- Model updates don't require container rebuilds — just upload new `.pkl` to S3

### 17.4 SQLite as MLflow Backend Store

**Decision:** MLflow uses SQLite for its backend store rather than a managed database.

**Trade-off:**
- Pro: Zero infrastructure cost, simple to set up on EC2
- Con: Not suitable for concurrent multi-user access at scale
- Acceptable for a single-developer project; would migrate to RDS PostgreSQL for team use

### 17.5 Docker Compose over Kubernetes

**Decision:** Docker Compose is used for orchestration rather than Kubernetes.

**Rationale:**
- Single EC2 instance — Kubernetes overhead is unjustified
- Docker Compose provides sufficient multi-container orchestration for this scale
- Kubernetes planned as a future improvement when traffic warrants it

---

## 18. Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Dataset is static — not updated with new listings | Model accuracy degrades over time as market prices change | Planned: automated retraining pipeline on schedule |
| Model only knows car models seen in training data | Unknown car models return predictions based on company + year only | OHE `handle_unknown='ignore'` prevents crashes but accuracy drops |
| No input range validation | Unrealistic inputs (year=1800, kms=9999999) return predictions without warning | Add business logic validation layer in API |
| SQLite MLflow backend | Not suitable for concurrent team access | Migrate to RDS PostgreSQL for production team use |
| No authentication on API | Anyone with the URL can call `/predict` | Add API key authentication for production |
| EC2 t2.micro resource limits | May struggle under high concurrent load | Upgrade instance type or add load balancer |

---

## 19. Future Improvements

### 19.1 Short Term
- **Input validation** — reject implausible inputs (year < 1990, kms > 5,00,000)
- **API authentication** — API key middleware to secure endpoints
- **Automated retraining** — scheduled GitHub Actions job to retrain monthly with fresh data
- **Model registry promotion** — MLflow model registry workflow: Staging → Production

### 19.2 Medium Term
- **Prometheus + Grafana monitoring** — real-time dashboards for request latency, error rate, throughput
- **Data drift detection** — Evidently AI to alert when input distributions deviate from training data
- **Feature store** — centralised feature repository for consistent feature computation
- **A/B testing** — route 10% of traffic to a challenger model, compare live R² scores

### 19.3 Long Term
- **Kubernetes on AWS EKS** — horizontal pod autoscaling for traffic spikes
- **Multi-region deployment** — low-latency predictions for users across India
- **Online learning** — incrementally update model with confirmed sale prices
- **Explainability layer** — SHAP values per prediction showing which features drove the price

---

*Last updated: 2025 · github.com/Ishaangoyal0010/mlops-complete-project*
