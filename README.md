# End-to-End MLOps Project with FastAPI, Streamlit and Docker deployed on Cloud with sucessful CI/CD

## 📌 Overview

This project is a **production-grade end-to-end Machine Learning system** that predicts car prices based on user inputs.
It demonstrates the complete ML lifecycle — from data processing and model training to **cloud deployment with CI/CD and experiment tracking**.

---

## 🧠 Key Highlights

* 🔹 Full ML pipeline (data → preprocessing → training → evaluation → deployment)
* 🔹 REST API using FastAPI
* 🔹 Interactive frontend using Streamlit
* 🔹 Experiment tracking with MLflow (deployed on cloud)
* 🔹 Model & artifact storage using AWS S3
* 🔹 Fully containerized using Docker
* 🔹 Multi-container orchestration using Docker Compose
* 🔹 Deployed on AWS EC2
* 🔹 CI/CD pipeline using GitHub Actions

---

## 🏗️ System Architecture

```text
User → Streamlit UI → FastAPI Backend → ML Model → Prediction
                ↓
        MLflow Tracking Server (Cloud)
                ↓
        AWS S3 (Model & Artifacts Storage)
                ↓
        Dockerized Deployment on EC2
```

---

## 📂 Project Structure

```text
complete_project/
│
├── app/                     # FastAPI backend
├── frontend/                # Streamlit frontend
├── artifacts/               # Saved models
├── notebooks/               # EDA & experiments
├── data/                    # Datasets
├── model/                   # Training logic .pkl file
├── steps/                   # Phases done to write the code
│
├── Dockerfile               # Backend container
├── docker-compose.yml       # Multi-container setup
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* **Programming:** Python
* **ML:** Scikit-learn, Pandas, NumPy
* **Backend:** FastAPI
* **Frontend:** Streamlit
* **Experiment Tracking:** MLflow (Cloud Hosted)
* **Storage:** AWS S3
* **Containerization:** Docker, Docker Compose
* **Cloud:** AWS EC2
* **CI/CD:** GitHub Actions

---

## 🔄 ML Pipeline Workflow

1. Data Ingestion
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Model Training & Evaluation
5. Experiment Tracking with MLflow
6. Model Storage in AWS S3
7. API Deployment (FastAPI)
8. UI Integration (Streamlit)
9. Containerization (Docker)
10. CI/CD Automation (Github Actions)
11. Deployment on AWS EC2

---

## 🐳 Dockerized Setup

### Pull images

```bash
docker pull ishaan0010/complete_project:v1
docker pull ishaan0010/streamlit_carapp:v2
```

### Run

```bash
docker-compose up -d
```

---

## 🌐 Live Application

* 🔹 **Frontend:** http://<EC2-PUBLIC-IP>:8501
* 🔹 **Backend:** http://<EC2-PUBLIC-IP>:8000/docs
* 🔹 **MLflow UI:** http://<EC2-PUBLIC-IP>:5000

---

## ⚡ CI/CD Pipeline

Automated using GitHub Actions:

* Code push triggers pipeline
* Docker images built automatically
* Images pushed to Docker Hub
* Deployment executed on AWS EC2

---

## ☁️ AWS Infrastructure

* EC2 for hosting containers
* S3 for storing models and artifacts
* MLflow tracking server deployed on cloud
* Docker Compose used for orchestration

---

## 📊 Model Details

* **Type:** Regression Model
* **Library:** Scikit-learn
* **Features:**

  * Car Name
  * Company
  * Year
  * Kilometers Driven
  * Fuel Type

---

## 📈 Future Improvements

* 🔹 Kubernetes deployment (EKS)
* 🔹 Monitoring (Prometheus + Grafana)
* 🔹 Feature Store integration
* 🔹 Auto-scaling infrastructure

---

## 🙌 Author

**Ishaan Goyal**

---

## ⭐ Support

If you found this project useful, give it a ⭐ on GitHub!
