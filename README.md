
# RoadSafe AI – Road Accident Severity Prediction

## 1. Project Overview

RoadSafe AI is a machine learning-based application that classifies road accident severity into three categories:

- Fatal
- Serious
- Slight

The application uses a trained Random Forest model and provides predictions through a web interface.

## 2. Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- MLflow
- FastAPI
- Streamlit
- Docker
- Docker Compose
- Docker Hub

## 3. System Architecture

The application consists of two services:

1. Backend – FastAPI loads the trained Random Forest model and provides a prediction API.
2. Frontend – Streamlit provides an interactive interface for entering accident details and viewing predictions.

## 4. Model Information

- Dataset: UK STATS19 Accident Dataset (2019)
- Algorithm: Random Forest
- Target: Accident Severity
- Classes: Fatal (1), Serious (2), Slight (3)

## 5. MLOps Workflow

1. Data collection and preprocessing
2. Exploratory data analysis
3. Model training and evaluation
4. Experiment tracking using MLflow
5. Model serialization using Joblib
6. API development using FastAPI
7. Frontend development using Streamlit
8. Containerization using Docker
9. Container orchestration using Docker Compose
10. Image publishing using Docker Hub

## 6. Docker Hub Images

- Backend: jemymathew/roadsafe-backend
- Frontend: jemymathew/roadsafe-frontend

## 7. Running the Application

Make sure Docker Desktop is running.

Execute:

```bash
docker compose up -d
```

Open the application:

- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

To stop the application:

```bash
docker compose down
```

## 8. Disclaimer

This application classifies recorded accident conditions using historical data. It does not predict whether an accident will occur and is not intended for operational road-safety decisions.
