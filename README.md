# RoadSafe AI – Road Accident Severity Prediction

## 1. Project Overview

RoadSafe AI is a machine learning-based application developed to classify road accident severity into three categories:

- Fatal (Class 1)
- Serious (Class 2)
- Slight (Class 3)

The system uses historical accident records from the UK STATS19 2019 dataset. A trained Random Forest classifier predicts accident severity based on recorded road, environmental, and accident-related conditions.

The application provides a web interface through Streamlit and a prediction API through FastAPI. Docker and Docker Compose are used for containerization and deployment.

**Important:** This is a retrospective classification system. It does not predict whether an accident will occur.

## 2. Dataset Description

- **Dataset:** UK STATS19 Road Accident Dataset (2019)
- **Original records:** 117,536
- **Input features:** 19
- **Target variable:** Accident_Severity
- **Classes:** Fatal, Serious, Slight

### Class Distribution

| Severity | Class | Records |
|---|---:|---:|
| Fatal | 1 | 1,658 |
| Serious | 2 | 23,422 |
| Slight | 3 | 92,456 |

The dataset contains numerical and categorical features describing accident conditions.

## 3. Data Preprocessing

The preprocessing workflow includes:

1. Handling missing numerical values using median imputation.
2. Handling missing categorical values using most-frequent imputation.
3. Encoding categorical variables using One-Hot Encoding.
4. Splitting the dataset into training and testing sets using stratified sampling.
5. Applying class-imbalance handling strategies during model experiments.

The dataset was divided into 80% training data and 20% testing data, using a random state of 42.

Preprocessing and classification are implemented as part of the machine learning pipeline.

## 4. Machine Learning Models

The project explores multiple classification algorithms:

- Decision Tree
- Random Forest
- Logistic Regression
- Class-weighted and class-balanced model configurations
- SMOTENC-based imbalance handling experiments

The final deployed application uses a trained **Random Forest classifier**.

Model evaluation considers:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Confusion Matrix
- Feature Importance

## 5. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Experiment Tracking | MLflow |
| Model Serialization | Joblib |
| Backend API | FastAPI |
| Frontend | Streamlit |
| API Server | Uvicorn |
| Containerization | Docker |
| Container Orchestration | Docker Compose |
| Image Registry | Docker Hub |
| Testing | Pytest |

## 6. System Architecture

The application consists of two main services.

### Backend – FastAPI

- Loads the trained Random Forest model.
- Accepts accident-condition inputs.
- Performs severity classification.
- Returns the predicted severity and class code.
- Records successful predictions in a log file.
- Records prediction errors for troubleshooting.

### Frontend – Streamlit

- Provides an interactive accident-condition input form.
- Communicates with the FastAPI backend.
- Displays the predicted accident severity.
- Displays prediction statistics and history.
- Provides a prediction monitoring dashboard.

### Architecture Flow

User Input → Streamlit Frontend → FastAPI Backend → Random Forest Model → Prediction Result → Monitoring Dashboard

## 7. MLOps Implementation

The project incorporates the following MLOps components:

1. Dataset preprocessing and preparation.
2. Model experimentation and comparison.
3. Experiment tracking using MLflow.
4. Model serialization using Joblib.
5. REST API development using FastAPI.
6. Interactive application development using Streamlit.
7. API input validation.
8. Prediction and error logging.
9. Monitoring dashboard for prediction summaries and history.
10. Docker-based containerization.
11. Docker Compose-based service management.
12. Docker Hub image publishing.
13. Automated testing using Pytest and GitHub Actions.

### Prediction Monitoring

The application maintains prediction logs containing timestamps, input information, and predicted severity.

The Streamlit monitoring dashboard displays:

- Total predictions.
- Fatal, Serious, and Slight prediction counts.
- Daily prediction trends.
- Prediction history.

This provides basic prediction monitoring. Automated model-drift detection and automatic retraining are not implemented.

## 8. API Documentation

The backend provides the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Check API status |
| POST | `/predict` | Predict accident severity |

Interactive API documentation is available through Swagger UI:

`http://localhost:8000/docs`

## 9. Docker Hub Images

The application uses two Docker Hub images:

- **Backend:** `jemymathew/roadsafe-backend:latest`
- **Frontend:** `jemymathew/roadsafe-frontend:latest`

## 10. Running the Application

### Prerequisites

- Docker Desktop
- Docker Compose

### Step 1: Clone the repository

```bash
git clone https://github.com/jemymathew/RoadSafe-AI-Road-Accident-Severity-Prediction.git
cd RoadSafe-AI-Road-Accident-Severity-Prediction
```

### Step 2: Start the application

Make sure Docker Desktop is running.

```bash
docker compose pull
docker compose up -d
```

### Step 3: Access the application

| Service | URL |
|---|---|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000 |
| Swagger API Documentation | http://localhost:8000/docs |

### Step 4: Stop the application

```bash
docker compose down
```

### Useful Docker Commands

Check running containers:

```bash
docker compose ps
```

View backend logs:

```bash
docker compose logs backend
```

View frontend logs:

```bash
docker compose logs frontend
```

## 11. Testing

Automated tests are implemented using Pytest.

Run the tests locally using:

```bash
pip install -r requirements-dev.txt
python -m pytest -v
```

GitHub Actions is configured to execute the automated tests as part of the CI workflow.

## 12. Model Limitations

- The model relies on historical accident records.
- Predictions represent learned statistical associations, not causal relationships.
- Class imbalance can affect the prediction performance of minority classes.
- Predictions may not generalize to different geographical regions or future conditions.
- The monitoring dashboard provides basic prediction statistics and does not implement automated model-drift detection.
- Model retraining requires a separate data preparation, training, evaluation, and validation process.

The application is intended for academic and demonstration purposes and should not be used as a substitute for professional road-safety assessments or operational decisions.

## 13. Future Enhancements

- Automated model performance monitoring.
- Model-drift detection.
- Scheduled model retraining.
- Model version management.
- Cloud-based deployment.
- Improved prediction performance for minority severity classes.

## 14. Repository

**GitHub:** https://github.com/jemymathew/RoadSafe-AI-Road-Accident-Severity-Prediction

**Docker Hub:** https://hub.docker.com/u/jemymathew

---

**Project:** RoadSafe AI – Road Accident Severity Prediction  
**Course:** MLOps Audit Course