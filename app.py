
import joblib
import pandas as pd
import logging
import json

from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ==========================================
# 1. LOGGING CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Prediction logger
prediction_logger = logging.getLogger("roadsafe.predictions")
prediction_logger.setLevel(logging.INFO)

prediction_handler = RotatingFileHandler(
    LOG_DIR / "predictions.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3
)

prediction_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(message)s")
)

prediction_logger.addHandler(prediction_handler)
prediction_logger.propagate = False


# Error logger
error_logger = logging.getLogger("roadsafe.errors")
error_logger.setLevel(logging.ERROR)

error_handler = RotatingFileHandler(
    LOG_DIR / "errors.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3
)

error_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
)

error_logger.addHandler(error_handler)
error_logger.propagate = False


# ==========================================
# 2. INITIALIZE FASTAPI
# ==========================================

app = FastAPI(
    title="Road Accident Severity Prediction API",
    description="API for predicting road accident severity",
    version="1.0.0"
)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

model = joblib.load(MODEL_PATH)


# ==========================================
# 4. DEFINE INPUT DATA
# ==========================================

class AccidentInput(BaseModel):

    Police_Force: int
    Road_Class_1: int = Field(alias="1st_Road_Class")
    Road_Class_2: int = Field(alias="2nd_Road_Class")

    Road_Type: int
    Junction_Detail: int
    Junction_Control: int

    Pedestrian_Crossing_Human_Control: int = Field(
        alias="Pedestrian_Crossing-Human_Control"
    )

    Pedestrian_Crossing_Physical_Facilities: int = Field(
        alias="Pedestrian_Crossing-Physical_Facilities"
    )

    Light_Conditions: int
    Weather_Conditions: int
    Road_Surface_Conditions: int

    Special_Conditions_at_Site: int
    Carriageway_Hazards: int
    Urban_or_Rural_Area: int

    Day_of_Week: int = Field(ge=0, le=6)

    Speed_limit: float = Field(gt=0)

    Month: int = Field(ge=1, le=12)
    Day: int = Field(ge=1, le=31)
    Hour: int = Field(ge=0, le=23)


# ==========================================
# 5. SEVERITY LABELS
# ==========================================

severity_labels = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}


# ==========================================
# 6. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Road Accident Severity Prediction API",
        "status": "Running",
        "documentation": "/docs"
    }


# ==========================================
# 7. PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict_accident(data: AccidentInput):

    try:

        input_data = pd.DataFrame([{
            "Police_Force": data.Police_Force,
            "1st_Road_Class": data.Road_Class_1,
            "2nd_Road_Class": data.Road_Class_2,
            "Road_Type": data.Road_Type,
            "Junction_Detail": data.Junction_Detail,
            "Junction_Control": data.Junction_Control,

            "Pedestrian_Crossing-Human_Control":
                data.Pedestrian_Crossing_Human_Control,

            "Pedestrian_Crossing-Physical_Facilities":
                data.Pedestrian_Crossing_Physical_Facilities,

            "Light_Conditions": data.Light_Conditions,
            "Weather_Conditions": data.Weather_Conditions,
            "Road_Surface_Conditions": data.Road_Surface_Conditions,

            "Special_Conditions_at_Site":
                data.Special_Conditions_at_Site,

            "Carriageway_Hazards": data.Carriageway_Hazards,
            "Urban_or_Rural_Area": data.Urban_or_Rural_Area,
            "Day_of_Week": data.Day_of_Week,

            "Speed_limit": data.Speed_limit,
            "Month": data.Month,
            "Day": data.Day,
            "Hour": data.Hour
        }])

        prediction = int(model.predict(input_data)[0])

        severity = severity_labels[prediction]

        # Log successful prediction
        prediction_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": input_data.to_dict(orient="records")[0],
            "predicted_severity_code": prediction,
            "predicted_severity": severity
        }

        prediction_logger.info(json.dumps(prediction_record))

        return {
            "predicted_severity_code": prediction,
            "predicted_severity": severity
        }

    except Exception:

        # Record technical error and traceback
        error_logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal server error."
        )
