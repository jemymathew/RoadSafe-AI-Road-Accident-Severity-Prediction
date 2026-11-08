
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "Running"


def test_prediction_endpoint():
    payload = {
        "Police_Force": 1,
        "1st_Road_Class": 3,
        "2nd_Road_Class": 6,
        "Road_Type": 6,
        "Junction_Detail": 0,
        "Junction_Control": -1,
        "Pedestrian_Crossing-Human_Control": 0,
        "Pedestrian_Crossing-Physical_Facilities": 0,
        "Light_Conditions": 1,
        "Weather_Conditions": 1,
        "Road_Surface_Conditions": 1,
        "Special_Conditions_at_Site": 0,
        "Carriageway_Hazards": 0,
        "Urban_or_Rural_Area": 1,
        "Day_of_Week": 2,
        "Speed_limit": 30,
        "Month": 6,
        "Day": 15,
        "Hour": 14
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_severity"] in [
        "Fatal", "Serious", "Slight"
    ]

    assert result["predicted_severity_code"] in [1, 2, 3]


def test_invalid_day_of_week():
    payload = {
        "Police_Force": 1,
        "1st_Road_Class": 3,
        "2nd_Road_Class": 6,
        "Road_Type": 6,
        "Junction_Detail": 0,
        "Junction_Control": -1,
        "Pedestrian_Crossing-Human_Control": 0,
        "Pedestrian_Crossing-Physical_Facilities": 0,
        "Light_Conditions": 1,
        "Weather_Conditions": 1,
        "Road_Surface_Conditions": 1,
        "Special_Conditions_at_Site": 0,
        "Carriageway_Hazards": 0,
        "Urban_or_Rural_Area": 1,
        "Day_of_Week": 8,
        "Speed_limit": 30,
        "Month": 6,
        "Day": 15,
        "Hour": 14
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
