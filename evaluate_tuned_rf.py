
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "data/accident_2019.csv",
    low_memory=False
)

# ==========================================
# 2. PREPROCESS DATA
# ==========================================

columns_to_drop = [
    "Accident_Index",
    "LSOA_of_Accident_Location",
    "Did_Police_Officer_Attend_Scene_of_Accident",
    "Number_of_Casualties",
    "Number_of_Vehicles",
    "Local_Authority_(District)",
    "Local_Authority_(Highway)",
    "1st_Road_Number",
    "2nd_Road_Number",
    "Location_Easting_OSGR",
    "Location_Northing_OSGR",
    "Longitude",
    "Latitude"
]

df = df.drop(columns=columns_to_drop)

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Day_of_Week"] = df["Date"].dt.day_of_week

df["Hour"] = pd.to_datetime(
    df["Time"],
    format="%H:%M",
    errors="coerce"
).dt.hour

df = df.drop(columns=["Date", "Time"])

df["Speed_limit"] = df["Speed_limit"].replace(
    -1, float("nan")
)

categorical_features = [
    "Police_Force",
    "1st_Road_Class",
    "2nd_Road_Class",
    "Road_Type",
    "Junction_Detail",
    "Junction_Control",
    "Pedestrian_Crossing-Human_Control",
    "Pedestrian_Crossing-Physical_Facilities",
    "Light_Conditions",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "Special_Conditions_at_Site",
    "Carriageway_Hazards",
    "Urban_or_Rural_Area",
    "Day_of_Week"
]

numerical_features = [
    "Speed_limit",
    "Month",
    "Day",
    "Hour"
]

X = df[categorical_features + numerical_features].copy()
y = df["Accident_Severity"].copy()

for col in categorical_features:
    X[col] = X[col].astype("object")

# ==========================================
# 3. SAME TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# 4. LOAD TUNED MODEL
# ==========================================

model = joblib.load(
    "models/tuned_rf_cv_best.joblib"
)

# ==========================================
# 5. PREDICTION
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# 6. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, predictions)

balanced_acc = balanced_accuracy_score(
    y_test, predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

print("\nTUNED RANDOM FOREST RESULTS")
print("=" * 45)

print("Accuracy:", round(accuracy, 4))
print("Balanced Accuracy:", round(balanced_acc, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=[1, 2, 3],
    target_names=["Fatal", "Serious", "Slight"],
    zero_division=0
))

print("\nConfusion Matrix:")

print(confusion_matrix(
    y_test,
    predictions,
    labels=[1, 2, 3]
))
