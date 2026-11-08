
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

print("Original dataset shape:", df.shape)


# ==========================================
# 2. REMOVE UNSUITABLE FEATURES
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


# ==========================================
# 3. DATE AND TIME PROCESSING
# ==========================================

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


# ==========================================
# 4. DEFINE FEATURES AND TARGET
# ==========================================

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
# 5. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 6. LOAD TUNED DECISION TREE
# ==========================================

model = joblib.load(
    "models/tuned_decision_tree.joblib"
)

print("\nTuned Decision Tree model loaded successfully!")


# ==========================================
# 7. PREDICTION
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 8. EVALUATION METRICS
# ==========================================

accuracy = accuracy_score(
    y_test, predictions
)

balanced_acc = balanced_accuracy_score(
    y_test, predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 50)
print("TUNED DECISION TREE RESULTS")
print("=" * 50)

print("\nAccuracy:", round(accuracy, 4))
print("Balanced Accuracy:", round(balanced_acc, 4))
print("Macro F1:", round(macro_f1, 4))


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=[1, 2, 3],
    target_names=["Fatal", "Serious", "Slight"],
    zero_division=0
))


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[1, 2, 3]
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 12. SAVE EVALUATION RESULTS
# ==========================================

import os
import json

os.makedirs("models/evaluation", exist_ok=True)

report = classification_report(
    y_test,
    predictions,
    labels=[1, 2, 3],
    target_names=["Fatal", "Serious", "Slight"],
    output_dict=True,
    zero_division=0
)

results = {
    "model": "Tuned Decision Tree",
    "accuracy": accuracy,
    "balanced_accuracy": balanced_acc,
    "macro_f1": macro_f1,
    "classification_report": report,
    "confusion_matrix": cm.tolist()
}

with open(
    "models/evaluation/tuned_decision_tree_report.json",
    "w"
) as file:
    json.dump(results, file, indent=4)

print("\nEvaluation report saved successfully!")
