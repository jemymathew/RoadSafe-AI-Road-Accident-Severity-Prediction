
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

# ======================================
# 1. LOAD DATA
# ======================================

df = pd.read_csv(
    "data/accident_2019.csv",
    low_memory=False
)

# ======================================
# 2. REMOVE UNSUITABLE COLUMNS
# ======================================

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

# ======================================
# 3. DATE AND TIME PROCESSING
# ======================================

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

df["Speed_limit"] = df["Speed_limit"].replace(-1, np.nan)

# ======================================
# 4. FEATURES AND TARGET
# ======================================

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

target = "Accident_Severity"

X = df[categorical_features + numerical_features].copy()
y = df[target].copy()

# Keep categorical variables as categorical values
for col in categorical_features:
    X[col] = X[col].astype("object")

# ======================================
# 5. TRAIN TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training distribution before SMOTENC:")
print(y_train.value_counts().sort_index())

print("\nTesting distribution:")
print(y_test.value_counts().sort_index())

# ======================================
# 6. IMPUTATION
# ======================================

# Impute numerical and categorical values
for col in numerical_features:
    X_train[col] = X_train[col].fillna(X_train[col].median())
    X_test[col] = X_test[col].fillna(X_train[col].median())

for col in categorical_features:
    mode_value = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(mode_value)
    X_test[col] = X_test[col].fillna(mode_value)

# Convert categorical columns to strings
for col in categorical_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# ======================================
# 7. SMOTENC
# ======================================

categorical_indices = [
    X_train.columns.get_loc(col)
    for col in categorical_features
]

smote = SMOTENC(
    categorical_features=categorical_indices,
    sampling_strategy="not majority",
    random_state=42,
    k_neighbors=5
)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

print("\nTraining distribution after SMOTENC:")
print(pd.Series(y_train_resampled).value_counts().sort_index())

# ======================================
# 8. ONE HOT ENCODING
# ======================================

preprocessor = ColumnTransformer([
    (
        "numeric",
        "passthrough",
        numerical_features
    ),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
])

# ======================================
# 9. RANDOM FOREST
# ======================================

model = SklearnPipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight=None,
        max_depth=35,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    ))
])

# ======================================
# 10. TRAIN AND EVALUATE
# ======================================

print("\nTraining Random Forest with SMOTENC...")

model.fit(X_train_resampled, y_train_resampled)

predictions = model.predict(X_test)

print("\nSMOTENC RESULTS")

print(
    "Accuracy:",
    round(accuracy_score(y_test, predictions), 4)
)

print(
    "Balanced Accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4)
)

print(
    "Macro F1:",
    round(f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    ), 4)
)

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
