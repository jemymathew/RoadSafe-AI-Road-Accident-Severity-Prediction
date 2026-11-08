
import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "data/accident_2019.csv",
    low_memory=False
)

print("Dataset shape:", df.shape)


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

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 6. PREPROCESSING
# ==========================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numeric_transformer, numerical_features),
    ("categorical", categorical_transformer, categorical_features)
])


# ==========================================
# 7. DECISION TREE PIPELINE
# ==========================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        random_state=42
    ))
])


# ==========================================
# 8. HYPERPARAMETER SEARCH SPACE
# ==========================================

param_grid = {
    "classifier__criterion": ["gini", "entropy", "log_loss"],

    "classifier__max_depth": [
        5, 10, 15, 20, 25, 30, None
    ],

    "classifier__min_samples_split": [
        2, 5, 10, 20
    ],

    "classifier__min_samples_leaf": [
        2, 5, 10, 20
    ],

    "classifier__class_weight": [
        "balanced",
        None,
        {1: 5, 2: 2, 3: 1},
        {1: 8, 2: 3, 3: 1}
    ]
}


# ==========================================
# 9. CROSS-VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ==========================================
# 10. RANDOMIZED SEARCH
# ==========================================

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=20,
    scoring="f1_macro",
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

print("\nStarting Decision Tree hyperparameter tuning...")
print("Testing 20 parameter combinations with 3-fold CV.")

search.fit(X_train, y_train)


# ==========================================
# 11. DISPLAY BEST RESULTS
# ==========================================

print("\n====================================")
print("BEST DECISION TREE RESULTS")
print("====================================")

print("Best Cross-validation Macro F1:",
      round(search.best_score_, 4))

print("\nBest Parameters:")

for parameter, value in search.best_params_.items():
    print(parameter, ":", value)


# ==========================================
# 12. SAVE MODEL AND RESULTS
# ==========================================

os.makedirs("models", exist_ok=True)

results_df = pd.DataFrame(search.cv_results_)

results_df = results_df.sort_values(
    by="rank_test_score"
)

results_df.to_csv(
    "models/decision_tree_tuning_results.csv",
    index=False
)

joblib.dump(
    search.best_estimator_,
    "models/tuned_decision_tree.joblib"
)

print("\nDecision Tree tuning completed!")
print("Results saved successfully.")
print("Model saved: models/tuned_decision_tree.joblib")
