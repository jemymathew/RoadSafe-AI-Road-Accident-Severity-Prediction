
import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. MLFLOW CONFIGURATION
# ==========================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(
    "Road Accident Severity Prediction"
)

print("MLflow tracking URI:", mlflow.get_tracking_uri())

# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "data/accident_2019.csv",
    low_memory=False
)

print("\nOriginal dataset shape:", df.shape)

# ==========================================
# 3. REMOVE UNSUITABLE FEATURES
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
# 4. DATE AND TIME PROCESSING
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

# ==========================================
# 5. HANDLE INVALID SPEED LIMIT
# ==========================================

df["Speed_limit"] = df["Speed_limit"].replace(
    -1, float("nan")
)

# ==========================================
# 6. DEFINE FEATURES AND TARGET
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

target = "Accident_Severity"

X = df[categorical_features + numerical_features].copy()
y = df[target].copy()

for col in categorical_features:
    X[col] = X[col].astype("object")

print("\nInput features:", X.shape)

print("\nTarget distribution:")
print(y.value_counts().sort_index())

# ==========================================
# 7. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 8. PREPROCESSING PIPELINE
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
# 9. DEFINE MODELS
# ==========================================

models = {

    "Decision Tree - Balanced": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=20,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest - No Weights": RandomForestClassifier(
        n_estimators=100,
        class_weight=None,
        max_depth=25,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42
    ),

    "Random Forest - Balanced": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        max_depth=25,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42
    ),

    "Random Forest - Custom Weights": RandomForestClassifier(
        n_estimators=100,
        class_weight={1: 8, 2: 2, 3: 1},
        max_depth=25,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42
    ),

    "Random Forest - Balanced Subsample Tuned": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        max_depth=35,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
}

# ==========================================
# 10. TRAIN, EVALUATE AND TRACK WITH MLFLOW
# ==========================================

results = {}

os.makedirs("models", exist_ok=True)
os.makedirs("models/mlflow_reports", exist_ok=True)

best_model = None
best_model_name = None
best_macro_f1 = -1
best_run_id = None

for name, classifier in models.items():

    print("\n" + "=" * 55)
    print("Training:", name)
    print("=" * 55)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Start one MLflow run per model
    with mlflow.start_run(run_name=name) as run:

        # ------------------------------
        # MODEL TRAINING
        # ------------------------------

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        # ------------------------------
        # EVALUATION
        # ------------------------------

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

        report = classification_report(
            y_test,
            predictions,
            labels=[1, 2, 3],
            target_names=["Fatal", "Serious", "Slight"],
            output_dict=True,
            zero_division=0
        )

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[1, 2, 3]
        )

        # ------------------------------
        # PRINT RESULTS
        # ------------------------------

        print("\nAccuracy:", round(accuracy, 4))
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
        print(cm)

        # ------------------------------
        # LOG PARAMETERS
        # ------------------------------

        mlflow.log_param("model_name", name)
        mlflow.log_param("dataset", "accident_2019.csv")
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("testing_samples", len(X_test))
        mlflow.log_param("input_features", X.shape[1])
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)

        model_params = classifier.get_params()

        for param_name, param_value in model_params.items():

            if param_value is not None:
                mlflow.log_param(
                    param_name,
                    str(param_value)
                )

        # ------------------------------
        # LOG METRICS
        # ------------------------------

        mlflow.log_metrics({
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,

            "fatal_precision": report["Fatal"]["precision"],
            "fatal_recall": report["Fatal"]["recall"],
            "fatal_f1": report["Fatal"]["f1-score"],

            "serious_precision": report["Serious"]["precision"],
            "serious_recall": report["Serious"]["recall"],
            "serious_f1": report["Serious"]["f1-score"],

            "slight_precision": report["Slight"]["precision"],
            "slight_recall": report["Slight"]["recall"],
            "slight_f1": report["Slight"]["f1-score"]
        })

        # ------------------------------
        # SAVE EVALUATION REPORT
        # ------------------------------

        safe_name = name.lower().replace(" ", "_").replace("-", "_")

        report_path = os.path.join(
            "models",
            "mlflow_reports",
            f"{safe_name}_report.json"
        )

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        cm_path = os.path.join(
            "models",
            "mlflow_reports",
            f"{safe_name}_confusion_matrix.csv"
        )

        pd.DataFrame(
            cm,
            index=["Fatal", "Serious", "Slight"],
            columns=["Predicted Fatal", "Predicted Serious", "Predicted Slight"]
        ).to_csv(cm_path)

        mlflow.log_artifact(
            report_path,
            artifact_path="evaluation"
        )

        mlflow.log_artifact(
            cm_path,
            artifact_path="evaluation"
        )

        # ------------------------------
        # LOG TRAINED MODEL
        # ------------------------------

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="trained_model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )

        # ------------------------------
        # STORE RESULTS
        # ------------------------------

        results[name] = {
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_acc,
            "Macro F1": macro_f1
        }

        # ------------------------------
        # SELECT BEST MODEL
        # ------------------------------

        if macro_f1 > best_macro_f1:

            best_macro_f1 = macro_f1
            best_model = pipeline
            best_model_name = name
            best_run_id = run.info.run_id

# ==========================================
# 11. MODEL COMPARISON
# ==========================================

print("\nMODEL COMPARISON")

results_df = pd.DataFrame(results).T

print(results_df.round(4))

comparison_path = "models/model_comparison.csv"

results_df.to_csv(comparison_path)

# ==========================================
# 12. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    "models/best_model.joblib"
)

print("\nBest model:", best_model_name)
print("Best Macro F1:", round(best_macro_f1, 4))
print("Best MLflow Run ID:", best_run_id)
print("Model saved successfully!")

# ==========================================
# 13. FEATURE IMPORTANCE
# ==========================================

print("\nFEATURE IMPORTANCE ANALYSIS")

best_classifier = best_model.named_steps["classifier"]

feature_names = (
    best_model.named_steps["preprocessor"]
    .get_feature_names_out()
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": best_classifier.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(importance_df.head(15).to_string(index=False))

importance_path = "models/feature_importance.csv"

importance_df.to_csv(
    importance_path,
    index=False
)

# Plot feature importance
top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Feature Importances - Random Forest")
plt.tight_layout()

plot_path = "models/feature_importance.png"

plt.savefig(plot_path, dpi=300)
plt.close()

# Log comparison and feature importance artifacts
# under the selected model's MLflow run.

with mlflow.start_run(run_id=best_run_id):

    mlflow.log_artifact(comparison_path)
    mlflow.log_artifact(importance_path)
    mlflow.log_artifact(plot_path)

    mlflow.set_tag("selected_best_model", "true")
    mlflow.set_tag("selection_metric", "macro_f1")

print("\nAll model experiments tracked successfully in MLflow!")
print("Best model:", best_model_name)
print("Best run ID:", best_run_id)
