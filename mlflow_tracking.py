
import mlflow
import mlflow.sklearn

# Set experiment name
mlflow.set_experiment("Road Accident Severity Prediction")

# Start an experiment run
with mlflow.start_run(run_name="Random Forest - SMOTENC"):

    # Log model details
    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("resampling", "SMOTENC")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 35)
    mlflow.log_param("class_weight", "None")

    # Log evaluation metrics
    mlflow.log_metric("accuracy", 0.6188)
    mlflow.log_metric("balanced_accuracy", 0.4092)
    mlflow.log_metric("macro_f1", 0.3644)

print("MLflow experiment tracking completed!")
