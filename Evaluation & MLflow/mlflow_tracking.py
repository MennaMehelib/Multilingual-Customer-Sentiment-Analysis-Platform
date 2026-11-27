from pathlib import Path

import json
import pandas as pd
import mlflow


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = Path(__file__).resolve().parent

RESULTS_DIR = EVALUATION_DIR / "evaluation_results"

CLASSICAL_METRICS_PATH = (
    RESULTS_DIR / "classical_metrics.csv"
)

TRANSFORMER_METRICS_PATH = (
    PROJECT_ROOT / "Advanced NLP Model"
    / "results" / "metrics.json"
)


# MLflow tracking location
MLFLOW_DB_PATH = EVALUATION_DIR / "mlflow.db"

MLFLOW_ARTIFACTS_DIR = EVALUATION_DIR / "mlartifacts"

MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# Configure MLflow with SQLite
mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)


# Create experiment if it does not exist
EXPERIMENT_NAME = "Multilingual Sentiment Analysis"

experiment = mlflow.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:
    mlflow.create_experiment(
        EXPERIMENT_NAME,
        artifact_location=MLFLOW_ARTIFACTS_DIR.resolve().as_uri()
    )


mlflow.set_experiment(EXPERIMENT_NAME)

print("MLflow Tracking Initialized Successfully!")
print(f"Database: {MLFLOW_DB_PATH}")
print(f"Artifacts: {MLFLOW_ARTIFACTS_DIR}")



# Load Classical Models metrics
classical_metrics = pd.read_csv(
    CLASSICAL_METRICS_PATH
)


# Validate required columns
required_columns = [
    "split",
    "model",
    "ngram",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro"
]

missing_columns = [
    column
    for column in required_columns
    if column not in classical_metrics.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# Check loaded results
print("\nClassical Models Metrics Loaded Successfully!")

print(f"Total records: {len(classical_metrics)}")

print("\nAvailable Models:")
print(
    classical_metrics[
        ["model", "ngram"]
    ].drop_duplicates()
)

print("\nAvailable Splits:")
print(classical_metrics["split"].unique())

# Load Transformer metrics
with open(
    TRANSFORMER_METRICS_PATH,
    "r",
    encoding="utf-8"
) as file:
    transformer_metrics = json.load(file)


# Validate required metrics
required_transformer_metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1"
]

for metric in required_transformer_metrics:
    if metric not in transformer_metrics:
        raise ValueError(
            f"Missing Transformer metric: {metric}"
        )


# Check loaded results
print("\nTransformer Metrics Loaded Successfully!")

for metric, value in transformer_metrics.items():
    print(f"{metric}: {value:.4f}")

# Log Classical Models to MLflow
for (model, ngram), group in classical_metrics.groupby(
    ["model", "ngram"]
):

    run_name = f"{model} - {ngram}"

    with mlflow.start_run(run_name=run_name):

        # Log model parameters
        mlflow.log_param("model", model)
        mlflow.log_param("ngram", ngram)

        # Log validation and test metrics
        for _, row in group.iterrows():

            split = row["split"]

            mlflow.log_metric(
                f"{split}_accuracy",
                float(row["accuracy"])
            )

            mlflow.log_metric(
                f"{split}_precision_macro",
                float(row["precision_macro"])
            )

            mlflow.log_metric(
                f"{split}_recall_macro",
                float(row["recall_macro"])
            )

            mlflow.log_metric(
                f"{split}_f1_macro",
                float(row["f1_macro"])
            )

        print(f"Logged successfully: {run_name}")


print("\nAll Classical Models Logged to MLflow!")

# Log Transformer to MLflow
with mlflow.start_run(
    run_name="XLM-RoBERTa - Transformer"
):

    # Log model parameters
    mlflow.log_param("model", "XLM-RoBERTa")
    mlflow.log_param("architecture", "Transformer")
    mlflow.log_param("max_length", 64)
    mlflow.log_param("epochs", 3)
    mlflow.log_param("learning_rate", 2e-5)
    mlflow.log_param("weight_decay", 0.01)
    mlflow.log_param("evaluation_split", "test")

    # Log evaluation metrics
    mlflow.log_metric(
        "test_accuracy",
        float(transformer_metrics["accuracy"])
    )

    mlflow.log_metric(
        "test_precision",
        float(transformer_metrics["precision"])
    )

    mlflow.log_metric(
        "test_recall",
        float(transformer_metrics["recall"])
    )

    mlflow.log_metric(
        "test_f1",
        float(transformer_metrics["f1"])
    )


print("\nTransformer Logged to MLflow Successfully!")

# Define artifact directories
ERROR_ANALYSIS_DIR = EVALUATION_DIR / "error_analysis"

MODEL_COMPARISON_DIR = EVALUATION_DIR / "model_comparison"

PREDICTIONS_DIR = EVALUATION_DIR / "predictions"

TRANSFORMER_RESULTS_DIR = (
    PROJECT_ROOT / "Advanced NLP Model" / "results"
)


# Log evaluation artifacts
with mlflow.start_run(
    run_name="Evaluation Artifacts"
):

    artifact_directories = {
        "evaluation_results": RESULTS_DIR,
        "error_analysis": ERROR_ANALYSIS_DIR,
        "model_comparison": MODEL_COMPARISON_DIR,
        "predictions": PREDICTIONS_DIR
    }

    for artifact_name, directory in artifact_directories.items():

        if directory.exists():
            mlflow.log_artifacts(
                str(directory),
                artifact_path=artifact_name
            )

            print(f"Logged artifacts: {artifact_name}")

        else:
            print(f"Directory not found: {directory}")


    # Log Transformer results
    transformer_predictions_path = (
        TRANSFORMER_RESULTS_DIR / "test_predictions.csv"
    )

    if transformer_predictions_path.exists():
        mlflow.log_artifact(
            str(transformer_predictions_path),
            artifact_path="transformer_results"
        )

        print("Logged Transformer predictions")


print("\nEvaluation Artifacts Logged Successfully!")

# Final MLflow Verification
experiment = mlflow.get_experiment_by_name(
    "Multilingual Sentiment Analysis"
)

if experiment is None:
    raise ValueError("MLflow experiment was not found!")


runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    output_format="pandas"
)


print("\n" + "=" * 50)
print("MLFLOW FINAL VERIFICATION")
print("=" * 50)

print(f"Total Runs: {len(runs)}")

if not runs.empty:
    print("\nRegistered Runs:")

    print(
        runs[
            [
                "tags.mlflow.runName",
                "status",
                "run_id"
            ]
        ].to_string(index=False)
    )

    print("\nMLflow Tracking Completed Successfully!")

else:
    print("No runs were found.")