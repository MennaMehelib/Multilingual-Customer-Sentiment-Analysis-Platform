from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_DIR = Path(__file__).resolve().parent

PREDICTIONS_PATH = (
    EVALUATION_DIR / "predictions" / "classical_predictions.csv"
)

OUTPUT_DIR = EVALUATION_DIR / "evaluation_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Fixed label order
LABELS = ["negative", "neutral", "positive"]


print("Evaluation paths configured successfully!")

# Load predictions
predictions_df = pd.read_csv(PREDICTIONS_PATH)


# Validate required columns
REQUIRED_COLUMNS = {
    "label",
    "predicted",
    "model",
    "ngram",
    "split",
    "language"
}

missing_columns = REQUIRED_COLUMNS - set(predictions_df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# Validate labels
if not predictions_df["label"].isin(LABELS).all():
    raise ValueError("Unexpected labels found in true labels.")

if not predictions_df["predicted"].isin(LABELS).all():
    raise ValueError("Unexpected labels found in predictions.")


# Validate dataset splits
if not predictions_df["split"].isin(
    ["validation", "test"]
).all():
    raise ValueError("Unexpected dataset split found.")


print(f"Loaded {len(predictions_df)} prediction records.")
print("Prediction data validation passed!")

# Additional integrity checks

expected_groups = 8

actual_groups = predictions_df.groupby(
    ["split", "model", "ngram"]
).ngroups

if actual_groups != expected_groups:
    raise ValueError(
        f"Expected {expected_groups} experiment groups, "
        f"but found {actual_groups}."
    )


expected_samples_per_group = 13500

group_sizes = predictions_df.groupby(
    ["split", "model", "ngram"]
).size()

if not (group_sizes == expected_samples_per_group).all():
    raise ValueError("Unexpected number of samples in an experiment group.")


print("All integrity checks passed!")

# Calculate evaluation metrics
def calculate_metrics(y_true, y_pred):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        )
    }

    return metrics

# Evaluate all experiments
evaluation_results = []

for (split_name, model_name, ngram_name), group in predictions_df.groupby(
    ["split", "model", "ngram"]
):

    metrics = calculate_metrics(
        group["label"],
        group["predicted"]
    )

    result = {
        "split": split_name,
        "model": model_name,
        "ngram": ngram_name,
        **metrics
    }

    evaluation_results.append(result)

    print(
        f"Evaluated: {model_name} + {ngram_name} "
        f"({split_name})"
    )


# Convert results into DataFrame
metrics_df = pd.DataFrame(evaluation_results)


print("\nAll experiments evaluated successfully!")
print(metrics_df)

# Select best model using validation F1
validation_results = metrics_df[
    metrics_df["split"] == "validation"
]

if validation_results.empty:
    raise ValueError("No validation results available for model selection.")

best_model = validation_results.loc[
    validation_results["f1_macro"].idxmax()
]


print("\nBest Classical Model (Validation):")
print(f"Model: {best_model['model']}")
print(f"N-grams: {best_model['ngram']}")
print(f"Validation F1 Macro: {best_model['f1_macro']:.4f}")

# Save evaluation metrics
METRICS_PATH = OUTPUT_DIR / "classical_metrics.csv"

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# Save best model selection
BEST_MODEL_PATH = OUTPUT_DIR / "best_classical_model.txt"

with open(BEST_MODEL_PATH, "w", encoding="utf-8") as file:
    file.write("Best Classical Model (selected using Validation F1)\n")
    file.write(f"Model: {best_model['model']}\n")
    file.write(f"N-grams: {best_model['ngram']}\n")
    file.write(f"Validation F1 Macro: {best_model['f1_macro']:.4f}\n")


print("\nEvaluation results saved successfully!")
print(f"Metrics: {METRICS_PATH}")
print(f"Best model: {BEST_MODEL_PATH}")

# Generate confusion matrices
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"
CONFUSION_DIR.mkdir(parents=True, exist_ok=True)


for (split_name, model_name, ngram_name), group in predictions_df.groupby(
    ["split", "model", "ngram"]
):

    cm = confusion_matrix(
        group["label"],
        group["predicted"],
        labels=LABELS
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(cm, cmap="Blues")

    ax.set_title(
        f"{model_name} - {ngram_name} ({split_name})"
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))

    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)

    safe_model = model_name.lower().replace(" ", "_")
    safe_ngram = ngram_name.lower().replace(" + ", "_plus_").replace(" ", "_")

    filename = f"{safe_model}_{safe_ngram}_{split_name}.png"

    fig.savefig(
        CONFUSION_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


print("\nAll confusion matrices saved successfully!")

# Generate classification reports
REPORTS_DIR = OUTPUT_DIR / "classification_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


for (split_name, model_name, ngram_name), group in predictions_df.groupby(
    ["split", "model", "ngram"]
):

    report = classification_report(
        group["label"],
        group["predicted"],
        labels=LABELS,
        target_names=LABELS,
        zero_division=0
    )

    safe_model = model_name.lower().replace(" ", "_")
    safe_ngram = ngram_name.lower().replace(" + ", "_plus_").replace(" ", "_")

    filename = f"{safe_model}_{safe_ngram}_{split_name}.txt"

    report_path = REPORTS_DIR / filename

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(
            f"Model: {model_name}\n"
            f"N-grams: {ngram_name}\n"
            f"Split: {split_name}\n\n"
        )

        file.write(report)


print("\nAll classification reports saved successfully!")