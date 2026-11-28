from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_DIR = Path(__file__).resolve().parent

PREDICTIONS_PATH = (
    EVALUATION_DIR / "predictions" / "classical_predictions.csv"
)

OUTPUT_DIR = EVALUATION_DIR / "error_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("Error analysis paths configured successfully!")

# Load predictions
predictions_df = pd.read_csv(PREDICTIONS_PATH)


# Validate required columns
REQUIRED_COLUMNS = {
    "sample_id",
    "label",
    "predicted",
    "language",
    "model",
    "ngram",
    "split",
    "confidence"
}

missing_columns = REQUIRED_COLUMNS - set(predictions_df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# Identify incorrect predictions
predictions_df["is_error"] = (
    predictions_df["label"] != predictions_df["predicted"]
)

# Validate confidence values
if predictions_df["confidence"].isna().any():
    raise ValueError("Missing confidence values found!")

if not predictions_df["confidence"].between(0, 1).all():
    raise ValueError("Confidence values must be between 0 and 1!")

print("Confidence values validation passed!")

print(f"Loaded {len(predictions_df)} prediction records.")
print(f"Total incorrect predictions: {predictions_df['is_error'].sum()}")
print("Prediction errors identified successfully!")


# Load previously calculated evaluation metrics
METRICS_PATH = (
    EVALUATION_DIR / "evaluation_results" / "classical_metrics.csv"
)

metrics_df = pd.read_csv(METRICS_PATH)


# Select the best model using Validation F1 Macro
validation_results = metrics_df[
    metrics_df["split"] == "validation"
]

if validation_results.empty:
    raise ValueError("No validation results found!")


best_model_row = validation_results.loc[
    validation_results["f1_macro"].idxmax()
]

best_model = best_model_row["model"]
best_ngram = best_model_row["ngram"]


# Filter test predictions for the selected model
best_predictions = predictions_df[
    (predictions_df["split"] == "test")
    & (predictions_df["model"] == best_model)
    & (predictions_df["ngram"] == best_ngram)
].copy()


if best_predictions.empty:
    raise ValueError("No test predictions found for the best model!")

# Validate selected test predictions
EXPECTED_TEST_SAMPLES = 13500

if len(best_predictions) != EXPECTED_TEST_SAMPLES:
    raise ValueError(
        f"Expected {EXPECTED_TEST_SAMPLES} test samples, "
        f"but found {len(best_predictions)}."
    )

if best_predictions["sample_id"].duplicated().any():
    raise ValueError("Duplicate sample IDs found in test predictions!")

print("Test prediction integrity checks passed!")

print("\nSelected Best Classical Model:")
print(f"Model: {best_model}")
print(f"N-grams: {best_ngram}")
print(f"Validation F1 Macro: {best_model_row['f1_macro']:.4f}")
print(f"Test samples selected: {len(best_predictions)}")

# Analyze errors by language
language_errors = (
    best_predictions
    .groupby("language")
    .agg(
        total_samples=("is_error", "count"),
        total_errors=("is_error", "sum")
    )
    .reset_index()
)

language_errors["error_rate"] = (
    language_errors["total_errors"]
    / language_errors["total_samples"]
)

language_errors["accuracy"] = (
    1 - language_errors["error_rate"]
)


# Save language error analysis
language_errors.to_csv(
    OUTPUT_DIR / "errors_by_language.csv",
    index=False
)

print("\nError Analysis by Language:")
print(language_errors)


# Analyze errors by actual sentiment
sentiment_errors = (
    best_predictions
    .groupby("label")
    .agg(
        total_samples=("is_error", "count"),
        total_errors=("is_error", "sum")
    )
    .reset_index()
)

sentiment_errors["error_rate"] = (
    sentiment_errors["total_errors"]
    / sentiment_errors["total_samples"]
)

sentiment_errors["accuracy"] = (
    1 - sentiment_errors["error_rate"]
)


# Save sentiment error analysis
sentiment_errors.to_csv(
    OUTPUT_DIR / "errors_by_sentiment.csv",
    index=False
)

print("\nError Analysis by Sentiment:")
print(sentiment_errors)


# Analyze misclassification patterns
misclassification_patterns = (
    best_predictions[best_predictions["is_error"]]
    .groupby(["label", "predicted"])
    .size()
    .reset_index(name="error_count")
    .sort_values("error_count", ascending=False)
)


# Save misclassification patterns
misclassification_patterns.to_csv(
    OUTPUT_DIR / "misclassification_patterns.csv",
    index=False
)

print("\nMost Common Misclassification Patterns:")
print(misclassification_patterns)

# Analyze confidence of incorrect predictions
errors_df = best_predictions[
    best_predictions["is_error"]
].copy()

errors_df["confidence_level"] = errors_df["confidence"].apply(
    lambda score: (
        "High Confidence" if score >= 0.80
        else "Low/Medium Confidence"
    )
)

# Calculate confidence error statistics
if not errors_df.empty:

    confidence_errors = (
        errors_df
        .groupby("confidence_level")
        .size()
        .reset_index(name="error_count")
    )

    confidence_errors["percentage"] = (
        confidence_errors["error_count"]
        / len(errors_df) * 100
    )

else:

    confidence_errors = pd.DataFrame(
        columns=["confidence_level", "error_count", "percentage"]
    )

    print("No incorrect predictions available for confidence analysis.")


# Save confidence error analysis
confidence_errors.to_csv(
    OUTPUT_DIR / "confidence_error_analysis.csv",
    index=False
)

print("\nConfidence Analysis of Incorrect Predictions:")
print(confidence_errors)

# Extract misclassified examples
misclassified_examples = errors_df[
    [
        "sample_id",
        "cleaned_text",
        "language",
        "label",
        "predicted",
        "confidence"
    ]
].copy()

# Sort by confidence to inspect highly confident mistakes first
misclassified_examples = misclassified_examples.sort_values(
    "confidence",
    ascending=False
)

# Save misclassified examples
misclassified_examples.to_csv(
    OUTPUT_DIR / "misclassified_examples.csv",
    index=False
)

print("\nMisclassified Examples:")
print(f"Total incorrect examples: {len(misclassified_examples)}")
print("\nTop 10 high-confidence mistakes:")
print(misclassified_examples.head(10).to_string(index=False))


# Plot error rate by language
plt.figure(figsize=(8, 5))

plt.bar(
    language_errors["language"],
    language_errors["error_rate"] * 100
)

plt.title("Error Rate by Language")
plt.xlabel("Language")
plt.ylabel("Error Rate (%)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_rate_by_language.png",
    dpi=300
)

plt.close()

print("Error rate by language chart saved successfully!")


# Plot error rate by sentiment
plt.figure(figsize=(8, 5))

plt.bar(
    sentiment_errors["label"],
    sentiment_errors["error_rate"] * 100
)

plt.title("Error Rate by Sentiment")
plt.xlabel("Actual Sentiment")
plt.ylabel("Error Rate (%)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "error_rate_by_sentiment.png",
    dpi=300
)

plt.close()

print("Error rate by sentiment chart saved successfully!")


# Plot most common misclassification patterns
if not misclassification_patterns.empty:

    misclassification_patterns["pattern"] = (
        misclassification_patterns["label"]
        + " → "
        + misclassification_patterns["predicted"]
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        misclassification_patterns["pattern"],
        misclassification_patterns["error_count"]
    )

    plt.title("Most Common Misclassification Patterns")
    plt.xlabel("Actual Sentiment → Predicted Sentiment")
    plt.ylabel("Number of Errors")

    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "misclassification_patterns.png",
        dpi=300
    )

    plt.close()

    print("Misclassification patterns chart saved successfully!")

else:
    print("No misclassification errors found.")


# Plot confidence distribution of incorrect predictions
if not errors_df.empty:

    plt.figure(figsize=(8, 5))

    plt.hist(
        errors_df["confidence"],
        bins=20,
        edgecolor="black"
    )

    plt.axvline(
        x=0.80,
        linestyle="--",
        label="High Confidence Threshold (0.80)"
    )

    plt.title("Confidence Distribution of Incorrect Predictions")
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Number of Incorrect Predictions")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "error_confidence_distribution.png",
        dpi=300
    )

    plt.close()

    print("Error confidence distribution chart saved successfully!")

else:
    print("No errors available for confidence analysis.")


# Generate Error Analysis Summary
total_samples = len(best_predictions)
total_errors = int(best_predictions["is_error"].sum())

overall_error_rate = (
    total_errors / total_samples * 100
    if total_samples > 0
    else 0
)

summary = f"""
CLASSICAL NLP ERROR ANALYSIS SUMMARY
=====================================

Selected Model:
Model: {best_model}
N-grams: {best_ngram}
Validation F1 Macro: {best_model_row['f1_macro']:.4f}

Test Set:
Total Samples: {total_samples}
Incorrect Predictions: {total_errors}
Correct Predictions: {total_samples - total_errors}
Overall Error Rate: {overall_error_rate:.2f}%

Error Analysis Files:
- errors_by_language.csv
- errors_by_sentiment.csv
- misclassification_patterns.csv
- confidence_error_analysis.csv
- misclassified_examples.csv

Generated Charts:
- error_rate_by_language.png
- error_rate_by_sentiment.png
- misclassification_patterns.png
- error_confidence_distribution.png
"""

summary_path = OUTPUT_DIR / "error_analysis_summary.txt"

summary_path.write_text(summary, encoding="utf-8")

print("\nError Analysis Summary:")
print(summary)
print(f"\nSummary saved to: {summary_path}")