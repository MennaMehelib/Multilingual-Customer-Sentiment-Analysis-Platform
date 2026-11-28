from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = Path(__file__).resolve().parent

CLASSICAL_PATH = (
    EVALUATION_DIR / "predictions" / "classical_predictions.csv"
)

TRANSFORMER_PATH = (
    PROJECT_ROOT / "Advanced NLP Model"
    / "results" / "test_predictions.csv"
)

OUTPUT_DIR = EVALUATION_DIR / "model_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load predictions
classical_df = pd.read_csv(CLASSICAL_PATH)
transformer_df = pd.read_csv(TRANSFORMER_PATH)


# Select best Classical Model on test set
classical_df = classical_df[
    (classical_df["split"] == "test")
    & (classical_df["model"] == "Logistic Regression")
    & (classical_df["ngram"] == "Unigrams")
].copy()


# Reset row indexes for alignment
classical_df = classical_df.reset_index(drop=True)
transformer_df = transformer_df.reset_index(drop=True)


# Validate datasets alignment
if len(classical_df) != len(transformer_df):
    raise ValueError("Prediction datasets have different sizes!")

for column in ["cleaned_text", "language", "label"]:
    if not classical_df[column].equals(transformer_df[column]):
        raise ValueError(
            f"Dataset mismatch detected in column: {column}"
        )


print("Both models are aligned on the same test samples!")
print(f"Number of samples: {len(classical_df)}")
print("Ready for model comparison.")

# Compare model predictions
comparison_df = pd.DataFrame({
    "sample_id": classical_df["sample_id"],
    "cleaned_text": classical_df["cleaned_text"],
    "language": classical_df["language"],
    "true_label": classical_df["label"],
    "classical_prediction": classical_df["predicted"],
    "transformer_prediction": transformer_df["predicted"]
})


# Identify model errors
comparison_df["classical_error"] = (
    comparison_df["true_label"]
    != comparison_df["classical_prediction"]
)

comparison_df["transformer_error"] = (
    comparison_df["true_label"]
    != comparison_df["transformer_prediction"]
)


# Identify prediction disagreement
comparison_df["models_disagree"] = (
    comparison_df["classical_prediction"]
    != comparison_df["transformer_prediction"]
)


print("\nModel Error Comparison:")
print(f"Classical Errors: {comparison_df['classical_error'].sum()}")
print(f"Transformer Errors: {comparison_df['transformer_error'].sum()}")
print(f"Model Disagreements: {comparison_df['models_disagree'].sum()}")

# Categorize prediction outcomes
comparison_df["error_category"] = "Both Correct"

comparison_df.loc[
    comparison_df["classical_error"]
    & ~comparison_df["transformer_error"],
    "error_category"
] = "Classical Wrong - Transformer Correct"

comparison_df.loc[
    ~comparison_df["classical_error"]
    & comparison_df["transformer_error"],
    "error_category"
] = "Classical Correct - Transformer Wrong"

comparison_df.loc[
    comparison_df["classical_error"]
    & comparison_df["transformer_error"],
    "error_category"
] = "Both Wrong"


# Count each category
comparison_summary = (
    comparison_df["error_category"]
    .value_counts()
    .reset_index()
)

comparison_summary.columns = [
    "error_category",
    "sample_count"
]


print("\nComparison Summary:")
print(comparison_summary)

# Extract samples where both models made incorrect predictions
both_wrong = comparison_df[
    comparison_df["error_category"] == "Both Wrong"
].copy()


# Save shared errors for manual inspection
both_wrong.to_csv(
    OUTPUT_DIR / "both_models_wrong.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\nSamples Misclassified by Both Models:")
print(f"Total samples: {len(both_wrong)}")

print("\nExamples:")
print(
    both_wrong[
        [
            "sample_id",
            "cleaned_text",
            "language",
            "true_label",
            "classical_prediction",
            "transformer_prediction"
        ]
    ].head(10).to_string(index=False)
)

# Extract samples where Transformer corrected Classical errors
transformer_corrected = comparison_df[
    comparison_df["error_category"]
    == "Classical Wrong - Transformer Correct"
].copy()


# Save corrected predictions
transformer_corrected.to_csv(
    OUTPUT_DIR / "transformer_corrected_classical.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\nErrors Corrected by Transformer:")
print(f"Total samples: {len(transformer_corrected)}")

print("\nExamples:")
print(
    transformer_corrected[
        [
            "sample_id",
            "cleaned_text",
            "language",
            "true_label",
            "classical_prediction",
            "transformer_prediction"
        ]
    ].head(10).to_string(index=False)
)


# Extract samples where Classical corrected Transformer errors
classical_corrected = comparison_df[
    comparison_df["error_category"]
    == "Classical Correct - Transformer Wrong"
].copy()


# Save corrected predictions
classical_corrected.to_csv(
    OUTPUT_DIR / "classical_corrected_transformer.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\nErrors Corrected by Classical Model:")
print(f"Total samples: {len(classical_corrected)}")

print("\nExamples:")
print(
    classical_corrected[
        [
            "sample_id",
            "cleaned_text",
            "language",
            "true_label",
            "classical_prediction",
            "transformer_prediction"
        ]
    ].head(10).to_string(index=False)
)

# Compare model errors by language
language_comparison = (
    comparison_df
    .groupby("language")
    .agg(
        total_samples=("sample_id", "count"),
        classical_errors=("classical_error", "sum"),
        transformer_errors=("transformer_error", "sum")
    )
    .reset_index()
)

language_comparison["classical_error_rate"] = (
    language_comparison["classical_errors"]
    / language_comparison["total_samples"]
)

language_comparison["transformer_error_rate"] = (
    language_comparison["transformer_errors"]
    / language_comparison["total_samples"]
)


# Save language comparison
language_comparison.to_csv(
    OUTPUT_DIR / "model_errors_by_language.csv",
    index=False
)


print("\nModel Error Comparison by Language:")
print(language_comparison)

# Compare model errors by actual sentiment
sentiment_comparison = (
    comparison_df
    .groupby("true_label")
    .agg(
        total_samples=("sample_id", "count"),
        classical_errors=("classical_error", "sum"),
        transformer_errors=("transformer_error", "sum")
    )
    .reset_index()
)

sentiment_comparison["classical_error_rate"] = (
    sentiment_comparison["classical_errors"]
    / sentiment_comparison["total_samples"]
)

sentiment_comparison["transformer_error_rate"] = (
    sentiment_comparison["transformer_errors"]
    / sentiment_comparison["total_samples"]
)


# Save sentiment comparison
sentiment_comparison.to_csv(
    OUTPUT_DIR / "model_errors_by_sentiment.csv",
    index=False
)


print("\nModel Error Comparison by Sentiment:")
print(sentiment_comparison)

# Identify samples where both models agree on the same wrong prediction
both_models_agree_wrong = comparison_df[
    (comparison_df["classical_error"])
    & (comparison_df["transformer_error"])
    & (
        comparison_df["classical_prediction"]
        == comparison_df["transformer_prediction"]
    )
].copy()


# Save potential label-noise candidates
both_models_agree_wrong.to_csv(
    OUTPUT_DIR / "both_models_agree_wrong.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\nPotential Label Noise Candidates:")
print(f"Total samples: {len(both_models_agree_wrong)}")

print("\nExamples:")
print(
    both_models_agree_wrong[
        [
            "sample_id",
            "cleaned_text",
            "language",
            "true_label",
            "classical_prediction",
            "transformer_prediction"
        ]
    ].head(10).to_string(index=False)
)

# Save the complete model comparison
comparison_df.to_csv(
    OUTPUT_DIR / "full_model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\nFull Model Comparison Saved Successfully!")
print(f"File: {OUTPUT_DIR / 'full_model_comparison.csv'}")
print(f"Total compared samples: {len(comparison_df)}")

# Save model comparison summary
comparison_summary["percentage"] = (
    comparison_summary["sample_count"]
    / len(comparison_df)
) * 100


comparison_summary.to_csv(
    OUTPUT_DIR / "model_comparison_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\nModel Comparison Summary:")
print(comparison_summary)

print("\nComparison summary saved successfully!")