from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

OUTPUT_DIR = Path(__file__).resolve().parent / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Dataset files
TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"


print("Project paths configured successfully!")


# Load datasets
train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)
test_df = pd.read_csv(TEST_PATH)


# Required columns
REQUIRED_COLUMNS = {"cleaned_text", "label", "language"}


def validate_dataset(df, dataset_name):
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing_columns}"
        )

    print(f"{dataset_name}: {df.shape[0]} rows loaded successfully.")


# Validate all datasets
validate_dataset(train_df, "Training dataset")
validate_dataset(validation_df, "Validation dataset")
validate_dataset(test_df, "Test dataset")


# Prepare text and labels
X_train = train_df["cleaned_text"].fillna("").astype(str)
y_train = train_df["label"].astype(str)

X_val = validation_df["cleaned_text"].fillna("").astype(str)
y_val = validation_df["label"].astype(str)

X_test = test_df["cleaned_text"].fillna("").astype(str)
y_test = test_df["label"].astype(str)


# Fixed label order
LABELS = ["negative", "neutral", "positive"]


# N-gram configurations
NGRAM_SETTINGS = {
    "Unigrams": (1, 1),
    "Unigrams + Bigrams": (1, 2)
}


# Model configurations
MODEL_FACTORIES = {
    "Naive Bayes": lambda: MultinomialNB(),

    "Logistic Regression": lambda: LogisticRegression(
        max_iter=1000
    )
}


# TF-IDF settings
TFIDF_SETTINGS = {
    "max_features": 50000,
    "sublinear_tf": True
}


print("Model configurations defined successfully.")


# Store predictions for all experiments
all_predictions = []

for ngram_name, ngram_range in NGRAM_SETTINGS.items():

    print(f"\nProcessing: {ngram_name}")

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        **TFIDF_SETTINGS
    )

    # Fit only on training data
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform validation and test data
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    for model_name, create_model in MODEL_FACTORIES.items():

        print(f"Training: {model_name} + {ngram_name}")

        # Train model
        model = create_model()
        model.fit(X_train_tfidf, y_train)

        # Generate predictions
        val_predictions = model.predict(X_val_tfidf)
        test_predictions = model.predict(X_test_tfidf)

        # Prediction confidence
        val_confidence = model.predict_proba(X_val_tfidf).max(axis=1)
        test_confidence = model.predict_proba(X_test_tfidf).max(axis=1)

        # Store validation predictions
        val_output = validation_df[
            ["cleaned_text", "language", "label"]
        ].copy()

        val_output["predicted"] = val_predictions
        val_output["model"] = model_name
        val_output["ngram"] = ngram_name
        val_output["split"] = "validation"

        val_output["sample_id"] = range(len(val_output))
        val_output["confidence"] = val_confidence

        # Store test predictions
        test_output = test_df[
            ["cleaned_text", "language", "label"]
        ].copy()

        test_output["predicted"] = test_predictions
        test_output["model"] = model_name
        test_output["ngram"] = ngram_name
        test_output["split"] = "test"

        test_output["sample_id"] = range(len(test_output))
        test_output["confidence"] = test_confidence

        all_predictions.extend([val_output, test_output])

        print("Validation and test predictions generated.")


# Combine all predictions
predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)


# Save predictions
PREDICTIONS_PATH = OUTPUT_DIR / "classical_predictions.csv"

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


print("\nAll predictions saved successfully!")
print(f"Total prediction records: {len(predictions_df)}")
print(f"Saved to: {PREDICTIONS_PATH}")

# Validate generated predictions
expected_records = 4 * (
    len(validation_df) + len(test_df)
)

assert len(predictions_df) == expected_records, (
    f"Expected {expected_records} records, "
    f"but got {len(predictions_df)}"
)

assert predictions_df["predicted"].notna().all()
assert predictions_df["confidence"].notna().all()

print("\nPrediction validation passed!")