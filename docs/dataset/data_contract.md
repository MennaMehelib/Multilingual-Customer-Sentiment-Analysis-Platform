# Multilingual Sentiment Analysis — Data Contract & Integration Specification

**Author / Role:** Person 1 (Data Collection & Preprocessing)  
**Target Consumers:** Person 2 (Classical NLP Baselines), Person 3 (Advanced Multilingual Transformer), Person 4 (Evaluation & MLflow), Person 5 (Platform Integration)  
**Status:** **APPROVED & LOCKED**  
**Version:** 1.0.0  

---

## 1. Overview & Architectural Guarantees

This document establishes the binding data contract between Person 1 and all downstream team members. 

By consuming the standardized datasets provided under `data/processed/splits/`, team members are guaranteed:
1. **Strict Zero Cross-Split Leakage**: No sample in the evaluation splits (`validation.csv`, `test.csv`) shares identical text with any sample in `train.csv` ($\text{Train} \cap \text{Validation} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$).
2. **Deterministic 1:1:1 Class Balance**: In every split, **Positive**, **Neutral**, and **Negative** classes have identical representation. No majority class bias exists.
3. **Deterministic 1:1:1 Language Balance**: In every split, **English (`en`)**, **Chinese (`zh`)**, and **Arabic (`ar`)** have identical representation.
4. **Clean Data**: Zero null, empty, or sub-minimal (<3 characters) text records exist.
5. **No Conflicting Labels**: Text instances with ambiguous/conflicting annotations have been audited and discarded.

---

## 2. File Locations & Split Sizes

All standardized datasets are stored in the canonical splits directory:

```text
data/processed/splits/
├── train.csv         (90,000 rows — exactly 10,000 per class per language)
├── validation.csv    (13,500 rows — exactly 1,500 per class per language)
└── test.csv          (13,500 rows — exactly 1,500 per class per language)
```

**Total Standardized Benchmark Size:** **117,000 rows** (Exactly 1:1:1 class balance and 1:1:1 language balance).

Each split file provides both the original text (`text`) and the standard preprocessed text (`cleaned_text`) side-by-side, eliminating redundant duplicate files while allowing downstream team members to use either representation directly.

---

## 3. Data Schema & Field Definitions

Each record in the main split files (`data/processed/splits/*.csv`) contains the following fields:

| Field Name | Type | Description | Allowed Values | Example |
|---|---|---|---|---|
| `text` | string | Customer feedback text before normalization (raw clean text) | Non-empty string ($\ge 3$ chars) | `"I don't like this <br> http://t.co 😊!!!!"` |
| `cleaned_text` | string | Customer feedback after Standard Preprocessing normalization | Non-empty string ($\ge 3$ chars) | `"i do not like this 😊!!"` |
| `label` | string | 3-Class Sentiment Polarity | `"positive"`, `"neutral"`, `"negative"` | `"negative"` |
| `language` | string | Two-letter ISO 639-1 language code | `"en"`, `"zh"`, `"ar"` | `"en"` |
| `source` | string | Provenance URL or originating dataset | Valid string | `"https://huggingface.co/datasets/Yelp/yelp_review_full"` |
| `domain` | string | Industry domain of the review | Valid string | `"places reviews"`, `"hotel reviews"` |

### Label Encoding Dictionary:
For models requiring integer target labels (such as PyTorch CrossEntropyLoss or Scikit-Learn classifiers):

```python
LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}
```

---

## 4. Consuming the Dataset (Code Snippets)

### A. Python / Pandas Loading:
```python
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed/splits")

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
```

### B. Consuming Preprocessing in Inference / Training Pipelines:
Downstream team members should consume the modular preprocessor from `src.preprocessing`:

```python
from src.preprocessing import preprocess_text

# Example inference cleaning
clean_text = preprocess_text(
    text="The battery life is amazing! 😊 https://deal.com",
    language="en",
    strategy="standard"  # Recommended default strategy
)
# Output: "the battery life is amazing! 😊"
```

---

## 5. Team Responsibilities & Boundaries

- **Person 2 (Classical NLP Baselines)**:
  - Input: `data/processed/splits/train.csv`, `test.csv`
  - Responsibilities: TF-IDF vectorization (word/n-gram/char), Naive Bayes, Logistic Regression, Confusion Matrix.
  - Constraint: Consume `train.csv` and `test.csv` as provided; do NOT re-split or drop languages.

- **Person 3 (Advanced Multilingual Transformer)**:
  - Input: `data/processed/splits/train.csv`, `validation.csv`, `test.csv`
  - Responsibilities: Tokenization (`XLM-RoBERTa` or `mBERT`), fine-tuning, checkpoint saving in `models/`.
  - Constraint: Use the exact label encoding `LABEL2ID` specified above.

- **Person 4 (Evaluation & MLflow)**:
  - Input: Metrics and predictions from Persons 2 & 3.
  - Responsibilities: Log runs, parameters, metrics, confusion matrices, and error analysis tables.

- **Person 5 (Platform & Deployment)**:
  - Input: Fine-tuned model from Person 3, preprocessing contract from Person 1.
  - Responsibilities: End-to-end sentiment prediction API & UI (`text -> lang -> clean -> predict -> sentiment + confidence`).
