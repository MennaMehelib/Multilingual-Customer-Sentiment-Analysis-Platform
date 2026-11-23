# Classical NLP Models

## Overview
This section focuses on implementing classical machine learning approaches for Sentiment Analysis.

The goal is to build baseline models and compare their performance using different text representation techniques.

## Tasks and Implementation

### 1. TF-IDF Vectorization
- Convert text data into numerical features using TF-IDF.
- Tokenization is performed automatically by `TfidfVectorizer`.

### 2. N-gram Experiments
Two configurations are tested:
- Unigrams: `(1, 1)`
- Unigrams + Bigrams: `(1, 2)`

### 3. Classical Machine Learning Models
The following models are implemented:
- Multinomial Naive Bayes
- Logistic Regression

Each model is trained and evaluated using both N-gram configurations.

## Evaluation Metrics
Model performance is evaluated using:
- Accuracy
- Precision (Macro)
- Recall (Macro)
- F1-score (Macro)
- Confusion Matrix
- Classification Report

## Dataset
The implementation uses:
- `train.csv` for model training.
- `validation.csv` for model evaluation and model selection.
- `test.csv` for final performance evaluation.

Expected dataset columns:
- `cleaned_text`
- `label`

## Project Structure

```text
Classical-NLP/
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── results/
├── classical_nlp.ipynb
├── requirements.txt
└── README.md
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run
1. Open `classical_nlp.ipynb`.
2. Make sure the dataset files are inside the `data/` directory.
3. Install the required dependencies.
4. Run all notebook cells in order.

## Results
The notebook generates:
- Evaluation metrics for all four experiments.
- Classification reports.
- Confusion matrix visualizations.
- Model comparison tables and charts.

All generated outputs are saved in the `results/` directory.

## Experiments Summary

| Model | N-gram |
|---|---|
| Naive Bayes | Unigrams |
| Naive Bayes | Unigrams + Bigrams |
| Logistic Regression | Unigrams |
| Logistic Regression | Unigrams + Bigrams |

The best configuration is selected based on the Validation F1-score.