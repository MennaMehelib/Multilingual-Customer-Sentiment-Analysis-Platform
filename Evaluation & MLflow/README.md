# Evaluation & MLflow

## Overview

This module evaluates and compares classical NLP models and XLM-RoBERTa for multilingual sentiment classification across Arabic, English, and Chinese. It includes unified evaluation, error analysis, model comparison, and experiment tracking using MLflow.

## Key Features

- Unified evaluation using Accuracy, Precision, Recall, and Macro F1.
- Classical model selection based on Validation Macro F1.
- Error analysis by language, sentiment, and prediction confidence.
- Comparative analysis of classical models and XLM-RoBERTa on the same test samples.
- MLflow tracking of experiments, metrics, parameters, and artifacts.

## Models Evaluated

- Logistic Regression (Unigrams)
- Logistic Regression (Unigrams + Bigrams)
- Naive Bayes (Unigrams)
- Naive Bayes (Unigrams + Bigrams)
- XLM-RoBERTa (Transformer)

## Results

| Model                                    | Test Macro F1 |
| ---------------------------------------- | ------------: |
| Logistic Regression – Unigrams           |        62.30% |
| Logistic Regression – Unigrams + Bigrams |        60.84% |
| Naive Bayes – Unigrams                   |        58.82% |
| Naive Bayes – Unigrams + Bigrams         |        58.01% |
| XLM-RoBERTa                              |    **73.64%** |

XLM-RoBERTa outperformed the best classical model by **11.34 percentage points** in Test Macro F1.

## Error Analysis

The best classical model showed its highest error rate on Chinese samples (53.58%), compared with Arabic (31.82%) and English (28.29%). The transformer improved overall performance across languages, while the classical model performed better on the neutral sentiment subset.

## Scripts

| Script                    | Purpose                                                |
| ------------------------- | ------------------------------------------------------ |
| `generate_predictions.py` | Generates classical model predictions.                 |
| `unified_evaluation.py`   | Evaluates models and selects the best classical model. |
| `error_analysis.py`       | Analyzes misclassifications and confidence.            |
| `compare_model_errors.py` | Compares classical and transformer predictions.        |
| `mlflow_tracking.py`      | Logs experiments, metrics, parameters, and artifacts.  |

## MLflow

Experiments are tracked using MLflow with a SQLite backend and local artifact storage. Six runs were successfully recorded, including four classical models, the transformer, and evaluation artifacts.

## Execution Order

```bash
python "Evaluation & MLflow/generate_predictions.py"
python "Evaluation & MLflow/unified_evaluation.py"
python "Evaluation & MLflow/error_analysis.py"
python "Evaluation & MLflow/compare_model_errors.py"
python "Evaluation & MLflow/mlflow_tracking.py"
```

## Notes

- The best classical model is selected using validation performance.
- Classical and transformer predictions are aligned on the same test samples.
- Model weights are not included; predictions, metrics, reports, and experiment artifacts are retained.
