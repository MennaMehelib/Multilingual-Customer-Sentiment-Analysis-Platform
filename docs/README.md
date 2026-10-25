# Project Documentation Index

Welcome to the documentation repository for the **Multilingual Customer Sentiment Analysis Platform**.

---

## 1. Dataset & Data Engineering (Person 1)
- [`dataset/data_contract.md`](dataset/data_contract.md): **Binding schema contract and integration rules for Persons 2 through 5.**
- [`dataset/language_selection_review.md`](dataset/language_selection_review.md): Independent data-science review and justification for selecting English (`en`), Chinese (`zh`), and Arabic (`ar`).
- [`dataset/dataset_construction_report.md`](dataset/dataset_construction_report.md): Step-by-step extraction audit, conflicting duplicate resolution, and zero-leakage mathematical verification.
- [`dataset/data_quality_report.md`](dataset/data_quality_report.md): Multi-criteria data quality analysis across class balance, source diversity, domain diversity, and text integrity.

---

## 2. Multilingual Preprocessing (Person 1)
- [`preprocessing/preprocessing_report.md`](preprocessing/preprocessing_report.md): Architectural design and specifications of the multilingual preprocessing package (`src/preprocessing/`).
- [`preprocessing/preprocessing_experiments.md`](preprocessing/preprocessing_experiments.md): Quantitative comparison and qualitative examples across Minimal, Standard, and Aggressive normalization strategies.

---

## 3. Visualizations
High-resolution generated charts are stored in [`visualizations/`](visualizations/):
- `01_language_distribution.png`: Language counts across Train, Val, and Test splits.
- `02_sentiment_distribution.png`: 1:1:1 class parity and counts per language.
- `03_text_length_distribution.png`: Character length density curves for Arabic, Chinese, and English.
- `04_domain_diversity.png`: Top feedback domains represented in the dataset.
- `05_leakage_and_audit_funnel.png`: Audit funnel showing raw streamed rows to zero-leakage benchmark splits.
- `06_preprocessing_strategy_comparison.png`: Comparison of affected rows and length reductions across normalization strategies.
- `07_before_after_text_distributions.png`: Direct Before vs. After character length KDE curves for English, Chinese, and Arabic.
