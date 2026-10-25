# Multilingual Customer Sentiment Analysis Platform

A production-grade, graduation-level Machine Learning and Natural Language Processing platform for automated sentiment classification across multilingual customer feedback.

---

## 1. Project Goal & Final Deliverable Architecture

The system enables a user to submit customer reviews and feedback in multiple languages and receives:
1. **Detected / Target Language**
2. **Sentiment Classification**: `positive`, `neutral`, or `negative`
3. **Model Confidence Score**
4. **An Interactive, Intuitive User Interface**

---

## 2. Team Structure & Division of Responsibilities

This graduation project is executed by a team of five members with strict ownership boundaries:

| Role | Primary Ownership | Core Deliverables | Hand-off Interface |
|---|---|---|---|
| **Person 1 (Current Stage)** | **Data Collection & Preprocessing** | Dataset selection, audit, cleaning, zero-leakage splits, preprocessing pipeline, visualizations, documentation | `data/processed/splits/`, `src/preprocessing/`, Data Contract |
| **Person 2** | **Classical NLP Models** | TF-IDF (word/n-gram/char), Naive Bayes, Logistic Regression baselines, baseline evaluation | Baseline model evaluation results & confusion matrices |
| **Person 3** | **Advanced NLP Model** | Multilingual Transformer (`XLM-RoBERTa` / `mBERT`), tokenizer, fine-tuning, inference checkpoint | Saved model weights in `models/` |
| **Person 4** | **Evaluation & MLflow** | MLflow experiment tracking, parameter logging, error analysis, cross-model comparison tables | MLflow tracking logs, error analysis report |
| **Person 5** | **Platform Integration & Deployment** | User interface (Web/UI), model serving, language detection, confidence scores, end-to-end demo | Working platform & user guide |

---

## 3. Dataset & Language Selection

### Selected Language Set: **{English (`en`), Chinese (`zh`), Arabic (`ar`)}**
Selected from `clapAI/MultiLingualSentiment` based on multi-criteria empirical evidence:
- **Linguistic Diversity**: 3 distinct language families and writing systems:
  - Latin Alphabet / Indo-European (`en`, Left-to-Right)
  - Hanzi Characters / Sino-Tibetan (`zh`, Logographic, unsegmented)
  - Arabic Script / Afroasiatic Semitic (`ar`, Right-to-Left Abjad)
- **Natural 3-Class Balance**: Every language maintains natural neutral class representation (~28%–41%), enabling exact 1:1:1 class parity.
- **Source & Domain Diversity**: Samples span 19+ domains across retail, e-commerce, hospitality, apps, and corporate feedback from multiple independent sources, eliminating platform-specific confounding.
- **Leakage Safety**: Overlap across splits was audited and 100% eliminated deterministically.

---

## 4. Standardized Benchmark Dataset (Data Contract)

Cleaned, zero-leakage, balanced datasets are locked in `data/processed/splits/`:

| Split File | Records | Languages (`en` / `zh` / `ar`) | Classes (`pos` / `neu` / `neg`) | Overlap with Other Splits |
|---|---:|---:|---:|---:|
| [`train.csv`](data/processed/splits/train.csv) | **90,000** | 30,000 / 30,000 / 30,000 | 30,000 / 30,000 / 30,000 (10k per cell) | **0** |
| [`validation.csv`](data/processed/splits/validation.csv) | **13,500** | 4,500 / 4,500 / 4,500 | 4,500 / 4,500 / 4,500 (1.5k per cell) | **0** |
| [`test.csv`](data/processed/splits/test.csv) | **13,500** | 4,500 / 4,500 / 4,500 | 4,500 / 4,500 / 4,500 (1.5k per cell) | **0** |
| **Total Benchmark** | **117,000** | **39,000 / 39,000 / 39,000** | **39,000 / 39,000 / 39,000** | **0.00% Leakage** |

Each file contains both `text` (clean original text) and `cleaned_text` (Standard Preprocessing applied), allowing downstream models to ingest either column directly.

### Data Schema (Main Splits):
```text
text          : string  (Original customer review text before normalization)
cleaned_text  : string  (Customer review text after Standard Preprocessing)
label         : string  ("positive", "neutral", "negative")
language      : string  ("en", "zh", "ar")
source        : string  (Originating dataset source URL)
domain        : string  (Feedback domain, e.g. "hotel reviews")
```

---

## 5. Preprocessing Pipeline (`src/preprocessing/`)

A modular, multilingual-aware preprocessing package supporting three configurable strategies:
- **`minimal`**: Decodes HTML, standardizes whitespace, preserves all punctuation and emojis.
- **`standard` (RECOMMENDED)**: Strips URLs/mentions, unpacks hashtags, reduces character repetitions (`sooooo` -> `soo`), expands English contractions (`don't` -> `do not`), normalizes Chinese punctuation and NFKC, normalizes Arabic Tatweel/Tashkeel/Alef, **strictly preserves sentiment emojis and emotion punctuation (`!`, `?`)**.
- **`aggressive`**: Strips all punctuation and emojis, normalizes characters into bare word tokens.

### Python Usage:
```python
from src.preprocessing import preprocess_text

# Example English
clean_en = preprocess_text("I don't like this service! http://t.co 😡", language="en", strategy="standard")
# Output: "i do not like this service! 😡"

# Example Arabic
clean_ar = preprocess_text("المنتـــــج رائع جدًا! شكراً @shop", language="ar", strategy="standard")
# Output: "المنتج رائع جدا! شكرا"

# Example Chinese
clean_zh = preprocess_text("质量非常好！　没有坏。", language="zh", strategy="standard")
# Output: "质量非常好! 没有坏."
```

---

## 6. Project Directory Layout

```text
Multilingual-Customer-Sentiment-Analysis-Platform/
│
├── data/
│   ├── raw/                           # Downloaded parquet shards (gitignored)
│   └── processed/
│       ├── dataset/                   # Extraction ledger & summary metrics
│       ├── language_selection/        # Language audit & multi-criteria rankings
│       ├── preprocessing/             # Preprocessing experiment CSVs & JSONs
│       ├── splits/                    # Clean train.csv, validation.csv, test.csv
│       └── visualizations/            # High-resolution generated charts
│
├── docs/
│   ├── dataset/
│   │   ├── data_contract.md           # Binding schema contract for Persons 2 & 3
│   │   ├── data_quality_report.md     # Quality and integrity audit
│   │   ├── dataset_construction_report.md # Step-by-step extraction audit
│   │   └── language_selection_review.md   # Scientific review of {en, zh, ar}
│   ├── preprocessing/
│   │   ├── preprocessing_experiments.md   # Empirical strategy comparison
│   │   └── preprocessing_report.md        # Preprocessing architectural specs
│   └── visualizations/                # Publication-quality charts (PNG)
│
├── src/
│   ├── data/                          # Data extraction and visualization scripts
│   │   ├── build_clean_dataset.py
│   │   ├── extract_splits_from_parquet.py
│   │   ├── generate_visualizations.py
│   │   ├── dataset_audit.py
│   │   └── final_language_selection.py
│   ├── preprocessing/                 # Modular multilingual preprocessing package
│   │   ├── __init__.py
│   │   ├── text_cleaner.py
│   │   ├── arabic_normalizer.py
│   │   ├── chinese_normalizer.py
│   │   ├── english_normalizer.py
│   │   ├── pipeline.py
│   │   └── run_experiments.py
│   ├── classical/                     # Hand-off location for Person 2
│   ├── transformer/                   # Hand-off location for Person 3
│   └── evaluation/                    # Hand-off location for Person 4
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 7. How to Run & Reproduce Person 1 Deliverables

1. **Activate Environment & Install Dependencies**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Extract & Build Clean Splits (Zero Leakage)**:
   ```powershell
   $env:PYTHONPATH="."; python src/data/extract_splits_from_parquet.py
   ```

3. **Run Preprocessing Experiments**:
   ```powershell
   $env:PYTHONPATH="."; python src/preprocessing/run_experiments.py
   ```

4. **Generate Visualization Charts**:
   ```powershell
   $env:PYTHONPATH="."; python src/data/generate_visualizations.py
   ```

---

## 8. Status & Hand-Off Notice

**Person 1 (Data Collection & Preprocessing) is 100% COMPLETE.**  
Persons 2 and 3 may immediately consume `data/processed/splits/train.csv`, `validation.csv`, and `test.csv` using the specifications in [`docs/dataset/data_contract.md`](docs/dataset/data_contract.md).
