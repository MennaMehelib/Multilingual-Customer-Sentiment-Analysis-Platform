# Multilingual Preprocessing Pipeline Architecture & Specification

**Author / Role:** Person 1 (Data Collection & Preprocessing)  
**Status:** **COMPLETE**  
**Package Path:** `src/preprocessing/`  
**Associated Artifacts:**  
- Comparison Table: `data/processed/preprocessing/preprocessing_comparison.csv`
- Experiment Results JSON: `data/processed/preprocessing/preprocessing_experiment_results.json`
- Experiment Documentation: `docs/preprocessing/preprocessing_experiments.md`
- Visualization: `docs/visualizations/06_preprocessing_strategy_comparison.png`

---

## 1. Design Principles

Customer sentiment analysis in a multilingual environment requires thoughtful preprocessing. Destructive cleaning techniques common in English NLP (e.g., stripping all punctuation, removing emojis, or aggressive stemming) often eliminate crucial sentiment markers:

1. **Emojis Carry Polarity**: Emojis (😊, ❤️, 😡, 👎, 💔) are among the strongest sentiment signals in customer reviews. They must be preserved.
2. **Punctuation Conveys Emotional Intensity**: Multiple exclamation marks (`!!!`) or question marks (`???`) signal strong enthusiasm or acute dissatisfaction. They should be normalized to avoid vocabulary explosion, but NOT stripped entirely.
3. **Contractions Contain Negation**: In English, contractions like `don't`, `can't`, and `won't` contain the negation particle `not`, which flips sentiment polarity. They must be expanded.
4. **Script-Specific Normalization is Required**:
   - **Arabic**: Requires Tatweel (`ـ`) removal, Tashkeel (diacritics) removal, and Alef normalization (`أ/إ/آ` -> `ا`).
   - **Chinese**: Requires Unicode NFKC normalization, full-width punctuation conversion, and redundant inter-character space removal.
   - **English**: Requires lowercasing, HTML entity unescaping, and contraction expansion.

---

## 2. Package Architecture

The preprocessing package is implemented in `src/preprocessing/`:

```text
src/preprocessing/
├── __init__.py               # Exports PreprocessConfig, preprocess_text, normalizers
├── text_cleaner.py           # Base decoding, HTML tag stripping, regex cleaners, emojis
├── arabic_normalizer.py      # Diacritics, Tatweel, Alef variants, Teh Marbuta
├── chinese_normalizer.py     # NFKC, full-width punctuation, inter-character spaces
├── english_normalizer.py     # Contraction expansion, casing
├── pipeline.py               # Configurable multi-strategy coordinator
└── run_experiments.py        # Empirical comparison benchmark suite
```

---

## 3. Supported Strategies

The pipeline exposes three predefined, configurable strategies:

### A. Strategy 1: Minimal Normalization
- Decodes HTML entities (`&amp;` -> `&`, `&#39;` -> `'`).
- Strips non-printable control characters.
- Standardizes tabs, newlines, and multiple spaces into a single space.
- **Preserves**: All casing, all punctuation, emojis, URLs, mentions, and raw script forms.

### B. Strategy 2: Standard Normalization (RECOMMENDED & DEFAULT)
- Decodes HTML entities and strips raw HTML tags.
- Removes non-sentiment noise: URLs (`http://...`) and user mentions (`@user`).
- Unpacks hashtags (`#greatservice` -> `greatservice`).
- Reduces repeated characters to maximum of 2 repeats (`sooooo` -> `soo`, `!!!!!` -> `!!`).
- **English**: Expands contractions (`don't` -> `do not`), lowercases text.
- **Chinese**: Normalizes full-width punctuation to standard punctuation, removes spaces between Hanzi.
- **Arabic**: Removes Tatweel (`ـ`), strips Tashkeel, normalizes Alef forms (`أ/إ/آ` -> `ا`).
- **Preserves**: All sentiment emojis, emotion punctuation (`!`, `?`), negation particles.

### C. Strategy 3: Aggressive Normalization
- Strips URLs, mentions, hashtags, and all Unicode emojis.
- Strips all punctuation marks and symbols.
- Normalizes all characters into bare word tokens.
- *Empirically proven to cause information loss and duplicate collapsing.*

---

## 4. Empirical Comparison Summary

Benchmarking on 6,000 representative customer reviews across the three languages revealed:

| Strategy | Overall Affected Reviews (%) | Empty Reviews Created | Average Length Reduction (%) | Semantic Duplicate Collapsing |
|---|---:|---:|---:|---:|
| **Minimal** | 14.3% | 0 | 0.0% | 0 |
| **Standard (Recommended)** | **81.0%** | **0** | **0.8%** | **0** |
| **Aggressive** | 93.4% | 2 | 3.4% | 2 |

**Conclusion:** **Standard Normalization** effectively purges out-of-vocabulary noise and unifies spelling while perfectly maintaining sentiment polarity and zero information loss.
