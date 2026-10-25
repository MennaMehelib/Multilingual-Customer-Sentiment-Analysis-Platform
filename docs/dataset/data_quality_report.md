# Dataset Quality & Integrity Report

**Author / Role:** Person 1 (Data Collection & Preprocessing)  
**Status:** **COMPLETE**  
**Associated Artifacts:**  
- Data Profiles: `data/processed/language_selection/final_selection/selected_language_profiles.csv`
- Zero-Leakage Splits: `data/processed/splits/`
- Audit Funnel Visualization: `docs/visualizations/05_leakage_and_audit_funnel.png`
- Domain Diversity Visualization: `docs/visualizations/04_domain_diversity.png`

---

## 1. Quality Objectives

In accordance with Person 1's core mandate, dataset selection and preparation was governed by data quality, class balance, source independence, and leakage prevention rather than raw volume:

$$\text{Validity & Utility} \gg \text{Raw Sample Count}$$

---

## 2. Multi-Criteria Quality Evaluation

| Quality Dimension | Assessment Method | Benchmark Status | Verification Evidence |
|---|---|---|---|
| **Class Balance** | Ratio of minority class to majority class | **Optimal (1.000)** | Exactly 10,000 samples per class in Train, 1,500 in Val and Test. |
| **Language Balance** | Volume parity across EN, ZH, AR | **Optimal (1.000)** | Exactly 33.3% per language across all splits. |
| **Cross-Split Leakage** | SHA-256 normalized text hash collision | **Zero Leakage (0.00%)** | 0 cross-split duplicates between Train, Val, and Test. |
| **Text Validity** | Minimum length $\ge 3$, non-empty | **100% Valid** | 0 empty or whitespace-only records exist. |
| **Label Consistency** | Conflicting duplicate text detection | **Cleaned** | Conflicting label collisions removed prior to split sampling. |
| **Script Diversity** | Cross-family linguistic coverage | **3 Families** | Latin (LTR), Chinese Hanzi (Logographic), Arabic (RTL Abjad). |
| **Domain Diversity** | Representation across customer industries | **Broad (19+ domains)** | Hospitality, e-commerce, banking, airlines, mobile apps. |

---

## 3. Class Balance Integrity

In raw multilingual sentiment corpora, the neutral class is often heavily depressed. For example, in raw Vietnamese (`vi`), neutral represents only 5.5% of samples, and in raw Turkish (`tr`), neutral was completely missing.

In our selected languages:
- **English**: Natural neutral share was 29.62%.
- **Chinese**: Natural neutral share was 28.60%.
- **Arabic**: Natural neutral share was 41.08%.

Because all three languages have natural neutral representation near ~30-40%, stratified downsampling achieved perfect 1:1:1 class balance without requiring artificial synthetic oversampling or label modification.

---

## 4. Source & Domain Independence

A severe defect in single-corpus multilingual datasets (such as `Amazon Reviews Multi`) is that models learn Amazon product review terminology rather than general customer sentiment.
- English reviews are drawn from **28 independent sources**, with the largest source capped at 25.66%.
- Chinese reviews are drawn from **5 independent sources**, with the largest source capped at 28.10%.
- Arabic reviews are drawn from **4 independent sources**, spanning corporate feedback, Kaust competition data, Abed Khooli 100k reviews, and tweets.

This domain spread ensures that downstream models will generalize effectively across diverse customer touchpoints.
