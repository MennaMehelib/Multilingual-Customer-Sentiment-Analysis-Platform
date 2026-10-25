# Dataset Construction & Leakage Audit Report

**Author / Role:** Person 1 (Data Collection & Preprocessing)  
**Status:** **COMPLETE**  
**Date:** October 2026  
**Artifact References:**  
- Clean Splits: `data/processed/splits/train.csv`, `validation.csv`, `test.csv`
- Audit Ledger: `data/processed/dataset/extraction_audit_ledger.json`
- Extraction Summary: `data/processed/dataset/extraction_summary.csv`
- Funnel Chart: `docs/visualizations/05_leakage_and_audit_funnel.png`

---

## 1. Executive Summary

This report documents the rigorous, deterministic extraction, duplicate elimination, cross-split leakage purging, and class-balanced benchmark split construction for the Multilingual Customer Sentiment Analysis Platform.

The source dataset is `clapAI/MultiLingualSentiment`, which contains over 3.93 million rows across all splits. To prevent majority-language dominance, domain bias, and cross-split test contamination, we constructed a standardized, zero-leakage benchmark dataset of **117,000 samples** representing **English (`en`)**, **Chinese (`zh`)**, and **Arabic (`ar`)** across **Positive**, **Neutral**, and **Negative** sentiment classes.

---

## 2. Extraction Pipeline & Audit Ledger

### A. Raw Shard Processing:
Raw dataset files were extracted from parquet shards directly:
- Train Shards: `train-00000.parquet`, `train-00001.parquet`, `train-00002.parquet` (Total raw rows: 3,147,478)
- Validation Shard: `validation.parquet` (Total raw rows: 393,435)
- Test Shard: `test.parquet` (Total raw rows: 393,436)
- **Total Streamed / Audited:** **3,934,349 rows**

### B. Cleaning & Filtering Funnel:
1. **Target Language & Label Filtering**: Only rows where `language` $\in \{\text{en}, \text{zh}, \text{ar}\}$ and `label` $\in \{\text{positive}, \text{neutral}, \text{negative}\}$ were accepted into the candidate pool.
2. **Text Validity Filter**: Texts with fewer than 3 characters or containing only whitespace/null were discarded. (0 invalid rows detected in candidate subset).
3. **Within-Split Duplicate & Conflict Resolution**:
   - For texts appearing multiple times with conflicting labels (e.g., annotated as both positive and negative), all instances were purged as noisy annotations.
   - For identical texts with matching labels within the same split, only the first occurrence was retained.
4. **Deterministic Cross-Split Leakage Purging**:
   - Every text in the training pool was normalized and indexed by its SHA-256 hash.
   - Any record in the validation candidate pool whose text hash appeared in training was purged (**107 cross-split leakage rows removed**).
   - Any record in the test candidate pool whose text hash appeared in training OR validation was purged (**131 cross-split leakage rows removed**).

---

## 3. Strict Zero-Leakage Verification

Following extraction and sampling, a comprehensive hash intersection audit was performed across all three finalized split files:

$$\text{Overlap}(\text{Train}, \text{Validation}) = |\text{Hashes}_{\text{Train}} \cap \text{Hashes}_{\text{Validation}}| = \mathbf{0}$$
$$\text{Overlap}(\text{Train}, \text{Test}) = |\text{Hashes}_{\text{Train}} \cap \text{Hashes}_{\text{Test}}| = \mathbf{0}$$
$$\text{Overlap}(\text{Validation}, \text{Test}) = |\text{Hashes}_{\text{Validation}} \cap \text{Hashes}_{\text{Test}}| = \mathbf{0}$$

**Result:** Zero data leakage is mathematically guaranteed across all evaluation sets.

---

## 4. Final Balanced Split Distributions

Deterministic, stratified random sampling (Random Seed: `42`) was applied to achieve exact class and language parity:

| Split | Total Rows | Rows per Language (EN / ZH / AR) | Rows per Class (Pos / Neu / Neg) | Stratification Ratio |
|---|---:|---:|---:|---:|
| **Train** | 90,000 | 30,000 / 30,000 / 30,000 | 30,000 / 30,000 / 30,000 (10,000 per cell) | Exact 1:1:1 |
| **Validation** | 13,500 | 4,500 / 4,500 / 4,500 | 4,500 / 4,500 / 4,500 (1,500 per cell) | Exact 1:1:1 |
| **Test** | 13,500 | 4,500 / 4,500 / 4,500 | 4,500 / 4,500 / 4,500 (1,500 per cell) | Exact 1:1:1 |
| **Total Benchmark** | **117,000** | **39,000 / 39,000 / 39,000** | **39,000 / 39,000 / 39,000** | **Exact 1:1:1** |

---

## 5. Domain & Source Representation

The standardized benchmark split maintains rich domain and source diversity across all three languages:

### Top Domains in Benchmark:
1. **Places & Hospitality**: Places reviews, Hotel reviews, Restaurant reviews.
2. **E-Commerce & Retail**: Amazon multi reviews, JD.com reviews, Products reviews, Books reviews.
3. **Technology & Apps**: Google Play store reviews, App reviews, GPT reviews.
4. **Transport & Airlines**: Airline customer reviews.
5. **Corporate & Finance**: Company reviews, Banking reviews.

The dataset is locked, verified, and ready for immediate consumption by Persons 2, 3, 4, and 5.
