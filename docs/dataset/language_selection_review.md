# Independent Data-Science Review of Final Language Selection

## Executive Summary

- **Evaluated Set**: English (`en`), Chinese (`zh`), Arabic (`ar`)
- **Evaluation Status**: **KEEP {en, zh, ar}**
- **Decision Rationale**: Based on multi-dimensional data quality, linguistic typology, customer sentiment domain realism, 3-class balance, source independence, and downstream model feasibility, `{en, zh, ar}` is decisively superior to all other tested combinations (`{en, zh, ja}`, `{en, zh, ko}`, `{en, es, de, fr}`).

---

## 1. Class Balance & Usable Balanced Capacity

Customer sentiment classification in this project is a 3-class problem: **Positive**, **Neutral**, **Negative**. A severe imbalance (especially an empty or near-empty neutral class) cripples classical baselines and biases transformer attention mechanisms.

| Language | Total Samples | Positive (%) | Neutral (%) | Negative (%) | Minority Class | Balanced 3-Class Capacity | Balance Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| **en** | 1,519,860 | 33.93% | 29.62% | 36.44% | 450,247 (Neu) | 1,350,741 | 0.9318 |
| **zh** | 444,835 | 36.86% | 28.60% | 34.54% | 127,226 (Neu) | 381,678 | 0.9174 |
| **ar** | 208,375 | 30.83% | 41.08% | 28.09% | 58,534 (Neg) | 175,602 | 0.8702 |

### Key Observations:
1. In all three selected languages, the minority class represents over 28% of the total language volume. This means every class has robust natural support without requiring synthetic oversampling.
2. Neutral reviews are notoriously difficult to source in customer sentiment datasets (customers usually write reviews when extremely pleased or angry). Many other languages in the dataset fail drastically here:
   - Vietnamese (`vi`): Neutral is only 5.47% (Balanced capacity shrinks to 20,850 despite 127k total).
   - Russian (`ru`): Neutral is only 14.01% (Balanced capacity 86k out of 205k).
   - Turkish (`tr`): Zero neutral samples in train (Excluded completely).
3. The smallest language in `{en, zh, ar}` is Arabic with 175,602 balanced capacity, providing an abundant pool for stratified sampling.

---

## 2. Source and Domain Diversity vs. Confounding Risk

A common flaw in multilingual NLP benchmarks is **source confounding**: when a language's samples originate almost exclusively from a single website or domain, the model learns the source/platform formatting rather than genuine multilingual sentiment.

| Candidate Language | Total Sources | Largest Source Count | Largest Source % | Domains |
|---|---:|---:|---:|---|
| **en** | 28 | 389,963 | 25.66% | Places, hotels, Amazon, apps, airlines, books, restaurants (19 domains) |
| **zh** | 5 | 124,984 | 28.10% | Amazon reviews, Tweets, Zenodo 22-domains, Books, JD.com (5 domains) |
| **ar** | 4 | 99,651 | 47.82% | Hotels/books/movies/products (Abed Khooli 100k), KAUST, Company reviews, Tweets (3 domains) |
| *ja* | 2 | 209,839 | 62.52% | Heavily dominated by Amazon Multi (62.5%) & Twitter |
| *ko* | 2 | 200,000 | 76.92% | Heavily dominated by Mendeley movie reviews (76.9%) |
| *es* | 3 | 209,744 | 94.09% | 94.1% single source (`amazon_reviews_multi`) |
| *de* | 2 | 209,876 | 98.60% | 98.6% single source (`amazon_reviews_multi`) |
| *fr* | 3 | 209,618 | 79.81% | 79.8% single source (`amazon_reviews_multi`) |
| *pt* | 3 | 45,859 | 93.23% | 93.2% single source (`b2w-reviews01`) |

### Critical Finding:
European languages in this dataset (`es`, `de`, `fr`, `pt`) suffer from extreme source concentration (80% to 98% from Amazon Reviews Multi). Training on them would essentially make the system an "Amazon Review Classifier" rather than a generalized customer sentiment analyzer.
In contrast, `{en, zh, ar}` exhibit the highest source and domain breadth across all candidates, spanning e-commerce, hospitality, transport, apps, social media feedback, and corporate customer reviews.

---

## 3. Linguistic Typology & Script Diversity

A genuine "Multilingual Platform" should demonstrate true multilingual capabilities rather than minor variations of the Latin alphabet.

1. **English (`en`)**:
   - Family: Indo-European (Germanic)
   - Script: Latin alphabet, space-delimited, left-to-right (LTR)
   - Morphology: Analytic / Fusional
2. **Chinese (`zh`)**:
   - Family: Sino-Tibetan
   - Script: Hanzi (Logographic), unsegmented (no word spaces), LTR
   - Morphology: Isolating
3. **Arabic (`ar`)**:
   - Family: Afroasiatic (Semitic)
   - Script: Arabic script (Abjad, cursive, diacritics, right-to-left RTL)
   - Morphology: Root-and-pattern (templatic), agglutinative clitics

### Why this matters for the team:
- **Person 1 (Preprocessing)**: Shows the team can handle real multilingual challenges: character-level normalization for Chinese, RTL and Tashkeel/Alef normalization for Arabic, and casing/contractions for English.
- **Person 2 (Classical NLP)**: Demonstrates that n-gram models can handle word-level tokens (EN), character n-grams (ZH), and morphologically rich n-grams (AR).
- **Person 3 (Transformer)**: Standard tokenizers like `xlm-roberta-base` and `bert-base-multilingual-cased` are pre-trained heavily on EN, ZH, and AR.
- **Person 5 (Platform UI)**: A live interface showcasing Latin (LTR), Chinese Hanzi, and Arabic (RTL) gives the project tremendous visual and practical graduation presentation impact.

---

## 4. Cross-Split Leakage & Quality Audit

The audit detected text duplicates across train, validation, and test splits:

| Language | Train-Validation | Train-Test | Validation-Test | Total Overlap | Overlap Rate |
|---|---:|---:|---:|---:|---:|
| **en** | 3,395 | 3,342 | 500 | 7,237 | 0.47% |
| **zh** | 299 | 295 | 43 | 637 | 0.14% |
| **ar** | 39 | 34 | 4 | 77 | 0.037% |
| **Total Set** | **3,733** | **3,671** | **547** | **7,951** | **0.36%** |

### Evaluation:
- The overall overlap rate in `{en, zh, ar}` is only 0.36% (7,951 records out of 2,173,070 rows).
- Arabic has virtually negligible cross-split overlap (only 77 rows).
- Deterministic cross-split deduplication can easily purge any test/validation records that appear in training text hashes, guaranteeing zero data leakage for model evaluation.

---

## 5. Review of Scoring Methodology & Biases

We audited the scoring formula in `src/data/final_language_selection.py`:
$$\text{Score} = 0.30 \times \text{Balance} + 0.25 \times \text{Capacity}_{\log} + 0.15 \times \text{SourceDiv} + 0.10 \times \text{DomainDiv} + 0.20 \times \text{LeakageSafety}$$

### Potential Biases Checked:
1. **Did log scaling unfairly advantage English?**
   - No. Log-scaling actually dampened English's 1.5M raw size advantage so that smaller, balanced languages could compete fairly.
2. **Did source concentration penalize Japanese or Korean unfairly?**
   - No. Korean has 76.9% movie reviews and Japanese has 62.5% Amazon reviews from only 2 sources each. In real customer feedback systems, single-source dominance creates domain bias.
3. **Did candidate set size 3 outscore 4 or 5?**
   - In `candidate_language_sets.csv`, the 4-language set `{en, zh, ar, ja}` scored 0.8305 (Rank 4), and 5-language `{en, zh, ar, ja, ko}` scored 0.8141 (Rank 10).
   - Adding Japanese or Korean increases total volume, but reduces average source and domain diversity while introducing script overlap (Japanese Kanji shares significant semantic subword space with Chinese Hanzi).
   - Keeping 3 typologically distinct languages (`en`, `zh`, `ar`) maximizes pedagogical clarity, modeling rigor, and computational feasibility for the graduation team.

---

## 6. Final Decision

### **KEEP {en, zh, ar}**

**Confirmation:**
The selection of **English (`en`)**, **Chinese (`zh`)**, and **Arabic (`ar`)** is scientifically justified, mathematically validated, and practically optimal for all team members. Downstream tasks will proceed using this 3-language configuration.
