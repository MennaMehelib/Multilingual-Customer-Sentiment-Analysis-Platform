# Final Language Selection Analysis

## Dataset

- Dataset: `clapAI/MultiLingualSentiment`
- Labels: positive, neutral, negative
- No fixed number of languages was imposed.
- Raw language size alone was not used as the selection criterion.

## Selection Principle

The analysis prioritizes dataset validity and balance over raw sample count. Source/domain concentration and cross-split overlap are included because a language can appear large and balanced while still being strongly associated with one source or domain.

### Analysis weights

| Criterion | Weight |
|---|---:|
| Class balance | 0.30 |
| Usable balanced capacity | 0.25 |
| Source diversity | 0.15 |
| Domain diversity | 0.10 |
| Leakage safety | 0.20 |

## Best Candidate Set

**Languages:** en, zh, ar

**Number of languages:** 3

**Set score:** 0.8507

**Mean balance score:** 0.9065

**Minimum balanced capacity:** 175,602

**Median balanced capacity:** 381,678

**Source diversity score:** 0.6614

**Domain diversity score:** 0.6614

**Leakage score:** 0.9978

**Cross-split overlap:** 7,951

## Selected Language Profiles

| Language | Train Total | Balanced Capacity | Balance Score | Sources | Domains | Cross-Split Overlap |
|---|---:|---:|---:|---:|---:|---:|
| en | 1,519,860 | 1,350,741 | 0.9318 | 28 | 19 | 7,237 |
| zh | 444,835 | 381,678 | 0.9174 | 5 | 5 | 637 |
| ar | 208,375 | 175,602 | 0.8702 | 4 | 3 | 77 |

## Important Interpretation

The selected set is the highest-scoring candidate under the stated analytical criteria. It is not a claim that this is the only scientifically valid language set. Before finalizing the dataset, the selected languages should still undergo text-quality inspection, duplicate inspection, label-quality auditing, and preprocessing experiments.

No model training was performed by this script.
