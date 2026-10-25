# Candidate Language Validation Report

This report validates candidate languages before final dataset selection.

The validation considers class balance, source concentration, domain concentration, and cross-split text overlap.

> Important: this report does not automatically declare a final language set.

## Language Summary

| Language | Total | Positive | Neutral | Negative | Balanced Capacity | Balance Score | Sources | Largest Source % | Domains | Largest Domain % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 1,519,860 | 515,731 | 450,247 | 553,882 | 1,350,741 | 0.9318 | 28 | 25.66% | 19 | 25.66% |
| zh | 444,835 | 163,957 | 127,226 | 153,652 | 381,678 | 0.9174 | 5 | 28.10% | 5 | 28.10% |
| ja | 335,656 | 125,708 | 83,969 | 125,979 | 251,907 | 0.8748 | 2 | 62.52% | 2 | 62.52% |
| ko | 259,998 | 79,998 | 100,000 | 80,000 | 239,994 | 0.9231 | 2 | 76.92% | 2 | 76.92% |
| ar | 208,375 | 64,251 | 85,590 | 58,534 | 175,602 | 0.8702 | 4 | 47.82% | 3 | 47.82% |
| es | 222,911 | 87,988 | 45,835 | 89,088 | 137,505 | 0.8060 | 3 | 94.09% | 3 | 94.09% |
| de | 212,853 | 84,916 | 42,978 | 84,959 | 128,934 | 0.8028 | 2 | 98.60% | 2 | 98.60% |
| fr | 262,645 | 109,712 | 42,962 | 109,971 | 128,886 | 0.7449 | 3 | 79.81% | 3 | 79.81% |
| ru | 205,186 | 94,521 | 28,753 | 81,912 | 86,259 | 0.6795 | 3 | 51.87% | 3 | 51.87% |
| pt | 49,188 | 16,011 | 17,166 | 16,011 | 48,033 | 0.9765 | 3 | 93.23% | 2 | 93.83% |
| vi | 127,068 | 63,497 | 6,950 | 56,621 | 20,850 | 0.5550 | 7 | 27.90% | 6 | 27.90% |
| hi | 9,423 | 2,257 | 4,359 | 2,807 | 6,771 | 0.7769 | 2 | 69.36% | 2 | 69.36% |
| id | 12,536 | 7,244 | 1,355 | 3,937 | 4,065 | 0.5302 | 2 | 89.19% | 2 | 89.19% |
| ms | 6,661 | 3,064 | 1,012 | 2,585 | 3,036 | 0.6919 | 1 | 100.00% | 1 | 100.00% |

## Cross-Split Overlap

| Language | Train-Val | Train-Test | Val-Test | Total Overlap |
|---|---:|---:|---:|---:|
| en | 3,395 | 3,342 | 500 | 7,237 |
| zh | 299 | 295 | 43 | 637 |
| ko | 3 | 2 | 2 | 7 |
| pt | 1 | 0 | 0 | 1 |
| ar | 39 | 34 | 4 | 77 |
| ja | 1 | 3 | 0 | 4 |
| es | 10 | 14 | 0 | 24 |
| de | 4 | 6 | 0 | 10 |
| fr | 19 | 12 | 2 | 33 |
| ru | 115 | 119 | 24 | 258 |
| hi | 4 | 5 | 0 | 9 |
| vi | 49 | 67 | 8 | 124 |
| ms | 1 | 1 | 0 | 2 |
| id | 0 | 0 | 0 | 0 |

## Interpretation Rules

- **Balance score:** higher is better.
- **Balanced capacity:** maximum number of rows that can be used for a perfectly balanced 3-class subset.
- **Source concentration:** lower is generally safer; very high concentration means a language may be strongly associated with one source.
- **Domain concentration:** lower is generally safer when the goal is general customer sentiment.
- **Cross-split overlap:** should ideally be zero.
