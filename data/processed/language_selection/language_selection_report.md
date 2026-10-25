# Multilingual Sentiment Language Selection Report

## Dataset

- Dataset: `clapAI/MultiLingualSentiment`
- Labels: positive, neutral, negative
- Selection is based on the actual dataset audit.
- No fixed number of languages was imposed.
- Total row count alone is NOT used as the selection criterion.

## Selection Criteria

A language is considered eligible when:

1. All three sentiment classes are present in training.
2. Each training class has at least 500 samples.
3. Each validation class has at least 100 samples.
4. Each test class has at least 100 samples.
5. The language has usable balanced training capacity.

The ranking favors class balance and usable balanced capacity, while also considering source and domain diversity.

## Eligible Languages

| Rank | Language | Train | Pos | Neu | Neg | Balanced Capacity | Balance Score | Sources | Domains | Score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | en | 1,215,709 | 412,535 | 360,246 | 442,928 | 1,080,738 | 0.8133 | 28 | 19 | 122.8583 |
| 2 | zh | 356,172 | 131,166 | 101,941 | 123,065 | 305,823 | 0.7772 | 5 | 5 | 109.1804 |
| 3 | ko | 207,998 | 63,998 | 80,000 | 64,000 | 191,994 | 0.8000 | 2 | 2 | 105.2193 |
| 4 | pt | 39,313 | 12,792 | 13,715 | 12,806 | 38,376 | 0.9327 | 3 | 2 | 104.3810 |
| 5 | ar | 166,677 | 51,382 | 68,485 | 46,810 | 140,430 | 0.6835 | 4 | 3 | 99.4292 |
| 6 | ja | 268,241 | 100,312 | 66,942 | 100,987 | 200,826 | 0.6629 | 2 | 2 | 98.5893 |
| 7 | es | 178,434 | 70,676 | 36,619 | 71,139 | 109,857 | 0.5148 | 3 | 3 | 89.3175 |
| 8 | de | 170,216 | 67,879 | 34,424 | 67,913 | 103,272 | 0.5069 | 2 | 2 | 87.4643 |
| 9 | fr | 210,298 | 87,892 | 34,414 | 87,992 | 103,242 | 0.3911 | 3 | 3 | 82.8246 |
| 10 | ru | 164,148 | 75,617 | 23,002 | 65,529 | 69,006 | 0.3042 | 3 | 3 | 76.4645 |
| 11 | hi | 7,514 | 1,819 | 3,473 | 2,222 | 5,457 | 0.5238 | 2 | 2 | 73.6064 |
| 12 | vi | 101,655 | 50,798 | 5,560 | 45,297 | 16,680 | 0.1095 | 7 | 6 | 62.1335 |
| 13 | ms | 5,331 | 2,445 | 794 | 2,092 | 2,382 | 0.3247 | 1 | 1 | 57.8904 |
| 14 | id | 10,007 | 5,752 | 1,084 | 3,171 | 3,252 | 0.1885 | 2 | 2 | 54.2539 |

## Excluded Languages

| Language | Train | Pos | Neu | Neg | Balanced Capacity | Reason |
|---|---:|---:|---:|---:|---:|---|
| it | 2,458 | 823 | 827 | 808 | 2,424 | validation split insufficient; test split insufficient |
| multilingual | 7,513 | 5,152 | 1,683 | 678 | 2,034 | validation split insufficient; test split insufficient |
| tr | 35,794 | 24,000 | 0 | 11,794 | 0 | missing one or more sentiment classes; training minority class below minimum; validation split insufficient; test split insufficient |

## Important Scientific Note

This report does not automatically declare the final language set. It ranks languages using measurable dataset properties. The final set should also be checked for source/domain confounding, duplicate leakage and label quality before model training.
