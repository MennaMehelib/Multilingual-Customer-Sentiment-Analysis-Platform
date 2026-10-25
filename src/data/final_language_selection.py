from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "language_selection"
)

OUTPUT_DIR = INPUT_DIR / "final_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILES_FILE = INPUT_DIR / "candidate_validation_profiles.csv"
OVERLAP_FILE = INPUT_DIR / "cross_split_overlap.csv"

# Languages that are not valid 3-class candidates
# according to the actual audit.
EXCLUDED_LANGUAGES = {
    "it",
    "multilingual",
    "tr",
}

# Do not let extremely tiny languages dominate.
# This is a practical screening threshold, not a
# scientific claim that this is the "correct" size.
MIN_BALANCED_CAPACITY = 3000

# We compare multiple candidate-set sizes.
SET_SIZES = [3, 4, 5, 6, 7, 8]


# ============================================================
# ANALYSIS PRIORITIES
# ============================================================
#
# These are NOT scientific truths.
# They simply encode the project's stated priority:
#
# 1. validity / class balance
# 2. usable amount of balanced data
# 3. source diversity
# 4. domain diversity
# 5. leakage / cross-split overlap
#
# We intentionally do NOT reward raw total size directly.
#

WEIGHTS = {
    "balance": 0.30,
    "capacity": 0.25,
    "source_diversity": 0.15,
    "domain_diversity": 0.10,
    "leakage": 0.20,
}


# ============================================================
# HELPERS
# ============================================================

def minmax(series: pd.Series) -> pd.Series:
    """
    Normalize values to [0, 1].
    """

    series = pd.to_numeric(series, errors="coerce").fillna(0.0)

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            1.0,
            index=series.index,
        )

    return (series - minimum) / (maximum - minimum)


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str:
    """
    Return the first available column from candidates.
    """

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"\nCould not find any of these columns: {candidates}"
        f"\nAvailable columns: {list(df.columns)}"
    )


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_overlap_rate(
    row: pd.Series,
    total: float,
) -> float:
    """
    Total cross-split overlap / total rows.
    """

    if total <= 0:
        return 0.0

    return (
        safe_float(row.get("total_cross_split_overlap", 0))
        / total
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("FINAL LANGUAGE SELECTION ANALYSIS")
print("=" * 90)

print("\nLoading validation profiles...")

if not PROFILES_FILE.exists():
    raise FileNotFoundError(
        f"Missing file:\n{PROFILES_FILE}\n\n"
        "Run validate_language_candidates.py first."
    )

profiles = pd.read_csv(PROFILES_FILE)

print(
    f"Loaded {len(profiles)} language records."
)

print("\nLoading cross-split overlap...")

if not OVERLAP_FILE.exists():
    raise FileNotFoundError(
        f"Missing file:\n{OVERLAP_FILE}\n\n"
        "Run validate_language_candidates.py first."
    )

overlap = pd.read_csv(OVERLAP_FILE)

print(
    f"Loaded {len(overlap)} overlap records."
)


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

profiles.columns = [
    str(column).strip()
    for column in profiles.columns
]

overlap.columns = [
    str(column).strip()
    for column in overlap.columns
]


# ============================================================
# FIND COLUMNS
# ============================================================

language_col = find_column(
    profiles,
    [
        "language",
        "lang",
    ],
)

capacity_col = find_column(
    profiles,
    [
        "balanced_capacity",
        "max_balanced_size",
    ],
)

balance_col = find_column(
    profiles,
    [
        "balance_score",
    ],
)

total_col = find_column(
    profiles,
    [
        "total",
        "total_count",
    ],
)

source_count_col = find_column(
    profiles,
    [
        "source_count",
        "sources",
        "num_sources",
    ],
)

domain_count_col = find_column(
    profiles,
    [
        "domain_count",
        "domains",
        "num_domains",
    ],
)

# IMPORTANT:
# The validation script currently stores concentration,
# not "largest_source_pct".
source_concentration_col = find_column(
    profiles,
    [
        "source_concentration",
        "largest_source_pct",
        "largest_source_percent",
    ],
)

domain_concentration_col = find_column(
    profiles,
    [
        "domain_concentration",
        "largest_domain_pct",
        "largest_domain_percent",
    ],
)

overlap_language_col = find_column(
    overlap,
    [
        "language",
        "lang",
    ],
)

train_val_col = find_column(
    overlap,
    [
        "train_validation_overlap",
    ],
)

train_test_col = find_column(
    overlap,
    [
        "train_test_overlap",
    ],
)

val_test_col = find_column(
    overlap,
    [
        "validation_test_overlap",
    ],
)

total_overlap_col = find_column(
    overlap,
    [
        "total_cross_split_overlap",
        "total_overlap",
    ],
)


# ============================================================
# CLEAN PROFILE DATA
# ============================================================

profiles[language_col] = (
    profiles[language_col]
    .astype(str)
    .str.strip()
    .str.lower()
)

numeric_columns = [
    capacity_col,
    balance_col,
    total_col,
    source_count_col,
    domain_count_col,
    source_concentration_col,
    domain_concentration_col,
]

for column in numeric_columns:
    profiles[column] = pd.to_numeric(
        profiles[column],
        errors="coerce",
    ).fillna(0.0)


# ============================================================
# CLEAN OVERLAP DATA
# ============================================================

overlap[overlap_language_col] = (
    overlap[overlap_language_col]
    .astype(str)
    .str.strip()
    .str.lower()
)

for column in [
    train_val_col,
    train_test_col,
    val_test_col,
    total_overlap_col,
]:
    overlap[column] = pd.to_numeric(
        overlap[column],
        errors="coerce",
    ).fillna(0)


# ============================================================
# MERGE OVERLAP INFORMATION
# ============================================================

overlap_small = overlap[
    [
        overlap_language_col,
        train_val_col,
        train_test_col,
        val_test_col,
        total_overlap_col,
    ]
].copy()

overlap_small = overlap_small.rename(
    columns={
        overlap_language_col: language_col,
        train_val_col: "train_validation_overlap",
        train_test_col: "train_test_overlap",
        val_test_col: "validation_test_overlap",
        total_overlap_col: "total_cross_split_overlap",
    }
)

profiles = profiles.merge(
    overlap_small,
    on=language_col,
    how="left",
)

for column in [
    "train_validation_overlap",
    "train_test_overlap",
    "validation_test_overlap",
    "total_cross_split_overlap",
]:
    profiles[column] = (
        pd.to_numeric(
            profiles[column],
            errors="coerce",
        )
        .fillna(0)
    )


# ============================================================
# INITIAL ELIGIBILITY
# ============================================================

profiles["eligible"] = (
    ~profiles[language_col].isin(EXCLUDED_LANGUAGES)
    & (
        profiles[capacity_col]
        >= MIN_BALANCED_CAPACITY
    )
)


# ============================================================
# NORMALIZED QUALITY COMPONENTS
# ============================================================

# Balance:
# higher is better.
profiles["balance_component"] = (
    profiles[balance_col]
    .clip(lower=0.0, upper=1.0)
)


# Balanced capacity:
# We use log scaling so English's huge size does not
# automatically dominate every other language.
capacity_log = (
    profiles[capacity_col]
    .clip(lower=1)
    .apply(lambda x: __import__("math").log1p(x))
)

profiles["capacity_component"] = minmax(
    capacity_log
)


# Source diversity:
# More sources generally gives less source concentration.
profiles["source_diversity_component"] = (
    1.0
    - profiles[source_concentration_col]
    .clip(
        lower=0.0,
        upper=1.0,
    )
)


# Domain diversity:
# More distributed domains generally gives less domain concentration.
profiles["domain_diversity_component"] = (
    1.0
    - profiles[domain_concentration_col]
    .clip(
        lower=0.0,
        upper=1.0,
    )
)


# Leakage:
# lower overlap is better.
profiles["overlap_rate"] = profiles.apply(
    lambda row: calculate_overlap_rate(
        row,
        safe_float(row[total_col]),
    ),
    axis=1,
)

# Convert overlap to a quality score.
#
# If overlap is 0 -> 1.0
# If overlap increases -> score decreases.
#
# We avoid making a tiny amount of overlap automatically
# reject a language; it is reported separately.
profiles["leakage_component"] = (
    1.0
    - profiles["overlap_rate"].clip(
        lower=0.0,
        upper=1.0,
    )
)


# ============================================================
# INDIVIDUAL LANGUAGE QUALITY SCORE
# ============================================================

profiles["quality_score"] = (
    WEIGHTS["balance"]
    * profiles["balance_component"]
    +
    WEIGHTS["capacity"]
    * profiles["capacity_component"]
    +
    WEIGHTS["source_diversity"]
    * profiles["source_diversity_component"]
    +
    WEIGHTS["domain_diversity"]
    * profiles["domain_diversity_component"]
    +
    WEIGHTS["leakage"]
    * profiles["leakage_component"]
)


# ============================================================
# RANK LANGUAGES
# ============================================================

eligible = profiles[
    profiles["eligible"]
].copy()

eligible = eligible.sort_values(
    by=[
        "quality_score",
        balance_col,
        capacity_col,
    ],
    ascending=[
        False,
        False,
        False,
    ],
).reset_index(drop=True)

eligible["rank"] = (
    eligible.index + 1
)


# ============================================================
# PRINT INDIVIDUAL RANKING
# ============================================================

print()
print("=" * 90)
print("ELIGIBLE LANGUAGE RANKING")
print("=" * 90)

for _, row in eligible.iterrows():

    language = row[language_col]

    print(
        f"{int(row['rank']):2d}. "
        f"{language:4s} | "
        f"quality={row['quality_score']:.4f} | "
        f"balance={row[balance_col]:.4f} | "
        f"capacity={int(row[capacity_col]):,} | "
        f"sources={int(row[source_count_col])} | "
        f"domains={int(row[domain_count_col])} | "
        f"overlap={int(row['total_cross_split_overlap']):,}"
    )


# ============================================================
# CANDIDATE SET EVALUATION
# ============================================================

print()
print("=" * 90)
print("CANDIDATE LANGUAGE-SET ANALYSIS")
print("=" * 90)

candidate_results = []


def evaluate_language_set(
    language_rows: pd.DataFrame,
) -> dict:

    languages = list(
        language_rows[language_col]
    )

    if not languages:
        return {}

    # --------------------------------------------------------
    # Balance
    # --------------------------------------------------------

    balance_score = (
        language_rows[balance_col]
        .mean()
    )

    # --------------------------------------------------------
    # Capacity
    # --------------------------------------------------------
    #
    # Important:
    # We do NOT sum capacities.
    #
    # The final balanced multilingual dataset should use
    # the same number of examples per class across languages.
    #
    # Therefore the bottleneck language matters.
    #

    minimum_capacity = (
        language_rows[capacity_col]
        .min()
    )

    median_capacity = (
        language_rows[capacity_col]
        .median()
    )

    # Log-normalized capacity.
    capacity_component = (
        __import__("math").log1p(
            minimum_capacity
        )
        /
        __import__("math").log1p(
            eligible[capacity_col].max()
        )
    )

    # --------------------------------------------------------
    # Source diversity
    # --------------------------------------------------------

    source_diversity = (
        language_rows[
            "source_diversity_component"
        ].mean()
    )

    # --------------------------------------------------------
    # Domain diversity
    # --------------------------------------------------------

    domain_diversity = (
        language_rows[
            "domain_diversity_component"
        ].mean()
    )

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

    leakage = (
        language_rows[
            "leakage_component"
        ].mean()
    )

    # --------------------------------------------------------
    # Overall set score
    # --------------------------------------------------------

    set_score = (
        WEIGHTS["balance"]
        * balance_score
        +
        WEIGHTS["capacity"]
        * capacity_component
        +
        WEIGHTS["source_diversity"]
        * source_diversity
        +
        WEIGHTS["domain_diversity"]
        * domain_diversity
        +
        WEIGHTS["leakage"]
        * leakage
    )

    return {
        "language_count": len(languages),
        "languages": languages,
        "score": set_score,
        "balance_score": balance_score,
        "minimum_balanced_capacity": minimum_capacity,
        "median_balanced_capacity": median_capacity,
        "source_diversity": source_diversity,
        "domain_diversity": domain_diversity,
        "leakage_score": leakage,
        "total_rows_before_balancing": int(
            language_rows[total_col].sum()
        ),
        "total_cross_split_overlap": int(
            language_rows[
                "total_cross_split_overlap"
            ].sum()
        ),
    }


# ============================================================
# GENERATE CANDIDATE SETS
# ============================================================

for set_size in SET_SIZES:

    if len(eligible) < set_size:
        continue

    print()
    print(
        f"Evaluating all candidate sets of size {set_size}..."
    )

    for combination_rows in combinations(
        eligible.index,
        set_size,
    ):

        subset = eligible.loc[
            list(combination_rows)
        ]

        result = evaluate_language_set(
            subset
        )

        candidate_results.append(
            result
        )


# ============================================================
# SORT CANDIDATE SETS
# ============================================================

candidate_df = pd.DataFrame(
    candidate_results
)

if candidate_df.empty:
    raise RuntimeError(
        "No candidate language sets could be evaluated."
    )

candidate_df = candidate_df.sort_values(
    by=[
        "score",
        "balance_score",
        "minimum_balanced_capacity",
    ],
    ascending=[
        False,
        False,
        False,
    ],
).reset_index(drop=True)


candidate_df["rank"] = (
    candidate_df.index + 1
)


# ============================================================
# DISPLAY BEST SETS
# ============================================================

print()
print("=" * 90)
print("TOP CANDIDATE LANGUAGE SETS")
print("=" * 90)

for _, row in candidate_df.head(20).iterrows():

    languages = ", ".join(
        row["languages"]
    )

    print(
        f"Rank {int(row['rank']):2d} | "
        f"size={int(row['language_count'])} | "
        f"score={row['score']:.4f} | "
        f"balance={row['balance_score']:.4f} | "
        f"minimum_capacity="
        f"{int(row['minimum_balanced_capacity']):,} | "
        f"overlap="
        f"{int(row['total_cross_split_overlap']):,}"
    )

    print(
        f"          {languages}"
    )


# ============================================================
# BEST SET
# ============================================================

best = candidate_df.iloc[0]

best_languages = list(
    best["languages"]
)

best_rows = eligible[
    eligible[language_col].isin(
        best_languages
    )
].copy()

best_rows = best_rows.sort_values(
    by="quality_score",
    ascending=False,
)


# ============================================================
# SAVE BEST LANGUAGE SET
# ============================================================

best_languages_path = (
    OUTPUT_DIR
    / "selected_languages.json"
)

with open(
    best_languages_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "selection_status": "candidate_best_set",
            "languages": best_languages,
            "language_count": len(best_languages),
            "score": float(
                best["score"]
            ),
            "balance_score": float(
                best["balance_score"]
            ),
            "minimum_balanced_capacity": int(
                best[
                    "minimum_balanced_capacity"
                ]
            ),
            "median_balanced_capacity": float(
                best[
                    "median_balanced_capacity"
                ]
            ),
            "source_diversity": float(
                best["source_diversity"]
            ),
            "domain_diversity": float(
                best["domain_diversity"]
            ),
            "leakage_score": float(
                best["leakage_score"]
            ),
            "total_cross_split_overlap": int(
                best[
                    "total_cross_split_overlap"
                ]
            ),
        },
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SAVE LANGUAGE RANKING
# ============================================================

language_ranking_path = (
    OUTPUT_DIR
    / "language_ranking.csv"
)

eligible.to_csv(
    language_ranking_path,
    index=False,
)


# ============================================================
# SAVE ALL CANDIDATE SETS
# ============================================================

candidate_sets_path = (
    OUTPUT_DIR
    / "candidate_language_sets.csv"
)

candidate_df_export = candidate_df.copy()

candidate_df_export["languages"] = (
    candidate_df_export["languages"]
    .apply(
        lambda values:
        ",".join(values)
    )
)

candidate_df_export.to_csv(
    candidate_sets_path,
    index=False,
)


# ============================================================
# SAVE BEST SET PROFILE
# ============================================================

best_profile_path = (
    OUTPUT_DIR
    / "selected_language_profiles.csv"
)

best_rows.to_csv(
    best_profile_path,
    index=False,
)


# ============================================================
# SAVE FINAL REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "final_language_selection_report.md"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "# Final Language Selection Analysis\n\n"
    )

    f.write(
        "## Dataset\n\n"
        "- Dataset: `clapAI/MultiLingualSentiment`\n"
        "- Labels: positive, neutral, negative\n"
        "- No fixed number of languages was imposed.\n"
        "- Raw language size alone was not used as the selection criterion.\n\n"
    )

    f.write(
        "## Selection Principle\n\n"
    )

    f.write(
        "The analysis prioritizes dataset validity and balance "
        "over raw sample count. Source/domain concentration and "
        "cross-split overlap are included because a language can "
        "appear large and balanced while still being strongly "
        "associated with one source or domain.\n\n"
    )

    f.write(
        "### Analysis weights\n\n"
    )

    f.write(
        "| Criterion | Weight |\n"
        "|---|---:|\n"
        f"| Class balance | {WEIGHTS['balance']:.2f} |\n"
        f"| Usable balanced capacity | {WEIGHTS['capacity']:.2f} |\n"
        f"| Source diversity | {WEIGHTS['source_diversity']:.2f} |\n"
        f"| Domain diversity | {WEIGHTS['domain_diversity']:.2f} |\n"
        f"| Leakage safety | {WEIGHTS['leakage']:.2f} |\n\n"
    )

    f.write(
        "## Best Candidate Set\n\n"
    )

    f.write(
        f"**Languages:** "
        f"{', '.join(best_languages)}\n\n"
    )

    f.write(
        f"**Number of languages:** "
        f"{len(best_languages)}\n\n"
    )

    f.write(
        f"**Set score:** "
        f"{best['score']:.4f}\n\n"
    )

    f.write(
        f"**Mean balance score:** "
        f"{best['balance_score']:.4f}\n\n"
    )

    f.write(
        f"**Minimum balanced capacity:** "
        f"{int(best['minimum_balanced_capacity']):,}\n\n"
    )

    f.write(
        f"**Median balanced capacity:** "
        f"{int(best['median_balanced_capacity']):,}\n\n"
    )

    f.write(
        f"**Source diversity score:** "
        f"{best['source_diversity']:.4f}\n\n"
    )

    f.write(
        f"**Domain diversity score:** "
        f"{best['domain_diversity']:.4f}\n\n"
    )

    f.write(
        f"**Leakage score:** "
        f"{best['leakage_score']:.4f}\n\n"
    )

    f.write(
        f"**Cross-split overlap:** "
        f"{int(best['total_cross_split_overlap']):,}\n\n"
    )

    f.write(
        "## Selected Language Profiles\n\n"
    )

    f.write(
        "| Language | Train Total | "
        "Balanced Capacity | Balance Score | "
        "Sources | Domains | Cross-Split Overlap |\n"
    )

    f.write(
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )

    for _, row in best_rows.iterrows():

        f.write(
            f"| {row[language_col]} | "
            f"{int(row[total_col]):,} | "
            f"{int(row[capacity_col]):,} | "
            f"{row[balance_col]:.4f} | "
            f"{int(row[source_count_col])} | "
            f"{int(row[domain_count_col])} | "
            f"{int(row['total_cross_split_overlap']):,} |\n"
        )

    f.write(
        "\n## Important Interpretation\n\n"
    )

    f.write(
        "The selected set is the highest-scoring candidate under "
        "the stated analytical criteria. It is not a claim that "
        "this is the only scientifically valid language set. "
        "Before finalizing the dataset, the selected languages "
        "should still undergo text-quality inspection, duplicate "
        "inspection, label-quality auditing, and preprocessing "
        "experiments.\n\n"
    )

    f.write(
        "No model training was performed by this script.\n"
    )


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print()
print("=" * 90)
print("FINAL LANGUAGE SELECTION ANALYSIS COMPLETE")
print("=" * 90)

print()
print("BEST CANDIDATE SET")
print("-" * 90)

print(
    "Languages:",
    ", ".join(best_languages)
)

print(
    "Number of languages:",
    len(best_languages)
)

print(
    "Score:",
    f"{best['score']:.4f}"
)

print(
    "Balance:",
    f"{best['balance_score']:.4f}"
)

print(
    "Minimum balanced capacity:",
    f"{int(best['minimum_balanced_capacity']):,}"
)

print(
    "Cross-split overlap:",
    f"{int(best['total_cross_split_overlap']):,}"
)

print()
print("FILES SAVED")
print("-" * 90)

print(
    "Language ranking:",
    language_ranking_path
)

print(
    "Candidate language sets:",
    candidate_sets_path
)

print(
    "Selected language profiles:",
    best_profile_path
)

print(
    "Selected languages JSON:",
    best_languages_path
)

print(
    "Final report:",
    report_path
)

print()
print(
    "No model training was performed."
)