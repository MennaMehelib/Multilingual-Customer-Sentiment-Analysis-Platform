from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math

from datasets import load_dataset


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "clapAI/MultiLingualSentiment"

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "language_selection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["positive", "neutral", "negative"]

# Languages with no usable 3-class setup are automatically excluded.
MIN_TRAIN_PER_CLASS = 500
MIN_VALIDATION_PER_CLASS = 100
MIN_TEST_PER_CLASS = 100

# We do NOT force a fixed language count.
# The script evaluates each language independently and then
# evaluates candidate sets.


# ============================================================
# Helpers
# ============================================================

def clean_language(value):
    if value is None:
        return "unknown"

    value = str(value).strip().lower()

    if not value:
        return "unknown"

    return value


def clean_label(value):
    if value is None:
        return "unknown"

    return str(value).strip().lower()


def safe_percentage(part, total):
    if total == 0:
        return 0.0

    return (part / total) * 100


def balance_capacity(counts):
    """
    Maximum number of samples that can remain after perfectly
    balancing the three sentiment classes.

    Example:
        positive = 1000
        neutral  = 500
        negative = 800

    capacity = 3 * 500 = 1500
    """

    values = [
        counts.get(label, 0)
        for label in LABELS
    ]

    if not all(values):
        return 0

    return 3 * min(values)


# ============================================================
# Audit dataset
# ============================================================

print("=" * 90)
print("Loading dataset...")
print("=" * 90)

dataset = load_dataset(
    DATASET_NAME,
    streaming=True,
)

print(dataset)


# Structure:
#
# statistics[split][language]["labels"]
# statistics[split][language]["sources"]
# statistics[split][language]["domains"]
#

statistics = defaultdict(
    lambda: defaultdict(
        lambda: {
            "total": 0,
            "labels": Counter(),
            "sources": Counter(),
            "domains": Counter(),
            "missing_text": 0,
        }
    )
)


# ============================================================
# Process splits
# ============================================================

for split in ["train", "validation", "test"]:

    print()
    print("=" * 90)
    print(f"Auditing split: {split}")
    print("=" * 90)

    processed = 0

    for row in dataset[split]:

        processed += 1

        if processed % 100_000 == 0:
            print(f"Processed {processed:,} rows...")

        language = clean_language(
            row.get("language")
        )

        label = clean_label(
            row.get("label")
        )

        source = str(
            row.get("source") or "unknown"
        ).strip()

        domain = str(
            row.get("domain") or "unknown"
        ).strip()

        text = row.get("text")

        stats = statistics[split][language]

        stats["total"] += 1

        if text is None or not str(text).strip():
            stats["missing_text"] += 1

        stats["labels"][label] += 1
        stats["sources"][source] += 1
        stats["domains"][domain] += 1

    print(
        f"Finished {split}: {processed:,} rows"
    )


# ============================================================
# Build language profiles
# ============================================================

languages = sorted(
    set(
        language
        for split in statistics.values()
        for language in split.keys()
    )
)

profiles = []

for language in languages:

    train = statistics["train"].get(language)
    validation = statistics["validation"].get(language)
    test = statistics["test"].get(language)

    if train is None:
        continue

    train_labels = train["labels"]
    val_labels = validation["labels"]
    test_labels = test["labels"]

    train_counts = {
        label: train_labels.get(label, 0)
        for label in LABELS
    }

    val_counts = {
        label: val_labels.get(label, 0)
        for label in LABELS
    }

    test_counts = {
        label: test_labels.get(label, 0)
        for label in LABELS
    }

    train_capacity = balance_capacity(
        train_counts
    )

    validation_capacity = balance_capacity(
        val_counts
    )

    test_capacity = balance_capacity(
        test_counts
    )

    # The smallest class proportion determines class balance.
    train_total = sum(train_counts.values())

    class_percentages = {
        label: safe_percentage(
            train_counts[label],
            train_total
        )
        for label in LABELS
    }

    minority_ratio = (
        min(class_percentages.values())
        if class_percentages
        else 0
    )

    majority_ratio = (
        max(class_percentages.values())
        if class_percentages
        else 0
    )

    # Balance score:
    # 1.0 = perfectly balanced
    # approaches 0 = highly imbalanced
    if majority_ratio > 0:
        balance_score = (
            minority_ratio / majority_ratio
        )
    else:
        balance_score = 0

    # Source concentration:
    # percentage of language data coming from
    # the largest source.
    total_source_rows = sum(
        train["sources"].values()
    )

    largest_source_count = (
        max(train["sources"].values())
        if train["sources"]
        else 0
    )

    largest_source_pct = safe_percentage(
        largest_source_count,
        total_source_rows,
    )

    source_diversity = len(
        train["sources"]
    )

    domain_diversity = len(
        train["domains"]
    )

    has_all_classes = all(
        train_counts[label] > 0
        for label in LABELS
    )

    has_valid_split_classes = (
        all(
            val_counts[label] >= MIN_VALIDATION_PER_CLASS
            for label in LABELS
        )
        and
        all(
            test_counts[label] >= MIN_TEST_PER_CLASS
            for label in LABELS
        )
    )

    eligible = (
        has_all_classes
        and train_capacity > 0
        and min(train_counts.values())
            >= MIN_TRAIN_PER_CLASS
        and has_valid_split_classes
    )

    # --------------------------------------------------------
    # Ranking score
    # --------------------------------------------------------
    #
    # This is deliberately transparent.
    #
    # We prioritize:
    #   1. three-class validity
    #   2. class balance
    #   3. balanced usable capacity
    #   4. source diversity
    #   5. domain diversity
    #
    # We do NOT simply rank by total rows.
    #

    capacity_score = (
        math.log1p(train_capacity)
        if train_capacity > 0
        else 0
    )

    source_score = math.log1p(
        source_diversity
    )

    domain_score = math.log1p(
        domain_diversity
    )

    ranking_score = (
        balance_score * 50
        + capacity_score * 5
        + source_score * 2
        + domain_score * 2
    )

    if not eligible:
        ranking_score = -1

    profiles.append(
        {
            "language": language,

            "train_total": train["total"],
            "validation_total": validation["total"],
            "test_total": test["total"],

            "train_positive": train_counts["positive"],
            "train_neutral": train_counts["neutral"],
            "train_negative": train_counts["negative"],

            "validation_positive": val_counts["positive"],
            "validation_neutral": val_counts["neutral"],
            "validation_negative": val_counts["negative"],

            "test_positive": test_counts["positive"],
            "test_neutral": test_counts["neutral"],
            "test_negative": test_counts["negative"],

            "train_balanced_capacity": train_capacity,
            "validation_balanced_capacity": validation_capacity,
            "test_balanced_capacity": test_capacity,

            "minority_class_count": min(
                train_counts.values()
            ),

            "minority_class_percentage": round(
                minority_ratio,
                3,
            ),

            "majority_class_percentage": round(
                majority_ratio,
                3,
            ),

            "balance_score": round(
                balance_score,
                4,
            ),

            "source_count": source_diversity,

            "largest_source_percentage": round(
                largest_source_pct,
                3,
            ),

            "domain_count": domain_diversity,

            "missing_text_train": train["missing_text"],
            "missing_text_validation": validation["missing_text"],
            "missing_text_test": test["missing_text"],

            "has_all_three_classes": has_all_classes,

            "valid_validation_split": all(
                val_counts[label] >= MIN_VALIDATION_PER_CLASS
                for label in LABELS
            ),

            "valid_test_split": all(
                test_counts[label] >= MIN_TEST_PER_CLASS
                for label in LABELS
            ),

            "eligible": eligible,

            "ranking_score": round(
                ranking_score,
                4,
            ),
        }
    )


# ============================================================
# Sort profiles
# ============================================================

profiles_sorted = sorted(
    profiles,
    key=lambda x: x["ranking_score"],
    reverse=True,
)


# ============================================================
# Save language profile CSV
# ============================================================

profile_csv = (
    OUTPUT_DIR
    / "language_profiles.csv"
)

fieldnames = list(
    profiles_sorted[0].keys()
)

with open(
    profile_csv,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        profiles_sorted
    )


# ============================================================
# Save JSON
# ============================================================

profile_json = (
    OUTPUT_DIR
    / "language_profiles.json"
)

with open(
    profile_json,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        profiles_sorted,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Generate Markdown report
# ============================================================

report_path = (
    OUTPUT_DIR
    / "language_selection_report.md"
)

eligible_languages = [
    p
    for p in profiles_sorted
    if p["eligible"]
]

excluded_languages = [
    p
    for p in profiles_sorted
    if not p["eligible"]
]


with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "# Multilingual Sentiment Language Selection Report\n\n"
    )

    f.write(
        "## Dataset\n\n"
    )

    f.write(
        f"- Dataset: `{DATASET_NAME}`\n"
        "- Labels: positive, neutral, negative\n"
        "- Selection is based on the actual dataset audit.\n"
        "- No fixed number of languages was imposed.\n"
        "- Total row count alone is NOT used as the selection criterion.\n\n"
    )

    f.write(
        "## Selection Criteria\n\n"
    )

    f.write(
        "A language is considered eligible when:\n\n"
        f"1. All three sentiment classes are present in training.\n"
        f"2. Each training class has at least {MIN_TRAIN_PER_CLASS:,} samples.\n"
        f"3. Each validation class has at least {MIN_VALIDATION_PER_CLASS:,} samples.\n"
        f"4. Each test class has at least {MIN_TEST_PER_CLASS:,} samples.\n"
        "5. The language has usable balanced training capacity.\n\n"
    )

    f.write(
        "The ranking favors class balance and usable balanced capacity, "
        "while also considering source and domain diversity.\n\n"
    )

    f.write(
        "## Eligible Languages\n\n"
    )

    f.write(
        "| Rank | Language | Train | Pos | Neu | Neg | "
        "Balanced Capacity | Balance Score | Sources | Domains | Score |\n"
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )

    for rank, p in enumerate(
        eligible_languages,
        start=1,
    ):

        f.write(
            f"| {rank} | {p['language']} | "
            f"{p['train_total']:,} | "
            f"{p['train_positive']:,} | "
            f"{p['train_neutral']:,} | "
            f"{p['train_negative']:,} | "
            f"{p['train_balanced_capacity']:,} | "
            f"{p['balance_score']:.4f} | "
            f"{p['source_count']} | "
            f"{p['domain_count']} | "
            f"{p['ranking_score']:.4f} |\n"
        )

    f.write(
        "\n## Excluded Languages\n\n"
    )

    f.write(
        "| Language | Train | Pos | Neu | Neg | "
        "Balanced Capacity | Reason |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
    )

    for p in excluded_languages:

        reasons = []

        if not p["has_all_three_classes"]:
            reasons.append(
                "missing one or more sentiment classes"
            )

        if p["minority_class_count"] < MIN_TRAIN_PER_CLASS:
            reasons.append(
                "training minority class below minimum"
            )

        if not p["valid_validation_split"]:
            reasons.append(
                "validation split insufficient"
            )

        if not p["valid_test_split"]:
            reasons.append(
                "test split insufficient"
            )

        reason = "; ".join(reasons)

        f.write(
            f"| {p['language']} | "
            f"{p['train_total']:,} | "
            f"{p['train_positive']:,} | "
            f"{p['train_neutral']:,} | "
            f"{p['train_negative']:,} | "
            f"{p['train_balanced_capacity']:,} | "
            f"{reason} |\n"
        )

    f.write(
        "\n## Important Scientific Note\n\n"
        "This report does not automatically declare the final language set. "
        "It ranks languages using measurable dataset properties. "
        "The final set should also be checked for source/domain confounding, "
        "duplicate leakage and label quality before model training.\n"
    )


# ============================================================
# Console output
# ============================================================

print()
print("=" * 90)
print("LANGUAGE SELECTION COMPLETE")
print("=" * 90)

print()
print("Eligible languages:")
print("-" * 90)

for rank, p in enumerate(
    eligible_languages,
    start=1,
):

    print(
        f"{rank:2}. {p['language']:15} "
        f"balanced_capacity={p['train_balanced_capacity']:>10,} "
        f"balance_score={p['balance_score']:.4f} "
        f"score={p['ranking_score']:.2f}"
    )


print()
print("Excluded languages:")
print("-" * 90)

for p in excluded_languages:

    print(
        f"{p['language']:15} "
        f"balanced_capacity={p['train_balanced_capacity']:>10,}"
    )


print()
print("=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    f"Language profiles CSV:\n{profile_csv}"
)

print(
    f"Language profiles JSON:\n{profile_json}"
)

print(
    f"Selection report:\n{report_path}"
)

print()
print("No model training was performed.")
print("No final language set was hard-coded.")