from datasets import load_dataset
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import hashlib


DATASET_NAME = "clapAI/MultiLingualSentiment"

OUTPUT_DIR = Path("data/processed/language_selection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "validation", "test"]


# ============================================================
# Configuration
# ============================================================

# Languages that survived the previous selection step.
# This is NOT the final decision.
CANDIDATE_LANGUAGES = [
    "en",
    "zh",
    "ko",
    "pt",
    "ar",
    "ja",
    "es",
    "de",
    "fr",
    "ru",
    "hi",
    "vi",
    "ms",
    "id",
]


# ============================================================
# Helpers
# ============================================================

def normalize_text(text):
    if text is None:
        return ""

    return " ".join(str(text).strip().split()).lower()


def text_hash(text):
    normalized = normalize_text(text)

    if not normalized:
        return None

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def safe_percentage(part, total):
    if total == 0:
        return 0.0

    return round((part / total) * 100, 4)


# ============================================================
# Load dataset
# ============================================================

print("=" * 90)
print("Loading dataset")
print("=" * 90)

dataset = load_dataset(
    DATASET_NAME,
    streaming=True
)

print(dataset)


# ============================================================
# Containers
# ============================================================

language_label_counts = defaultdict(Counter)
language_source_counts = defaultdict(Counter)
language_domain_counts = defaultdict(Counter)

language_total = Counter()

split_language_total = defaultdict(Counter)

# Text hashes per split/language.
# Used for cross-split overlap detection.
split_hashes = {
    split: defaultdict(set)
    for split in SPLITS
}


# ============================================================
# Pass 1 — collect statistics
# ============================================================

for split in SPLITS:

    print()
    print("=" * 90)
    print(f"Processing split: {split}")
    print("=" * 90)

    processed = 0

    for row in dataset[split]:

        language = str(row.get("language", "")).strip().lower()
        label = str(row.get("label", "")).strip().lower()
        source = str(row.get("source", "")).strip()
        domain = str(row.get("domain", "")).strip()

        if language not in CANDIDATE_LANGUAGES:
            continue

        language_total[language] += 1
        split_language_total[split][language] += 1

        language_label_counts[language][label] += 1

        if source:
            language_source_counts[language][source] += 1

        if domain:
            language_domain_counts[language][domain] += 1

        # Store hashes for overlap checking.
        h = text_hash(row.get("text"))

        if h:
            split_hashes[split][language].add(h)

        processed += 1

        if processed % 100000 == 0:
            print(f"Processed {processed:,} candidate-language rows...")

    print(
        f"Finished {split}: "
        f"{processed:,} candidate-language rows"
    )


# ============================================================
# Language statistics
# ============================================================

print()
print("=" * 90)
print("BUILDING LANGUAGE VALIDATION REPORT")
print("=" * 90)


language_report = []

for language in CANDIDATE_LANGUAGES:

    counts = language_label_counts[language]

    positive = counts.get("positive", 0)
    neutral = counts.get("neutral", 0)
    negative = counts.get("negative", 0)

    total = positive + neutral + negative

    minority = min(
        positive,
        neutral,
        negative
    )

    balanced_capacity = minority * 3

    max_class = max(
        positive,
        neutral,
        negative
    )

    # 1 = perfectly balanced.
    # Lower = more imbalanced.
    if total > 0:
        balance_score = (
            1
            - (
                max_class - minority
            ) / total
        )
    else:
        balance_score = 0

    # Source concentration:
    # percentage represented by the largest source.
    sources = language_source_counts[language]

    if sources:
        largest_source_count = max(sources.values())
        source_concentration = (
            largest_source_count / total
            if total
            else 0
        )
    else:
        source_concentration = 0

    # Domain concentration.
    domains = language_domain_counts[language]

    if domains:
        largest_domain_count = max(domains.values())
        domain_concentration = (
            largest_domain_count / total
            if total
            else 0
        )
    else:
        domain_concentration = 0

    language_report.append({
        "language": language,
        "total": total,

        "positive": positive,
        "neutral": neutral,
        "negative": negative,

        "positive_pct": safe_percentage(
            positive, total
        ),

        "neutral_pct": safe_percentage(
            neutral, total
        ),

        "negative_pct": safe_percentage(
            negative, total
        ),

        "minority_class_count": minority,

        "balanced_capacity": balanced_capacity,

        "balance_score": round(
            balance_score,
            6
        ),

        "source_count": len(sources),

        "largest_source_count": (
            max(sources.values())
            if sources
            else 0
        ),

        "source_concentration": round(
            source_concentration,
            6
        ),

        "domain_count": len(domains),

        "largest_domain_count": (
            max(domains.values())
            if domains
            else 0
        ),

        "domain_concentration": round(
            domain_concentration,
            6
        ),
    })


# ============================================================
# Cross-split overlap
# ============================================================

print()
print("=" * 90)
print("CHECKING CROSS-SPLIT TEXT OVERLAP")
print("=" * 90)


overlap_report = []

for language in CANDIDATE_LANGUAGES:

    train_hashes = split_hashes["train"][language]
    val_hashes = split_hashes["validation"][language]
    test_hashes = split_hashes["test"][language]

    train_val = train_hashes & val_hashes
    train_test = train_hashes & test_hashes
    val_test = val_hashes & test_hashes

    overlap_report.append({

        "language": language,

        "train_unique_texts": len(
            train_hashes
        ),

        "validation_unique_texts": len(
            val_hashes
        ),

        "test_unique_texts": len(
            test_hashes
        ),

        "train_validation_overlap": len(
            train_val
        ),

        "train_test_overlap": len(
            train_test
        ),

        "validation_test_overlap": len(
            val_test
        ),

        "total_cross_split_overlap": (
            len(train_val)
            + len(train_test)
            + len(val_test)
        ),
    })


# ============================================================
# Save language report
# ============================================================

language_csv = (
    OUTPUT_DIR /
    "candidate_validation_profiles.csv"
)

with open(
    language_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=language_report[0].keys()
    )

    writer.writeheader()
    writer.writerows(language_report)


# ============================================================
# Save overlap report
# ============================================================

overlap_csv = (
    OUTPUT_DIR /
    "cross_split_overlap.csv"
)

with open(
    overlap_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=overlap_report[0].keys()
    )

    writer.writeheader()
    writer.writerows(overlap_report)


# ============================================================
# Save detailed source report
# ============================================================

source_report = {}

for language in CANDIDATE_LANGUAGES:

    source_report[language] = [
        {
            "source": source,
            "count": count
        }

        for source, count
        in language_source_counts[language].most_common()
    ]


source_json = (
    OUTPUT_DIR /
    "language_sources.json"
)

with open(
    source_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        source_report,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# Save detailed domain report
# ============================================================

domain_report = {}

for language in CANDIDATE_LANGUAGES:

    domain_report[language] = [
        {
            "domain": domain,
            "count": count
        }

        for domain, count
        in language_domain_counts[language].most_common()
    ]


domain_json = (
    OUTPUT_DIR /
    "language_domains.json"
)

with open(
    domain_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        domain_report,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# Generate Markdown report
# ============================================================

report_path = (
    OUTPUT_DIR /
    "candidate_validation_report.md"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("# Candidate Language Validation Report\n\n")

    f.write(
        "This report validates candidate languages before "
        "final dataset selection.\n\n"
    )

    f.write(
        "The validation considers class balance, "
        "source concentration, domain concentration, "
        "and cross-split text overlap.\n\n"
    )

    f.write(
        "> Important: this report does not automatically "
        "declare a final language set.\n\n"
    )

    f.write("## Language Summary\n\n")

    f.write(
        "| Language | Total | Positive | Neutral | Negative | "
        "Balanced Capacity | Balance Score | "
        "Sources | Largest Source % | Domains | Largest Domain % |\n"
    )

    f.write(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )

    for row in sorted(
        language_report,
        key=lambda x: x["balanced_capacity"],
        reverse=True
    ):

        f.write(
            f"| {row['language']} "
            f"| {row['total']:,} "
            f"| {row['positive']:,} "
            f"| {row['neutral']:,} "
            f"| {row['negative']:,} "
            f"| {row['balanced_capacity']:,} "
            f"| {row['balance_score']:.4f} "
            f"| {row['source_count']} "
            f"| {row['source_concentration'] * 100:.2f}% "
            f"| {row['domain_count']} "
            f"| {row['domain_concentration'] * 100:.2f}% |\n"
        )

    f.write("\n## Cross-Split Overlap\n\n")

    f.write(
        "| Language | Train-Val | Train-Test | Val-Test | Total Overlap |\n"
    )

    f.write(
        "|---|---:|---:|---:|---:|\n"
    )

    for row in overlap_report:

        f.write(
            f"| {row['language']} "
            f"| {row['train_validation_overlap']:,} "
            f"| {row['train_test_overlap']:,} "
            f"| {row['validation_test_overlap']:,} "
            f"| {row['total_cross_split_overlap']:,} |\n"
        )

    f.write("\n## Interpretation Rules\n\n")

    f.write(
        "- **Balance score:** higher is better.\n"
        "- **Balanced capacity:** maximum number of rows "
        "that can be used for a perfectly balanced 3-class subset.\n"
        "- **Source concentration:** lower is generally safer; "
        "very high concentration means a language may be strongly "
        "associated with one source.\n"
        "- **Domain concentration:** lower is generally safer "
        "when the goal is general customer sentiment.\n"
        "- **Cross-split overlap:** should ideally be zero.\n"
    )


# ============================================================
# Console summary
# ============================================================

print()
print("=" * 90)
print("VALIDATION COMPLETE")
print("=" * 90)

print()
print("Language profiles:")
print(language_csv)

print()
print("Cross-split overlap:")
print(overlap_csv)

print()
print("Source report:")
print(source_json)

print()
print("Domain report:")
print(domain_json)

print()
print("Markdown report:")
print(report_path)

print()
print("No model training was performed.")
print("No final language set was selected.")