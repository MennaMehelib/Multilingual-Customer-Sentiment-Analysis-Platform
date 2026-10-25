from collections import Counter, defaultdict
from datasets import load_dataset


DATASET_NAME = "clapAI/MultiLingualSentiment"


def audit_split(dataset, split_name):
    language_counts = Counter()
    language_label_counts = defaultdict(Counter)
    language_source_counts = defaultdict(Counter)
    language_domain_counts = defaultdict(Counter)

    total = 0
    missing_text = 0

    print(f"\nAuditing split: {split_name}")

    for row in dataset[split_name]:
        total += 1

        text = row.get("text")
        label = row.get("label")
        language = row.get("language")
        source = row.get("source")
        domain = row.get("domain")

        if not text or not str(text).strip():
            missing_text += 1

        language_counts[language] += 1
        language_label_counts[language][label] += 1

        if source:
            language_source_counts[language][source] += 1

        if domain:
            language_domain_counts[language][domain] += 1

        if total % 100000 == 0:
            print(f"  Processed {total:,} rows...")

    print(f"\nFinished {split_name}: {total:,} rows")

    return {
        "total": total,
        "missing_text": missing_text,
        "language_counts": language_counts,
        "language_label_counts": language_label_counts,
        "language_source_counts": language_source_counts,
        "language_domain_counts": language_domain_counts,
    }


def print_language_summary(result):
    print("\n" + "=" * 90)
    print("LANGUAGE × LABEL SUMMARY")
    print("=" * 90)

    labels = ["positive", "neutral", "negative"]

    for language in sorted(result["language_counts"]):
        total = result["language_counts"][language]
        counts = result["language_label_counts"][language]

        positive = counts.get("positive", 0)
        neutral = counts.get("neutral", 0)
        negative = counts.get("negative", 0)

        minority = min(positive, neutral, negative)
        balanced_capacity = minority * 3

        print(
            f"{language:12}"
            f" total={total:10,}"
            f" positive={positive:10,}"
            f" neutral={neutral:10,}"
            f" negative={negative:10,}"
            f" balanced_capacity={balanced_capacity:10,}"
        )


def print_source_summary(result):
    print("\n" + "=" * 90)
    print("TOP SOURCES PER LANGUAGE")
    print("=" * 90)

    for language in sorted(result["language_source_counts"]):
        print(f"\n[{language}]")

        sources = result["language_source_counts"][language]

        for source, count in sources.most_common(5):
            print(f"  {count:10,}  {source}")


def print_domain_summary(result):
    print("\n" + "=" * 90)
    print("DOMAINS PER LANGUAGE")
    print("=" * 90)

    for language in sorted(result["language_domain_counts"]):
        print(f"\n[{language}]")

        domains = result["language_domain_counts"][language]

        for domain, count in domains.most_common():
            print(f"  {count:10,}  {domain}")


def main():
    print("Loading dataset in streaming mode...")

    dataset = load_dataset(
        DATASET_NAME,
        streaming=True,
    )

    print("\nDataset loaded.")
    print(dataset)

    results = {}

    for split in dataset:
        results[split] = audit_split(dataset, split)

    for split, result in results.items():
        print(f"\n\n{'#' * 90}")
        print(f"# SPLIT: {split}")
        print(f"{'#' * 90}")

        print_language_summary(result)

        print(
            f"\nMissing/empty text: "
            f"{result['missing_text']:,}"
        )

        print_source_summary(result)
        print_domain_summary(result)


if __name__ == "__main__":
    main()