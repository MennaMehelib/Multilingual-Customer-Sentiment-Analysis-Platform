"""
Leakage-Safe Multilingual Dataset Extractor & Split Builder.
Extracts {en, zh, ar} from raw parquet files, deterministically cleans
cross-split leakage and conflicting duplicates, creates a standardized, balanced
benchmark split, and outputs audit metrics and summary reports.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


TARGET_LANGUAGES = ["en", "zh", "ar"]
TARGET_LABELS = ["positive", "neutral", "negative"]

# Balanced Benchmark Split Quotas
TRAIN_PER_CLASS_PER_LANG = 10000     # 10k * 3 classes * 3 langs = 90,000 train rows
VAL_PER_CLASS_PER_LANG = 1500        # 1.5k * 3 classes * 3 langs = 13,500 val rows
TEST_PER_CLASS_PER_LANG = 1500       # 1.5k * 3 classes * 3 langs = 13,500 test rows

RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
DATA_DIR = BASE_DIR / "data" / "processed"
SPLITS_DIR = DATA_DIR / "splits"
DATASET_DIR = DATA_DIR / "dataset"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)


def compute_text_hash(text: str) -> str:
    """Normalize text and compute sha256 hash for leakage and duplicate detection."""
    if not text:
        return ""
    normalized = " ".join(str(text).strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_and_filter_parquet(file_path: Path, split_name: str):
    """Read parquet, filter for target languages/labels and valid text."""
    print(f"\n[1] Reading {file_path.name} for {split_name}...")
    df = pd.read_parquet(
        file_path,
        columns=["text", "label", "language", "source", "domain"],
    )
    total_raw = len(df)
    print(f"  Raw rows loaded: {total_raw:,}")

    # Normalize text fields
    df["language"] = df["language"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()

    # Filter target languages & labels
    df = df[df["language"].isin(TARGET_LANGUAGES) & df["label"].isin(TARGET_LABELS)].copy()
    lang_filtered = len(df)

    # Filter invalid/empty text (< 3 chars)
    df = df[df["text"].str.len() >= 3].copy()
    valid_text_count = len(df)
    invalid_empty = lang_filtered - valid_text_count

    print(f"  Target language & label rows: {lang_filtered:,} (filtered {invalid_empty:,} invalid/short texts)")

    # Compute text hashes
    df["text_hash"] = df["text"].apply(compute_text_hash)

    return df, total_raw, invalid_empty


def build_clean_splits():
    print("=" * 80)
    print("LEAKAGE-SAFE DATASET EXTRACTION AND BALANCED BENCHMARK SPLIT CREATION")
    print(f"Target Languages: {TARGET_LANGUAGES}")
    print(f"Target Labels:    {TARGET_LABELS}")
    print(f"Random Seed:      {RANDOM_SEED}")
    print("=" * 80)

    train_shards = sorted(RAW_DIR.glob("train-*.parquet"))
    val_file = RAW_DIR / "validation.parquet"
    test_file = RAW_DIR / "test.parquet"

    if not train_shards:
        raise FileNotFoundError(f"Missing train shards in: {RAW_DIR}")
    for f in [val_file, test_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing required raw file: {f}")

    # Load raw data
    train_pools = []
    raw_train_total = 0
    train_invalid_empty = 0
    for shard_file in train_shards:
        pool_shard, raw_tot, inv_tot = load_and_filter_parquet(shard_file, f"train ({shard_file.name})")
        train_pools.append(pool_shard)
        raw_train_total += raw_tot
        train_invalid_empty += inv_tot
    train_pool = pd.concat(train_pools, ignore_index=True)

    val_pool, raw_val_total, val_invalid_empty = load_and_filter_parquet(val_file, "validation")
    test_pool, raw_test_total, test_invalid_empty = load_and_filter_parquet(test_file, "test")

    audit_ledger = {
        "target_languages": TARGET_LANGUAGES,
        "target_labels": TARGET_LABELS,
        "random_seed": RANDOM_SEED,
        "raw_counts": {
            "train_raw_streamed": raw_train_total,
            "validation_raw_streamed": raw_val_total,
            "test_raw_streamed": raw_test_total,
        },
        "invalid_empty_removed": {
            "train": train_invalid_empty,
            "validation": val_invalid_empty,
            "test": test_invalid_empty,
        },
        "leakage_removed": {},
        "splits": {},
    }

    # =========================================================================
    # Step 1: Process TRAIN Split
    # =========================================================================
    print("\n[2] Building TRAIN split (target: 10,000 per class per language)...")
    # Resolve within-split duplicate conflicts
    train_pool = train_pool.drop_duplicates(subset=["text_hash", "label"], keep="first")
    # Drop texts that appear with conflicting labels
    label_conflict_hashes = train_pool[train_pool.duplicated(subset=["text_hash"], keep=False)]["text_hash"].unique()
    train_pool = train_pool[~train_pool["text_hash"].isin(label_conflict_hashes)].copy()

    # Stratified deterministic sampling
    train_sampled_rows = []
    for lang in TARGET_LANGUAGES:
        for label in TARGET_LABELS:
            sub = train_pool[(train_pool["language"] == lang) & (train_pool["label"] == label)]
            if len(sub) < TRAIN_PER_CLASS_PER_LANG:
                raise ValueError(f"Insufficient samples for train {lang}-{label}: {len(sub)} < {TRAIN_PER_CLASS_PER_LANG}")
            sampled = sub.sample(n=TRAIN_PER_CLASS_PER_LANG, random_state=RANDOM_SEED)
            train_sampled_rows.append(sampled)

    train_df = pd.concat(train_sampled_rows, ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    train_hashes = set(train_df["text_hash"])
    print(f"  Constructed train split: {len(train_df):,} rows. Unique hashes: {len(train_hashes):,}")

    # =========================================================================
    # Step 2: Process VALIDATION Split (with Leakage Purge)
    # =========================================================================
    print("\n[3] Building VALIDATION split (target: 1,500 per class per language)...")
    initial_val_count = len(val_pool)
    # Leakage removal: remove any text in train
    val_pool = val_pool[~val_pool["text_hash"].isin(train_hashes)].copy()
    val_leakage_removed = initial_val_count - len(val_pool)
    print(f"  Removed {val_leakage_removed:,} cross-split leakage rows from validation (found in train).")
    audit_ledger["leakage_removed"]["train_val_overlap"] = int(val_leakage_removed)

    # Resolve within-split duplicates
    val_pool = val_pool.drop_duplicates(subset=["text_hash", "label"], keep="first")
    val_conflicts = val_pool[val_pool.duplicated(subset=["text_hash"], keep=False)]["text_hash"].unique()
    val_pool = val_pool[~val_pool["text_hash"].isin(val_conflicts)].copy()

    # Stratified deterministic sampling
    val_sampled_rows = []
    for lang in TARGET_LANGUAGES:
        for label in TARGET_LABELS:
            sub = val_pool[(val_pool["language"] == lang) & (val_pool["label"] == label)]
            if len(sub) < VAL_PER_CLASS_PER_LANG:
                raise ValueError(f"Insufficient samples for val {lang}-{label}: {len(sub)} < {VAL_PER_CLASS_PER_LANG}")
            sampled = sub.sample(n=VAL_PER_CLASS_PER_LANG, random_state=RANDOM_SEED)
            val_sampled_rows.append(sampled)

    val_df = pd.concat(val_sampled_rows, ignore_index=True)
    val_df = val_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    val_hashes = set(val_df["text_hash"])
    print(f"  Constructed validation split: {len(val_df):,} rows. Unique hashes: {len(val_hashes):,}")

    # =========================================================================
    # Step 3: Process TEST Split (with Leakage Purge from Train AND Val)
    # =========================================================================
    print("\n[4] Building TEST split (target: 1,500 per class per language)...")
    initial_test_count = len(test_pool)
    # Leakage removal: remove any text in train or validation
    forbidden_hashes = train_hashes.union(val_hashes)
    test_pool = test_pool[~test_pool["text_hash"].isin(forbidden_hashes)].copy()
    test_leakage_removed = initial_test_count - len(test_pool)
    print(f"  Removed {test_leakage_removed:,} cross-split leakage rows from test (found in train/val).")
    audit_ledger["leakage_removed"]["train_test_or_val_test_overlap"] = int(test_leakage_removed)

    # Resolve within-split duplicates
    test_pool = test_pool.drop_duplicates(subset=["text_hash", "label"], keep="first")
    test_conflicts = test_pool[test_pool.duplicated(subset=["text_hash"], keep=False)]["text_hash"].unique()
    test_pool = test_pool[~test_pool["text_hash"].isin(test_conflicts)].copy()

    # Stratified deterministic sampling
    test_sampled_rows = []
    for lang in TARGET_LANGUAGES:
        for label in TARGET_LABELS:
            sub = test_pool[(test_pool["language"] == lang) & (test_pool["label"] == label)]
            if len(sub) < TEST_PER_CLASS_PER_LANG:
                raise ValueError(f"Insufficient samples for test {lang}-{label}: {len(sub)} < {TEST_PER_CLASS_PER_LANG}")
            sampled = sub.sample(n=TEST_PER_CLASS_PER_LANG, random_state=RANDOM_SEED)
            test_sampled_rows.append(sampled)

    test_df = pd.concat(test_sampled_rows, ignore_index=True)
    test_df = test_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    test_hashes = set(test_df["text_hash"])
    print(f"  Constructed test split: {len(test_df):,} rows. Unique hashes: {len(test_hashes):,}")

    # =========================================================================
    # Step 4: Strict Zero-Leakage Verification
    # =========================================================================
    overlap_train_val = len(train_hashes.intersection(val_hashes))
    overlap_train_test = len(train_hashes.intersection(test_hashes))
    overlap_val_test = len(val_hashes.intersection(test_hashes))

    print("\n" + "=" * 80)
    print("VERIFICATION OF CROSS-SPLIT ZERO LEAKAGE:")
    print(f"  Train & Validation Overlap: {overlap_train_val}")
    print(f"  Train & Test Overlap:       {overlap_train_test}")
    print(f"  Validation & Test Overlap:  {overlap_val_test}")
    print("=" * 80)

    assert overlap_train_val == 0, f"Error: Train-Val overlap detected: {overlap_train_val}"
    assert overlap_train_test == 0, f"Error: Train-Test overlap detected: {overlap_train_test}"
    assert overlap_val_test == 0, f"Error: Val-Test overlap detected: {overlap_val_test}"

    audit_ledger["post_verification"] = {
        "verified_train_val_overlap": overlap_train_val,
        "verified_train_test_overlap": overlap_train_test,
        "verified_val_test_overlap": overlap_val_test,
        "zero_leakage_guaranteed": True,
    }

    # =========================================================================
    # Step 5: Save Splits and Generate Reports
    # =========================================================================
    # Drop internal text_hash before saving
    train_clean = train_df.drop(columns=["text_hash"])
    val_clean = val_df.drop(columns=["text_hash"])
    test_clean = test_df.drop(columns=["text_hash"])

    train_out = SPLITS_DIR / "train.csv"
    val_out = SPLITS_DIR / "validation.csv"
    test_out = SPLITS_DIR / "test.csv"

    train_clean.to_csv(train_out, index=False, encoding="utf-8")
    val_clean.to_csv(val_out, index=False, encoding="utf-8")
    test_clean.to_csv(test_out, index=False, encoding="utf-8")

    print(f"\nSaved clean splits to:\n  {train_out}\n  {val_out}\n  {test_out}")

    # Record detailed distributions in audit ledger
    for sname, sdf in [("train", train_clean), ("validation", val_clean), ("test", test_clean)]:
        audit_ledger["splits"][sname] = {
            "total_rows": len(sdf),
            "language_counts": dict(Counter(sdf["language"])),
            "sentiment_counts": dict(Counter(sdf["label"])),
            "lang_x_sentiment": dict(Counter(zip(sdf["language"], sdf["label"])).items()),
            "top_sources": Counter(sdf["source"]).most_common(5),
            "top_domains": Counter(sdf["domain"]).most_common(5),
        }
        # JSON serializer helper for tuple keys
        audit_ledger["splits"][sname]["lang_x_sentiment"] = {
            f"{k[0]}_{k[1]}": v for k, v in audit_ledger["splits"][sname]["lang_x_sentiment"].items()
        }

    ledger_path = DATASET_DIR / "extraction_audit_ledger.json"
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(audit_ledger, f, indent=2, ensure_ascii=False)
    print(f"Saved audit ledger: {ledger_path}")

    # Save summary CSV
    summary_records = []
    for sname, sdf in [("train", train_clean), ("validation", val_clean), ("test", test_clean)]:
        rec = {
            "split": sname,
            "total_rows": len(sdf),
            "en_samples": (sdf["language"] == "en").sum(),
            "zh_samples": (sdf["language"] == "zh").sum(),
            "ar_samples": (sdf["language"] == "ar").sum(),
            "positive_samples": (sdf["label"] == "positive").sum(),
            "neutral_samples": (sdf["label"] == "neutral").sum(),
            "negative_samples": (sdf["label"] == "negative").sum(),
        }
        summary_records.append(rec)

    summary_df = pd.DataFrame(summary_records)
    summary_path = DATASET_DIR / "extraction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved extraction summary: {summary_path}")

    print("\nDataset extraction, balancing, and zero-leakage construction COMPLETE!")


if __name__ == "__main__":
    build_clean_splits()
