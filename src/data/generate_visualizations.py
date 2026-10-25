"""
Multilingual Sentiment Analysis Visualization Suite.
Generates publication-quality charts for dataset distribution, balance,
domain diversity, leakage verification, and before/after preprocessing comparisons.
Saves all charts under docs/visualizations/ and data/processed/visualizations/.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"
DATASET_DIR = BASE_DIR / "data" / "processed" / "dataset"
PREPROC_DIR = BASE_DIR / "data" / "processed" / "preprocessing"

DOCS_VIZ_DIR = BASE_DIR / "docs" / "visualizations"
DATA_VIZ_DIR = BASE_DIR / "data" / "processed" / "visualizations"

DOCS_VIZ_DIR.mkdir(parents=True, exist_ok=True)
DATA_VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Styling palette
PALETTE = {
    "positive": "#2ecc71",
    "neutral": "#f39c12",
    "negative": "#e74c3c",
    "en": "#3498db",
    "zh": "#9b59b6",
    "ar": "#1abc9c",
}

sns.set_theme(style="whitegrid", font="sans-serif")


def save_plot(fig, filename: str):
    """Save figure to both docs and data/processed artifact folders."""
    path1 = DOCS_VIZ_DIR / filename
    path2 = DATA_VIZ_DIR / filename
    fig.savefig(path1, dpi=300, bbox_inches="tight")
    fig.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_language_distribution(train_df, val_df, test_df):
    """Plot distribution of languages across train, val, and test splits."""
    fig, ax = plt.subplots(figsize=(9, 5))

    data = []
    for split_name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        counts = df["language"].value_counts()
        for lang in ["en", "zh", "ar"]:
            data.append({
                "Split": split_name,
                "Language": lang.upper(),
                "Count": counts.get(lang, 0),
            })
    plot_df = pd.DataFrame(data)

    sns.barplot(data=plot_df, x="Split", y="Count", hue="Language", palette=[PALETTE["en"], PALETTE["zh"], PALETTE["ar"]], ax=ax)
    ax.set_title("Dataset Distribution by Language Across Benchmark Splits", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Records", fontsize=12)
    ax.set_xlabel("Split", fontsize=12)
    for container in ax.containers:
        ax.bar_label(container, fmt="%,d", padding=3, fontsize=9)

    save_plot(fig, "01_language_distribution.png")


def plot_sentiment_distribution(train_df):
    """Plot class balance across sentiment categories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Overall class counts
    counts = train_df["label"].value_counts()
    labels = ["positive", "neutral", "negative"]
    colors = [PALETTE[l] for l in labels]

    ax1.pie(
        [counts.get(l, 0) for l in labels],
        labels=[l.capitalize() for l in labels],
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        textprops={"fontsize": 11, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax1.set_title("Overall Training Sentiment Class Balance (1:1:1)", fontsize=13, fontweight="bold")

    # Grouped by language
    grouped = pd.crosstab(train_df["language"], train_df["label"])[labels]
    grouped.index = [i.upper() for i in grouped.index]
    grouped.plot(kind="bar", stacked=False, color=colors, ax=ax2, edgecolor="black", linewidth=0.5)
    ax2.set_title("Sentiment Counts per Language (Exactly 10k per Class)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Record Count", fontsize=12)
    ax2.set_xlabel("Language", fontsize=12)
    ax2.legend(title="Sentiment", loc="upper right")
    ax2.tick_params(axis="x", rotation=0)
    for container in ax2.containers:
        ax2.bar_label(container, fmt="%,d", padding=3, fontsize=8)

    plt.tight_layout()
    save_plot(fig, "02_sentiment_distribution.png")


def plot_text_length_distribution(train_df):
    """Plot text length distributions across languages."""
    fig, ax = plt.subplots(figsize=(10, 5))

    train_df = train_df.copy()
    train_df["char_length"] = train_df["text"].str.len()

    # Clip for clean visualization (99th percentile)
    clip_val = train_df["char_length"].quantile(0.98)
    sub = train_df[train_df["char_length"] <= clip_val]

    sns.kdeplot(
        data=sub,
        x="char_length",
        hue="language",
        common_norm=False,
        fill=True,
        alpha=0.3,
        palette=[PALETTE["en"], PALETTE["zh"], PALETTE["ar"]],
        ax=ax,
    )
    ax.set_title("Character Length Distribution by Language", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Character Length (Capped at 98th Percentile)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(title="Language", labels=["AR (Arabic)", "ZH (Chinese)", "EN (English)"])

    save_plot(fig, "03_text_length_distribution.png")


def plot_domain_distribution(train_df):
    """Plot domain representation across the training split."""
    fig, ax = plt.subplots(figsize=(11, 6))

    top_domains = train_df["domain"].value_counts().head(10).reset_index()
    top_domains.columns = ["domain", "count"]

    sns.barplot(data=top_domains, y="domain", x="count", palette="viridis", ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_title("Top Customer Feedback Domains in Standardized Training Benchmark", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Reviews", fontsize=12)
    ax.set_ylabel("Domain Category", fontsize=12)
    for container in ax.containers:
        ax.bar_label(container, fmt="%,d", padding=4, fontsize=9)

    save_plot(fig, "04_domain_diversity.png")


def plot_leakage_and_audit(ledger):
    """Visualize duplicate filtering, cross-split leakage purge, and zero-leakage verification."""
    fig, ax = plt.subplots(figsize=(10, 5))

    metrics = [
        "Raw Streamed Rows",
        "Filtered Empty/Short",
        "Leakage Purged (Train-Val)",
        "Leakage Purged (Train-Test)",
        "Final Clean Benchmark",
    ]

    raw_tot = sum(ledger["raw_counts"].values())
    inv_tot = sum(ledger["invalid_empty_removed"].values())
    leak_val = ledger["leakage_removed"].get("train_val_overlap", 0)
    leak_test = ledger["leakage_removed"].get("train_test_or_val_test_overlap", 0)
    final_tot = sum(s["total_rows"] for s in ledger["splits"].values())

    values = [raw_tot, inv_tot, leak_val, leak_test, final_tot]
    colors = ["#7f8c8d", "#e74c3c", "#d35400", "#e67e22", "#27ae60"]

    bars = ax.barh(metrics, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_title("Dataset Cleaning, Leakage Purge & Benchmark Construction Audit", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Records (Log Scale)", fontsize=12)
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.1, bar.get_y() + bar.get_height() / 2, f"{int(width):,}", va="center", fontsize=9, fontweight="bold")

    save_plot(fig, "05_leakage_and_audit_funnel.png")


def plot_preprocessing_comparison(comp_csv_path: Path):
    """Plot comparison of Minimal, Standard, and Aggressive preprocessing strategies."""
    if not comp_csv_path.exists():
        print("  Comparison CSV not found. Skipping preprocessing comparison chart.")
        return

    df = pd.read_csv(comp_csv_path)
    df["Affected_Pct_Num"] = df["Affected Rows (%)"].str.rstrip("%").astype(float)
    df["Reduction_Pct_Num"] = df["Length Reduction (%)"].str.rstrip("%").astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Affected rows %
    sns.barplot(data=df, x="Language", y="Affected_Pct_Num", hue="Strategy", palette="Blues_r", ax=ax1, edgecolor="black", linewidth=0.5)
    ax1.set_title("Percentage of Reviews Modified by Preprocessing Strategy", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Affected Rows (%)", fontsize=12)
    ax1.set_xlabel("Language", fontsize=12)
    ax1.set_ylim(0, 115)
    for container in ax1.containers:
        ax1.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8)

    # Length reduction %
    sns.barplot(data=df, x="Language", y="Reduction_Pct_Num", hue="Strategy", palette="Oranges_r", ax=ax2, edgecolor="black", linewidth=0.5)
    ax2.set_title("Average Text Length Reduction (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Length Reduction (%)", fontsize=12)
    ax2.set_xlabel("Language", fontsize=12)
    for container in ax2.containers:
        ax2.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8)

    plt.tight_layout()
    save_plot(fig, "06_preprocessing_strategy_comparison.png")


def plot_before_after_preprocessing(train_df: pd.DataFrame):
    """Plot direct before vs after character length distributions for each language."""
    if "cleaned_text" not in train_df.columns:
        print("  'cleaned_text' column not found. Skipping before/after distribution plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    languages = [("en", "English"), ("zh", "Chinese"), ("ar", "Arabic")]

    for idx, (lang_code, lang_name) in enumerate(languages):
        ax = axes[idx]
        sub = train_df[train_df["language"] == lang_code]
        raw_lens = sub["text"].astype(str).str.len()
        clean_lens = sub["cleaned_text"].astype(str).str.len()

        # Clip at 98th percentile for clean display
        clip_val = raw_lens.quantile(0.98)
        mask = raw_lens <= clip_val

        plot_data = pd.DataFrame({
            "Length": list(raw_lens[mask]) + list(clean_lens[mask]),
            "Stage": ["Before (Raw)"] * int(mask.sum()) + ["After (Standard)"] * int(mask.sum()),
        })

        sns.kdeplot(data=plot_data, x="Length", hue="Stage", common_norm=False, fill=True, alpha=0.3, ax=ax, palette=["#e74c3c", "#2ecc71"])
        ax.set_title(f"{lang_name} ({lang_code.upper()})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Character Count", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)

    fig.suptitle("Text Length Distribution: Before vs. After Standard Preprocessing", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    save_plot(fig, "07_before_after_text_distributions.png")


def generate_all_visualizations():
    print("=" * 80)
    print("GENERATING PERSON 1 VISUALIZATION SUITE")
    print("=" * 80)

    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "validation.csv"
    test_path = SPLITS_DIR / "test.csv"
    ledger_path = DATASET_DIR / "extraction_audit_ledger.json"
    comp_csv_path = PREPROC_DIR / "preprocessing_comparison.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError("Splits not found. Please extract splits first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    plot_language_distribution(train_df, val_df, test_df)
    plot_sentiment_distribution(train_df)
    plot_text_length_distribution(train_df)
    plot_domain_distribution(train_df)
    plot_leakage_and_audit(ledger)
    plot_preprocessing_comparison(comp_csv_path)
    plot_before_after_preprocessing(train_df)

    print("\nVisualization generation complete! Saved to docs/visualizations/ and data/processed/visualizations/.")


if __name__ == "__main__":
    generate_all_visualizations()
