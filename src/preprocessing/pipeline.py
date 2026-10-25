from dataclasses import dataclass
from typing import Literal
import re

from .text_cleaner import (
    unescape_html,
    remove_html_tags,
    remove_urls,
    remove_mentions,
    unpack_hashtags,
    strip_hashtags,
    reduce_repeated_characters,
    clean_control_characters,
    standardize_whitespace,
    remove_emojis,
)
from .arabic_normalizer import normalize_arabic_text
from .chinese_normalizer import normalize_chinese_text
from .english_normalizer import normalize_english_text


StrategyType = Literal["minimal", "standard", "aggressive"]

# Regex to strip all punctuation in aggressive mode
PUNCTUATION_PATTERN = re.compile(r"[^\w\s\u4e00-\u9fff\u0600-\u06ff]", re.UNICODE)


@dataclass
class PreprocessConfig:
    strategy: StrategyType = "standard"
    preserve_emojis: bool = True
    preserve_punctuation: bool = True
    reduce_repeats: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    expand_contractions: bool = True
    lowercase_english: bool = True


def get_default_config(strategy: StrategyType) -> PreprocessConfig:
    """Return the predefined configuration for a given strategy."""
    if strategy == "minimal":
        return PreprocessConfig(
            strategy="minimal",
            preserve_emojis=True,
            preserve_punctuation=True,
            reduce_repeats=False,
            remove_urls=False,
            remove_mentions=False,
            expand_contractions=False,
            lowercase_english=False,
        )
    elif strategy == "standard":
        return PreprocessConfig(
            strategy="standard",
            preserve_emojis=True,
            preserve_punctuation=True,
            reduce_repeats=True,
            remove_urls=True,
            remove_mentions=True,
            expand_contractions=True,
            lowercase_english=True,
        )
    elif strategy == "aggressive":
        return PreprocessConfig(
            strategy="aggressive",
            preserve_emojis=False,
            preserve_punctuation=False,
            reduce_repeats=True,
            remove_urls=True,
            remove_mentions=True,
            expand_contractions=True,
            lowercase_english=True,
        )
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Choose from 'minimal', 'standard', 'aggressive'.")


def preprocess_text(
    text: str,
    language: str,
    strategy: StrategyType = "standard",
    config: PreprocessConfig = None,
) -> str:
    """
    Multilingual preprocessing function supporting English ('en'), Chinese ('zh'), and Arabic ('ar').

    Args:
        text: Raw input text.
        language: Language code ('en', 'zh', 'ar').
        strategy: 'minimal', 'standard', or 'aggressive'.
        config: Optional custom PreprocessConfig overriding default strategy presets.

    Returns:
        Cleaned, normalized text.
    """
    if not text or not str(text).strip():
        return ""

    text = str(text)

    if config is None:
        cfg = get_default_config(strategy)
    else:
        cfg = config

    lang = str(language).strip().lower()

    # Step 1: Base decoding & control character cleanup
    text = unescape_html(text)
    text = clean_control_characters(text)

    # Step 2: Minimal strategy branch
    if cfg.strategy == "minimal":
        return standardize_whitespace(text)

    # Step 3: Standard & Aggressive Common Cleanups
    text = remove_html_tags(text)

    if cfg.remove_urls:
        text = remove_urls(text)

    if cfg.remove_mentions:
        text = remove_mentions(text)

    if cfg.strategy == "aggressive":
        text = strip_hashtags(text)
    else:
        text = unpack_hashtags(text)

    if cfg.reduce_repeats:
        text = reduce_repeated_characters(text, max_repeats=2)

    # Step 4: Language-specific normalization
    if lang == "ar":
        if cfg.strategy == "aggressive":
            text = normalize_arabic_text(
                text,
                strip_diacritics=True,
                strip_tatweel=True,
                norm_alef=True,
                norm_teh=True,
                norm_maqsura=True,
            )
        else:
            text = normalize_arabic_text(
                text,
                strip_diacritics=True,
                strip_tatweel=True,
                norm_alef=True,
                norm_teh=False,
                norm_maqsura=False,
            )
    elif lang == "zh":
        text = normalize_chinese_text(
            text,
            standardize_punct=(cfg.strategy != "aggressive"),
            clean_spacing=True,
            nfkc_norm=True,
        )
    elif lang == "en":
        text = normalize_english_text(
            text,
            expand_contr=cfg.expand_contractions,
            lowercase=cfg.lowercase_english,
        )

    # Step 5: Emoji handling
    if not cfg.preserve_emojis:
        text = remove_emojis(text)

    # Step 6: Punctuation handling
    if not cfg.preserve_punctuation:
        text = PUNCTUATION_PATTERN.sub(" ", text)

    # Step 7: Final whitespace cleanup
    return standardize_whitespace(text)
