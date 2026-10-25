"""
Multilingual Preprocessing Package for Customer Sentiment Analysis.
Supports English ('en'), Chinese ('zh'), and Arabic ('ar').
"""

from .pipeline import (
    PreprocessConfig,
    preprocess_text,
    get_default_config,
    StrategyType,
)
from .text_cleaner import (
    unescape_html,
    remove_html_tags,
    remove_urls,
    remove_mentions,
    unpack_hashtags,
    strip_hashtags,
    reduce_repeated_characters,
    standardize_whitespace,
    remove_emojis,
)
from .arabic_normalizer import normalize_arabic_text
from .chinese_normalizer import normalize_chinese_text
from .english_normalizer import normalize_english_text

__all__ = [
    "PreprocessConfig",
    "preprocess_text",
    "get_default_config",
    "StrategyType",
    "unescape_html",
    "remove_html_tags",
    "remove_urls",
    "remove_mentions",
    "unpack_hashtags",
    "strip_hashtags",
    "reduce_repeated_characters",
    "standardize_whitespace",
    "remove_emojis",
    "normalize_arabic_text",
    "normalize_chinese_text",
    "normalize_english_text",
]
