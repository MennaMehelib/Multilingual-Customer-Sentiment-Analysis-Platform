import re
import unicodedata


# Chinese full-width to half-width punctuation mapping
CHINESE_PUNCT_MAP = {
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "：": ":",
    "；": ";",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "《": "<",
    "》": ">",
    "、": ",",
    "…": "...",
    "—": "-",
    "～": "~",
}

# Regex to detect whitespace between two Chinese characters
CHINESE_CHAR_RANGE = r"[\u4e00-\u9fff]"
SPACES_BETWEEN_CHINESE = re.compile(
    rf"({CHINESE_CHAR_RANGE})\s+({CHINESE_CHAR_RANGE})"
)


def fullwidth_to_halfwidth(text: str) -> str:
    """
    Convert full-width Latin characters and numbers to standard ASCII half-width.
    Uses Unicode NFKC normalization.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def normalize_chinese_punctuation(text: str) -> str:
    """Normalize full-width Chinese punctuation marks to standard equivalents."""
    if not text:
        return ""
    for full_char, half_char in CHINESE_PUNCT_MAP.items():
        text = text.replace(full_char, half_char)
    return text


def remove_spaces_between_chinese_chars(text: str) -> str:
    """
    Remove unnatural whitespace inserted between adjacent Chinese characters
    while preserving spacing between Chinese and Latin words.
    """
    if not text:
        return ""
    prev = None
    while prev != text:
        prev = text
        text = SPACES_BETWEEN_CHINESE.sub(r"\1\2", text)
    return text


def normalize_chinese_text(
    text: str,
    standardize_punct: bool = True,
    clean_spacing: bool = True,
    nfkc_norm: bool = True,
) -> str:
    """
    Normalize Chinese text preserving semantic characters and sentiment punctuation.
    """
    if not text:
        return ""

    if nfkc_norm:
        text = fullwidth_to_halfwidth(text)

    if standardize_punct:
        text = normalize_chinese_punctuation(text)

    if clean_spacing:
        text = remove_spaces_between_chinese_chars(text)

    return text
