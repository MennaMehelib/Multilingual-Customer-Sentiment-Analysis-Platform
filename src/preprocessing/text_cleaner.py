import html
import re
import unicodedata


# Common regex patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MENTION_PATTERN = re.compile(r"@[\w_]+", re.UNICODE)
HASHTAG_PATTERN = re.compile(r"#([\w_]+)", re.UNICODE)
REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{2,}", re.DOTALL)
MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def unescape_html(text: str) -> str:
    """Unescape HTML entities like &amp;, &quot;, &#39;."""
    if not text:
        return ""
    return html.unescape(text)


def remove_html_tags(text: str) -> str:
    """Remove raw HTML markup like <br>, <p>, </div>."""
    if not text:
        return ""
    return HTML_TAG_PATTERN.sub(" ", text)


def remove_urls(text: str, placeholder: str = "") -> str:
    """Remove URLs or replace with an optional token."""
    if not text:
        return ""
    return URL_PATTERN.sub(placeholder, text)


def remove_mentions(text: str, placeholder: str = "") -> str:
    """Remove user mentions like @username."""
    if not text:
        return ""
    return MENTION_PATTERN.sub(placeholder, text)


def unpack_hashtags(text: str) -> str:
    """Convert #greatproduct to greatproduct to preserve word sentiment."""
    if not text:
        return ""
    return HASHTAG_PATTERN.sub(r"\1", text)


def strip_hashtags(text: str) -> str:
    """Completely remove hashtags and their text."""
    if not text:
        return ""
    return re.sub(r"#[\w_]+", "", text, flags=re.UNICODE)


def reduce_repeated_characters(text: str, max_repeats: int = 2) -> str:
    """
    Reduce exaggerated character repetition like 'soooo' -> 'soo' or '!!!!!' -> '!!'.
    Preserves sentiment emphasis while reducing vocabulary explosion.
    """
    if not text:
        return ""
    return REPEATED_CHARS_PATTERN.sub(r"\1" * max_repeats, text)


def clean_control_characters(text: str) -> str:
    """Strip non-printable Unicode control characters."""
    if not text:
        return ""
    return CONTROL_CHARS_PATTERN.sub(" ", text)


def standardize_whitespace(text: str) -> str:
    """Normalize tabs, newlines, and multiple spaces into a single space."""
    if not text:
        return ""
    return MULTIPLE_SPACES_PATTERN.sub(" ", text).strip()


def remove_emojis(text: str) -> str:
    """
    Remove Unicode emoji characters (used in aggressive normalization).
    For standard and minimal strategies, emojis are preserved for sentiment value.
    """
    if not text:
        return ""
    # Matches common Emoji ranges in Unicode
    emoji_pattern = re.compile(
        r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(" ", text)
