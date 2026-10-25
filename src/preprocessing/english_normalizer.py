import re


# English contractions map
CONTRACTIONS_MAP = {
    r"\bcan\'t\b": "cannot",
    r"\bwon\'t\b": "will not",
    r"\bshan\'t\b": "shall not",
    r"\bain\'t\b": "is not",
    r"n\'t\b": " not",
    r"\'re\b": " are",
    r"\'d\b": " would",
    r"\'ll\b": " will",
    r"\'ve\b": " have",
    r"\'m\b": " am",
    r"\blet\'s\b": "let us",
    r"\bthat\'s\b": "that is",
    r"\bthere\'s\b": "there is",
    r"\bwhat\'s\b": "what is",
    r"\bhere\'s\b": "here is",
    r"\bit\'s\b": "it is",
}

CONTRACTION_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in CONTRACTIONS_MAP.items()
]


def expand_contractions(text: str) -> str:
    """
    Expand standard English contractions to preserve sentiment negation (e.g. don't -> do not).
    Handles both ASCII (') and typographical curly apostrophes (’ and ‘).
    """
    if not text:
        return ""
    # Normalize typographical apostrophes to standard ASCII single quote
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    for pattern, replacement in CONTRACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def normalize_english_text(
    text: str,
    expand_contr: bool = True,
    lowercase: bool = True,
) -> str:
    """
    Normalize English text.
    """
    if not text:
        return ""

    if expand_contr:
        text = expand_contractions(text)

    if lowercase:
        text = text.lower()

    return text
