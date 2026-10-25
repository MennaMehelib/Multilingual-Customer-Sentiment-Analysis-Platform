import re


# Arabic Unicode ranges and special characters
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # Tashkeel and Tatweel
TATWEEL_CHAR = "\u0640"

# Normalization patterns
ALEF_PATTERN = re.compile(r"[إأآٱ]")
TEH_MARBUTA_PATTERN = re.compile(r"ة\b")
ALEF_MAQSURA_PATTERN = re.compile(r"ى\b")

# Arabic punctuation
ARABIC_PUNCTUATION_PATTERN = re.compile(r"[،؛؟]")


def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritical marks (Tashkeel: fatha, damma, kasra, sukun, tanwin, shadda)."""
    if not text:
        return ""
    return ARABIC_DIACRITICS.sub("", text)


def remove_tatweel(text: str) -> str:
    """Remove Arabic Tatweel / Kashida character used for letter elongation (e.g. جميـل -> جميل)."""
    if not text:
        return ""
    return text.replace(TATWEEL_CHAR, "")


def normalize_alef(text: str) -> str:
    """Normalize Alef forms (إ, أ, آ, ٱ) to bare Alef (ا)."""
    if not text:
        return ""
    return ALEF_PATTERN.sub("ا", text)


def normalize_teh_marbuta(text: str) -> str:
    """Normalize Teh Marbuta (ة) to Heh (ه) at word end."""
    if not text:
        return ""
    return TEH_MARBUTA_PATTERN.sub("ه", text)


def normalize_alef_maqsura(text: str) -> str:
    """Normalize Alef Maqsura (ى) to Yeh (ي) at word end."""
    if not text:
        return ""
    return ALEF_MAQSURA_PATTERN.sub("ي", text)


def normalize_arabic_text(
    text: str,
    strip_diacritics: bool = True,
    strip_tatweel: bool = True,
    norm_alef: bool = True,
    norm_teh: bool = False,
    norm_maqsura: bool = False,
) -> str:
    """
    Apply configured Arabic text normalizations.
    Preserves sentiment and negation particles (مش, لا, غير, ممتاز, سيء).
    """
    if not text:
        return ""

    if strip_tatweel:
        text = remove_tatweel(text)

    if strip_diacritics:
        text = remove_diacritics(text)

    if norm_alef:
        text = normalize_alef(text)

    if norm_teh:
        text = normalize_teh_marbuta(text)

    if norm_maqsura:
        text = normalize_alef_maqsura(text)

    return text
