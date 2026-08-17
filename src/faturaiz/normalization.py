"""Deterministic normalization helpers with no external network dependency."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")
_DIGITS = re.compile(r"\d+")


def ascii_upper(value: str) -> str:
    """Normalize Unicode text to uppercase ASCII where possible."""
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).upper()


def normalize_identifier(value: str) -> str:
    """Remove punctuation and case differences from invoice identifiers."""
    return _NON_ALNUM.sub("", ascii_upper(value))


def digit_signature(value: str) -> str:
    """Return concatenated numeric segments from an identifier."""
    return "".join(_DIGITS.findall(value))


def safe_reference(value: str, pepper: str) -> str:
    """Create a short pseudonymous reference; never return the raw vendor value."""
    digest = hashlib.sha256(f"{pepper}:{value}".encode()).hexdigest()
    return f"ref_{digest[:12]}"
