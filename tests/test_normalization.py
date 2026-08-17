from faturaiz.normalization import (
    ascii_upper,
    digit_signature,
    normalize_identifier,
    safe_reference,
)


def test_turkish_text_normalizes_deterministically() -> None:
    assert ascii_upper("İşlem-çağrı") == "ISLEM-CAGRI"  # noqa: RUF001
    assert normalize_identifier(" inv-100 / A ") == "INV100A"


def test_digit_signature_extracts_numeric_segments() -> None:
    assert digit_signature("INV-20-A-004") == "20004"


def test_safe_reference_is_stable_and_pseudonymous() -> None:
    first = safe_reference("SUP-SECRET", "pepper")
    assert first == safe_reference("SUP-SECRET", "pepper")
    assert "SECRET" not in first
    assert first != safe_reference("SUP-SECRET", "other")
