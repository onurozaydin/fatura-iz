import pandas as pd
import pytest

from faturaiz.synthetic import generate_invoices


def test_generator_is_deterministic_and_marks_disclosure_fields() -> None:
    first = generate_invoices(seed=9, base_invoices=120, duplicate_rate=0.2)
    second = generate_invoices(seed=9, base_invoices=120, duplicate_rate=0.2)
    pd.testing.assert_frame_equal(first, second)
    assert first["is_duplicate_record"].sum() > 0
    assert set(first["duplicate_type"]) <= {
        "original",
        "exact",
        "format_shift",
        "amount_shift",
        "weak_resubmission",
    }


def test_generator_rejects_unsafe_parameters() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        generate_invoices(seed=1, base_invoices=10)
    with pytest.raises(ValueError, match="between"):
        generate_invoices(seed=1, base_invoices=100, duplicate_rate=0)
