from pathlib import Path

import pytest

from faturaiz.config import load_settings


def test_default_config_loads() -> None:
    settings = load_settings()
    assert settings.project.seed == 20260817
    assert settings.service.max_batch_size == 500


def test_invalid_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        """
[project]
seed=1
synthetic_base_invoices=100
synthetic_duplicate_rate=2.0
[candidates]
max_date_gap_days=60
max_amount_relative_gap=0.05
min_invoice_similarity=0.5
[model]
min_validation_precision=0.8
false_positive_review_cost_try=50.0
[service]
max_batch_size=10
artifact_path="a"
manifest_path="b"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate_rate"):
        load_settings(path)
