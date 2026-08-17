from pathlib import Path

import pytest

from faturaiz.model import ModelBundle, save_bundle
from faturaiz.reproducibility import assert_reproducible


def _write_snapshot(root: Path, bundle: ModelBundle, *, digest_label: str) -> None:
    save_bundle(bundle, root / "models/model.joblib", root / "models/manifest.json")
    (root / "reports").mkdir(parents=True)
    (root / "reports/evaluation.json").write_text(
        '{"f1": 0.9, "artifact_sha256": "' + digest_label + '"}\n', encoding="utf-8"
    )
    (root / "reports/data_quality.json").write_text('{"rows": 10}\n', encoding="utf-8")
    (root / "data/samples").mkdir(parents=True)
    (root / "data/samples/synthetic_invoices.csv").write_text(
        "invoice_id,amount\nINV-1,100.0\n", encoding="utf-8"
    )


def test_reproducibility_ignores_only_platform_dependent_artifact_bytes(
    tmp_path: Path, trained_bundle: ModelBundle
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_snapshot(reference, trained_bundle, digest_label="reference")
    _write_snapshot(candidate, trained_bundle, digest_label="candidate")
    assert_reproducible(reference, candidate)


def test_reproducibility_rejects_changed_threshold(
    tmp_path: Path, trained_bundle: ModelBundle
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    changed = ModelBundle(
        pipeline=trained_bundle.pipeline,
        threshold=trained_bundle.threshold + 0.01,
        feature_names=trained_bundle.feature_names,
    )
    _write_snapshot(reference, trained_bundle, digest_label="reference")
    _write_snapshot(candidate, changed, digest_label="candidate")
    with pytest.raises(AssertionError):
        assert_reproducible(reference, candidate)
