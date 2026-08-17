"""Semantic reproducibility checks for trained model artifacts and evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from faturaiz.model import ModelBundle, load_verified_bundle


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _assert_bundle_equivalent(reference: ModelBundle, candidate: ModelBundle) -> None:
    assert reference.threshold == candidate.threshold
    assert reference.feature_names == candidate.feature_names
    assert reference.synthetic_training_data == candidate.synthetic_training_data

    reference_scaler = reference.pipeline.named_steps["scale"]
    candidate_scaler = candidate.pipeline.named_steps["scale"]
    for attribute in ("mean_", "scale_", "var_", "n_samples_seen_"):
        np.testing.assert_allclose(
            np.asarray(getattr(reference_scaler, attribute)),
            np.asarray(getattr(candidate_scaler, attribute)),
            rtol=0.0,
            atol=1e-12,
        )

    reference_classifier = reference.pipeline.named_steps["classifier"]
    candidate_classifier = candidate.pipeline.named_steps["classifier"]
    for attribute in ("classes_", "coef_", "intercept_", "n_iter_"):
        np.testing.assert_allclose(
            np.asarray(getattr(reference_classifier, attribute)),
            np.asarray(getattr(candidate_classifier, attribute)),
            rtol=0.0,
            atol=1e-12,
        )
    for parameter in ("class_weight", "max_iter", "random_state", "solver"):
        assert getattr(reference_classifier, parameter) == getattr(candidate_classifier, parameter)


def assert_reproducible(reference_root: Path, candidate_root: Path) -> None:
    """Verify integrity plus semantic equivalence of two complete training outputs."""
    reference = load_verified_bundle(
        reference_root / "models/model.joblib", reference_root / "models/manifest.json"
    )
    candidate = load_verified_bundle(
        candidate_root / "models/model.joblib", candidate_root / "models/manifest.json"
    )
    _assert_bundle_equivalent(reference, candidate)

    reference_manifest = _read_json(reference_root / "models/manifest.json")
    candidate_manifest = _read_json(candidate_root / "models/manifest.json")
    reference_manifest.pop("sha256")
    candidate_manifest.pop("sha256")
    assert reference_manifest == candidate_manifest

    reference_evaluation = _read_json(reference_root / "reports/evaluation.json")
    candidate_evaluation = _read_json(candidate_root / "reports/evaluation.json")
    reference_evaluation.pop("artifact_sha256")
    candidate_evaluation.pop("artifact_sha256")
    assert reference_evaluation == candidate_evaluation

    for relative_path in (
        Path("reports/data_quality.json"),
        Path("data/samples/synthetic_invoices.csv"),
    ):
        assert (reference_root / relative_path).read_bytes() == (
            candidate_root / relative_path
        ).read_bytes()
