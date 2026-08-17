from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from faturaiz.candidates import generate_candidate_pairs
from faturaiz.features import build_pair_features
from faturaiz.model import (
    ModelBundle,
    evaluate,
    feature_contributions,
    load_verified_bundle,
    save_bundle,
    select_threshold,
)


def test_threshold_respects_precision_floor() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    threshold = select_threshold(probabilities, labels, min_precision=1.0)
    predictions = probabilities >= threshold
    assert predictions.tolist() == [False, False, True, True]


def test_evaluation_returns_named_metrics() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    result = evaluate(labels, (probabilities > 0.5).astype(int), probabilities)
    assert result.f1 == 1.0
    assert result.confusion_matrix == [[2, 0], [0, 2]]


def test_artifact_digest_blocks_tampering(tmp_path: Path, trained_bundle: ModelBundle) -> None:
    artifact = tmp_path / "model.joblib"
    manifest = tmp_path / "manifest.json"
    save_bundle(trained_bundle, artifact, manifest)
    loaded = load_verified_bundle(artifact, manifest)
    assert loaded.threshold == trained_bundle.threshold
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        load_verified_bundle(artifact, manifest)


def test_explanation_returns_ranked_contributions(
    small_batch: pd.DataFrame,
    settings,
    trained_bundle: ModelBundle,  # type: ignore[no-untyped-def]
) -> None:
    pairs = generate_candidate_pairs(small_batch, settings.candidates)
    features = build_pair_features(small_batch, pairs)
    explanation = feature_contributions(trained_bundle, features.iloc[0])
    assert len(explanation) == 4
    assert {"feature", "value", "contribution"} == set(explanation[0])
