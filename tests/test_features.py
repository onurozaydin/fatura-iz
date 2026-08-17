import pandas as pd

from faturaiz.candidates import generate_candidate_pairs
from faturaiz.features import FEATURE_NAMES, build_pair_features, exact_baseline


def test_features_are_bounded_and_ordered(small_batch: pd.DataFrame, settings) -> None:  # type: ignore[no-untyped-def]
    pairs = generate_candidate_pairs(small_batch, settings.candidates)
    features = build_pair_features(small_batch, pairs)
    assert tuple(features.columns) == FEATURE_NAMES
    assert features.ge(0).all().all()
    assert features.le(1).all().all()


def test_exact_baseline_only_flags_exact_signature(small_batch: pd.DataFrame, settings) -> None:  # type: ignore[no-untyped-def]
    pairs = generate_candidate_pairs(small_batch, settings.candidates)
    features = build_pair_features(small_batch, pairs)
    predictions = exact_baseline(features)
    assert predictions.sum() == 1
