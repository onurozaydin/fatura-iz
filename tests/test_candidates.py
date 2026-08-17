import pandas as pd
import pytest

from faturaiz.candidates import (
    candidate_recall,
    generate_candidate_pairs,
    temporal_group_split,
    validate_invoice_frame,
)


def test_temporal_split_keeps_groups_disjoint(invoices: pd.DataFrame) -> None:
    splits = temporal_group_split(invoices)
    group_sets = [set(frame["group_id"]) for frame in splits.values()]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    assert sum(len(frame) for frame in splits.values()) == len(invoices)


def test_candidate_generation_retains_labelled_duplicate(small_batch, settings) -> None:  # type: ignore[no-untyped-def]
    pairs = generate_candidate_pairs(small_batch, settings.candidates)
    assert int(pairs["label"].sum()) == 1
    assert candidate_recall(small_batch, pairs) == 1.0


def test_schema_validation_rejects_duplicate_ids(small_batch: pd.DataFrame) -> None:
    bad = pd.concat([small_batch, small_batch.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        validate_invoice_frame(bad)


def test_schema_validation_rejects_missing_column(small_batch: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_invoice_frame(small_batch.drop(columns=["amount"]))
