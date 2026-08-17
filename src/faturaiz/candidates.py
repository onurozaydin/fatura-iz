"""Scalable candidate-pair generation and leakage-safe temporal splitting."""

from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any, cast

import pandas as pd

from faturaiz.config import CandidateConfig
from faturaiz.normalization import normalize_identifier

REQUIRED_COLUMNS = {
    "invoice_id",
    "vendor_account_id",
    "invoice_number",
    "invoice_date",
    "amount",
    "currency",
    "po_number",
    "bank_fingerprint",
    "description",
}


def validate_invoice_frame(frame: pd.DataFrame, *, require_labels: bool = True) -> None:
    """Fail fast on schema, duplicates, invalid values, and unsafe identifiers."""
    required = REQUIRED_COLUMNS | ({"group_id"} if require_labels else set())
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("invoice data must not be empty")
    if frame["invoice_id"].duplicated().any():
        raise ValueError("invoice_id values must be unique")
    if (pd.to_numeric(frame["amount"], errors="coerce") <= 0).any():
        raise ValueError("amount values must be positive numbers")
    if frame[list(required)].isna().any().any():
        raise ValueError("required fields must not contain null values")
    parsed = pd.to_datetime(frame["invoice_date"], errors="coerce")
    if parsed.isna().any():
        raise ValueError("invoice_date must contain valid ISO dates")


def temporal_group_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split whole synthetic invoice groups chronologically into 70/15/15 partitions."""
    validate_invoice_frame(frame)
    dated = frame.assign(_date=pd.to_datetime(frame["invoice_date"]))
    anchors = (
        dated.groupby("group_id", as_index=False).agg(_date=("_date", "min")).sort_values("_date")
    )
    train_end = int(len(anchors) * 0.70)
    validation_end = int(len(anchors) * 0.85)
    group_sets = {
        "train": set(anchors.iloc[:train_end]["group_id"]),
        "validation": set(anchors.iloc[train_end:validation_end]["group_id"]),
        "test": set(anchors.iloc[validation_end:]["group_id"]),
    }
    return {
        name: frame[frame["group_id"].isin(groups)].copy().reset_index(drop=True)
        for name, groups in group_sets.items()
    }


def _blocked_pairs(frame: pd.DataFrame) -> Iterable[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    for columns in (("vendor_account_id", "currency"), ("bank_fingerprint", "currency")):
        for _, group in frame.groupby(list(columns), sort=False):
            indices = list(group.index)
            for position, left in enumerate(indices):
                for right in indices[position + 1 :]:
                    pair = (min(left, right), max(left, right))
                    if pair not in seen:
                        seen.add(pair)
                        yield pair


def generate_candidate_pairs(frame: pd.DataFrame, config: CandidateConfig) -> pd.DataFrame:
    """Create bounded candidate pairs using operational blocking and cheap prefilters."""
    validate_invoice_frame(frame, require_labels="group_id" in frame.columns)
    working = frame.copy()
    working["_date"] = pd.to_datetime(working["invoice_date"])
    working["_invoice_norm"] = working["invoice_number"].map(normalize_identifier)
    pairs: list[dict[str, object]] = []

    for left_index, right_index in _blocked_pairs(working):
        left = cast(dict[str, Any], working.loc[left_index].to_dict())
        right = cast(dict[str, Any], working.loc[right_index].to_dict())
        date_gap = abs((left["_date"] - right["_date"]).days)
        if date_gap > config.max_date_gap_days:
            continue
        maximum_amount = max(float(left["amount"]), float(right["amount"]), 0.01)
        relative_gap = abs(float(left["amount"]) - float(right["amount"])) / maximum_amount
        invoice_similarity = SequenceMatcher(
            None, str(left["_invoice_norm"]), str(right["_invoice_norm"])
        ).ratio()
        if (
            relative_gap > config.max_amount_relative_gap
            and invoice_similarity < config.min_invoice_similarity
        ):
            continue
        row: dict[str, object] = {
            "left_index": left_index,
            "right_index": right_index,
            "left_invoice_id": left["invoice_id"],
            "right_invoice_id": right["invoice_id"],
        }
        if "group_id" in working.columns:
            row["label"] = int(str(left["group_id"]) == str(right["group_id"]))
        pairs.append(row)
    return pd.DataFrame(pairs)


def candidate_recall(frame: pd.DataFrame, pairs: pd.DataFrame) -> float:
    """Measure whether blocking retained all labelled duplicate pairs."""
    positive_groups = frame.groupby("group_id").filter(lambda group: len(group) > 1)
    total = positive_groups["group_id"].nunique()
    if total == 0:
        return 1.0
    found = int(pairs.loc[pairs["label"] == 1].shape[0])
    return min(1.0, found / total)
