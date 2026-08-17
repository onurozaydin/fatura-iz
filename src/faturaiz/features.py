"""Auditable pairwise feature engineering."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, cast

import numpy as np
import pandas as pd

from faturaiz.normalization import digit_signature, normalize_identifier

FEATURE_NAMES = (
    "vendor_same",
    "bank_same",
    "currency_same",
    "invoice_similarity",
    "digit_similarity",
    "amount_relative_similarity",
    "date_proximity",
    "po_same",
    "description_similarity",
)


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def build_pair_features(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Build numeric, bounded features for each candidate pair."""
    rows: list[dict[str, float]] = []
    for pair in pairs.itertuples(index=False):
        left_index = int(cast(Any, pair.left_index))
        right_index = int(cast(Any, pair.right_index))
        left = cast(dict[str, Any], frame.loc[left_index].to_dict())
        right = cast(dict[str, Any], frame.loc[right_index].to_dict())
        left_invoice = normalize_identifier(str(left["invoice_number"]))
        right_invoice = normalize_identifier(str(right["invoice_number"]))
        left_digits = digit_signature(left_invoice)
        right_digits = digit_signature(right_invoice)
        amount_max = max(float(left["amount"]), float(right["amount"]), 0.01)
        amount_gap = abs(float(left["amount"]) - float(right["amount"])) / amount_max
        date_gap = abs(
            (pd.Timestamp(left["invoice_date"]) - pd.Timestamp(right["invoice_date"])).days
        )
        po_left = str(left["po_number"])
        po_right = str(right["po_number"])
        rows.append(
            {
                "vendor_same": float(left["vendor_account_id"] == right["vendor_account_id"]),
                "bank_same": float(left["bank_fingerprint"] == right["bank_fingerprint"]),
                "currency_same": float(left["currency"] == right["currency"]),
                "invoice_similarity": _similarity(left_invoice, right_invoice),
                "digit_similarity": _similarity(left_digits, right_digits),
                "amount_relative_similarity": max(0.0, 1.0 - amount_gap),
                "date_proximity": float(np.exp(-date_gap / 30.0)),
                "po_same": float(bool(po_left) and po_left == po_right),
                "description_similarity": _similarity(
                    normalize_identifier(str(left["description"])),
                    normalize_identifier(str(right["description"])),
                ),
            }
        )
    return pd.DataFrame(rows, columns=list(FEATURE_NAMES))


def exact_baseline(features: pd.DataFrame) -> np.ndarray:
    """Conservative baseline: same vendor, invoice number, and amount."""
    return (
        (
            (features["vendor_same"] == 1)
            & (features["invoice_similarity"] == 1)
            & (features["amount_relative_similarity"] >= 0.999999)
        )
        .astype(int)
        .to_numpy()
    )
