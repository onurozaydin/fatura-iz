"""Human-in-the-loop decision policy above the probabilistic model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Decision:
    state: str
    action: str


def decide(probability: float, threshold: float, features: pd.Series) -> Decision:
    """Map evidence to PASS, REVIEW, or HOLD; never claim a confirmed duplicate."""
    exact_signature = (
        features["vendor_same"] == 1
        and features["invoice_similarity"] == 1
        and features["amount_relative_similarity"] >= 0.999999
    )
    if exact_signature and probability >= threshold:
        return Decision("HOLD", "Pause payment and request two-record verification.")
    if probability >= threshold:
        return Decision("REVIEW", "Queue for accounts-payable analyst review before payment.")
    return Decision("PASS", "No candidate exceeded the review threshold; retain normal controls.")
