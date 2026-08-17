import pandas as pd

from faturaiz.policy import decide


def _features(invoice_similarity: float) -> pd.Series:
    return pd.Series(
        {
            "vendor_same": 1.0,
            "invoice_similarity": invoice_similarity,
            "amount_relative_similarity": 1.0,
        }
    )


def test_policy_holds_exact_high_risk_pair() -> None:
    assert decide(0.9, 0.5, _features(1.0)).state == "HOLD"


def test_policy_reviews_near_match_and_passes_low_score() -> None:
    assert decide(0.8, 0.5, _features(0.8)).state == "REVIEW"
    assert decide(0.2, 0.5, _features(0.8)).state == "PASS"
