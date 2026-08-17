"""Application service: validate, generate candidates, score, explain, and pseudonymize."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from faturaiz.candidates import generate_candidate_pairs
from faturaiz.config import CandidateConfig
from faturaiz.features import build_pair_features
from faturaiz.model import ModelBundle, feature_contributions, predict_probabilities
from faturaiz.normalization import safe_reference
from faturaiz.policy import decide


def triage_batch(
    frame: pd.DataFrame,
    *,
    bundle: ModelBundle,
    candidate_config: CandidateConfig,
    reference_pepper: str,
) -> list[dict[str, Any]]:
    """Return only reviewable candidate pairs; raw vendor identifiers are excluded."""
    pairs = generate_candidate_pairs(frame, candidate_config)
    if pairs.empty:
        return []
    features = build_pair_features(frame, pairs)
    probabilities = predict_probabilities(bundle, features)
    results: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs.itertuples(index=False)):
        probability = float(probabilities[index])
        decision = decide(probability, bundle.threshold, features.iloc[index])
        if decision.state == "PASS":
            continue
        left = frame.loc[int(cast(Any, pair.left_index))]
        results.append(
            {
                "left_invoice_id": pair.left_invoice_id,
                "right_invoice_id": pair.right_invoice_id,
                "vendor_ref": safe_reference(str(left["vendor_account_id"]), reference_pepper),
                "risk_score": round(probability, 6),
                "state": decision.state,
                "action": decision.action,
                "evidence": feature_contributions(bundle, features.iloc[index]),
            }
        )
    return sorted(results, key=lambda row: float(row["risk_score"]), reverse=True)
