"""End-to-end reproducible training and evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from faturaiz.candidates import candidate_recall, generate_candidate_pairs, temporal_group_split
from faturaiz.config import Settings
from faturaiz.features import FEATURE_NAMES, build_pair_features, exact_baseline
from faturaiz.model import (
    ModelBundle,
    evaluate,
    evaluation_dict,
    predict_probabilities,
    save_bundle,
    select_threshold,
    train_model,
)
from faturaiz.synthetic import generate_invoices


def _pair_exposure(frame: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [
            max(
                float(
                    cast(
                        Any,
                        frame.loc[int(cast(Any, pair.left_index)), "amount"],
                    )
                ),
                0.0,
            )
            for pair in pairs.itertuples()
        ],
        dtype=float,
    )


def _operational_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    exposure: np.ndarray,
    review_cost: float,
) -> dict[str, float]:
    positive_exposure = float(exposure[labels == 1].sum())
    captured = float(exposure[(labels == 1) & (predictions == 1)].sum())
    false_negative_cost = float(exposure[(labels == 1) & (predictions == 0)].sum())
    false_positive_cost = float(((labels == 0) & (predictions == 1)).sum() * review_cost)
    return {
        "review_rate": float(predictions.mean()),
        "duplicate_value_capture": captured / positive_exposure if positive_exposure else 1.0,
        "expected_control_cost_try": false_negative_cost + false_positive_cost,
    }


def _error_analysis(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    type_results: dict[str, list[int]] = {}
    for index, pair in enumerate(pairs.itertuples(index=False)):
        if labels[index] != 1:
            continue
        left = frame.loc[int(cast(Any, pair.left_index))]
        right = frame.loc[int(cast(Any, pair.right_index))]
        duplicate_type = str(
            right["duplicate_type"]
            if bool(right["is_duplicate_record"])
            else left["duplicate_type"]
        )
        type_results.setdefault(duplicate_type, []).append(int(predictions[index]))
    return {
        "false_positive_pairs": int(((labels == 0) & (predictions == 1)).sum()),
        "false_negative_pairs": int(((labels == 1) & (predictions == 0)).sum()),
        "recall_by_duplicate_type": {
            name: float(np.mean(values)) for name, values in sorted(type_results.items())
        },
        "support_by_duplicate_type": {
            name: len(values) for name, values in sorted(type_results.items())
        },
    }


def run_training(settings: Settings, output_dir: Path = Path(".")) -> dict[str, Any]:
    """Generate synthetic data, split, train, freeze threshold, test, and persist evidence."""
    invoices = generate_invoices(
        seed=settings.project.seed,
        base_invoices=settings.project.synthetic_base_invoices,
        duplicate_rate=settings.project.synthetic_duplicate_rate,
    )
    splits = temporal_group_split(invoices)
    pair_sets: dict[str, pd.DataFrame] = {}
    feature_sets: dict[str, pd.DataFrame] = {}
    labels: dict[str, np.ndarray] = {}
    recalls: dict[str, float] = {}
    for name, frame in splits.items():
        pairs = generate_candidate_pairs(frame, settings.candidates)
        pair_sets[name] = pairs
        feature_sets[name] = build_pair_features(frame, pairs)
        labels[name] = pairs["label"].to_numpy(dtype=int)
        recalls[name] = candidate_recall(frame, pairs)

    pipeline = train_model(feature_sets["train"], labels["train"], seed=settings.project.seed)
    temporary_bundle = ModelBundle(pipeline, 0.5, FEATURE_NAMES)
    validation_probabilities = predict_probabilities(temporary_bundle, feature_sets["validation"])
    threshold = select_threshold(
        validation_probabilities,
        labels["validation"],
        min_precision=settings.model.min_validation_precision,
    )
    bundle = ModelBundle(pipeline, threshold, FEATURE_NAMES)
    test_probabilities = predict_probabilities(bundle, feature_sets["test"])
    model_predictions = (test_probabilities >= threshold).astype(int)
    baseline_predictions = exact_baseline(feature_sets["test"])
    baseline_probabilities = baseline_predictions.astype(float)
    model_evaluation = evaluate(labels["test"], model_predictions, test_probabilities)
    baseline_evaluation = evaluate(labels["test"], baseline_predictions, baseline_probabilities)
    exposure = _pair_exposure(splits["test"], pair_sets["test"])

    artifact_path = output_dir / settings.service.artifact_path
    manifest_path = output_dir / settings.service.manifest_path
    digest = save_bundle(bundle, artifact_path, manifest_path)
    metrics: dict[str, Any] = {
        "data_disclosure": (
            "fully synthetic; generated deterministically; no real entity or invoice"
        ),
        "seed": settings.project.seed,
        "rows": len(invoices),
        "base_invoice_groups": int(invoices["group_id"].nunique()),
        "duplicate_records": int(invoices["is_duplicate_record"].sum()),
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "split_candidate_pairs": {name: len(frame) for name, frame in pair_sets.items()},
        "candidate_recall": recalls,
        "validation_selected_threshold": threshold,
        "test_positive_pairs": int(labels["test"].sum()),
        "test_negative_pairs": int((labels["test"] == 0).sum()),
        "model": evaluation_dict(model_evaluation),
        "baseline": evaluation_dict(baseline_evaluation),
        "model_operations": _operational_metrics(
            labels["test"],
            model_predictions,
            exposure,
            settings.model.false_positive_review_cost_try,
        ),
        "baseline_operations": _operational_metrics(
            labels["test"],
            baseline_predictions,
            exposure,
            settings.model.false_positive_review_cost_try,
        ),
        "error_analysis": _error_analysis(
            splits["test"], pair_sets["test"], labels["test"], model_predictions
        ),
        "artifact_sha256": digest,
    }
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "evaluation.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    quality = {
        "synthetic": True,
        "rows": len(invoices),
        "unique_invoice_ids": int(invoices["invoice_id"].nunique()),
        "null_cells": int(invoices.isna().sum().sum()),
        "non_positive_amounts": int((invoices["amount"] <= 0).sum()),
        "date_min": str(invoices["invoice_date"].min()),
        "date_max": str(invoices["invoice_date"].max()),
        "duplicate_groups": int((invoices.groupby("group_id").size() > 1).sum()),
        "candidate_recall": recalls,
    }
    (reports_dir / "data_quality.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )
    samples_dir = output_dir / "data" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.concat(
        [
            invoices[invoices["group_id"].duplicated(keep=False)].head(60),
            invoices[~invoices["group_id"].duplicated(keep=False)].head(60),
        ],
        ignore_index=True,
    )
    sample.drop(columns=["group_id"]).to_csv(samples_dir / "synthetic_invoices.csv", index=False)
    return metrics
