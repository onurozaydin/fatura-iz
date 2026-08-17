"""Training, evaluation, explainability, and artifact integrity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from faturaiz.features import FEATURE_NAMES


@dataclass(frozen=True)
class Evaluation:
    average_precision: float
    roc_auc: float
    f1: float
    precision: float
    recall: float
    confusion_matrix: list[list[int]]


@dataclass
class ModelBundle:
    pipeline: Pipeline
    threshold: float
    feature_names: tuple[str, ...]
    synthetic_training_data: bool = True


def train_model(features: pd.DataFrame, labels: np.ndarray, *, seed: int) -> Pipeline:
    """Fit an interpretable balanced logistic-regression pipeline."""
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )
    return model.fit(features[list(FEATURE_NAMES)], labels)


def select_threshold(
    probabilities: np.ndarray, labels: np.ndarray, *, min_precision: float
) -> float:
    """Select on validation only: highest F1 while satisfying a precision floor."""
    candidates: list[tuple[float, float, float]] = []
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        if precision >= min_precision:
            candidates.append((f1, precision, float(threshold)))
    if not candidates:
        raise ValueError("validation data cannot satisfy the configured precision floor")
    return max(candidates)[2]


def evaluate(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> Evaluation:
    """Return multiple classification metrics without hiding class imbalance."""
    return Evaluation(
        average_precision=float(average_precision_score(labels, probabilities)),
        roc_auc=float(roc_auc_score(labels, probabilities)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]).astype(int).tolist(),
    )


def predict_probabilities(bundle: ModelBundle, features: pd.DataFrame) -> np.ndarray:
    """Predict positive-class probabilities using the frozen feature order."""
    result: np.ndarray = bundle.pipeline.predict_proba(features[list(bundle.feature_names)])[:, 1]
    return result


def feature_contributions(bundle: ModelBundle, feature_row: pd.Series) -> list[dict[str, Any]]:
    """Explain one logistic prediction using signed standardized contributions."""
    scaler = bundle.pipeline.named_steps["scale"]
    classifier = bundle.pipeline.named_steps["classifier"]
    feature_frame = feature_row[list(bundle.feature_names)].to_frame().T.astype(float)
    values = feature_frame.to_numpy(dtype=float)
    transformed = scaler.transform(feature_frame)[0]
    coefficients = classifier.coef_[0]
    contributions = transformed * coefficients
    ranked: list[dict[str, Any]] = sorted(
        (
            {
                "feature": name,
                "value": round(float(value), 6),
                "contribution": round(float(contribution), 6),
            }
            for name, value, contribution in zip(
                bundle.feature_names, values[0], contributions, strict=True
            )
        ),
        key=lambda item: abs(float(item["contribution"])),
        reverse=True,
    )
    return ranked[:4]


def save_bundle(bundle: ModelBundle, artifact_path: Path, manifest_path: Path) -> str:
    """Persist a local model and write a SHA-256 integrity manifest."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path, compress=3)
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest = {
        "artifact": artifact_path.name,
        "sha256": digest,
        "feature_names": list(bundle.feature_names),
        "threshold": bundle.threshold,
        "synthetic_training_data": bundle.synthetic_training_data,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return digest


def load_verified_bundle(artifact_path: Path, manifest_path: Path) -> ModelBundle:
    """Verify the trusted local artifact digest before deserializing it."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if actual != manifest["sha256"]:
        raise ValueError("model artifact integrity verification failed")
    loaded = joblib.load(artifact_path)
    if not isinstance(loaded, ModelBundle):
        raise TypeError("artifact does not contain a ModelBundle")
    return loaded


def evaluation_dict(evaluation: Evaluation) -> dict[str, Any]:
    """Convert an evaluation result into JSON-compatible values."""
    return asdict(evaluation)
