from dataclasses import replace
from pathlib import Path

from faturaiz.config import ProjectConfig
from faturaiz.training import run_training


def test_end_to_end_training_writes_verified_evidence(tmp_path: Path, settings) -> None:  # type: ignore[no-untyped-def]
    compact = replace(
        settings,
        project=ProjectConfig(seed=123, synthetic_base_invoices=500, synthetic_duplicate_rate=0.2),
    )
    metrics = run_training(compact, tmp_path)
    assert metrics["candidate_recall"]["test"] >= 0.95
    assert metrics["model"]["f1"] >= metrics["baseline"]["f1"]
    assert (tmp_path / "models/model.joblib").exists()
    assert (tmp_path / "reports/evaluation.json").exists()
    assert (tmp_path / "data/samples/synthetic_invoices.csv").exists()
