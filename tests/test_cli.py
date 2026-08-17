from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import faturaiz.cli as cli


def test_cli_train_command(monkeypatch, capsys, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cli, "run_training", lambda settings, output: {"status": "tested"})
    monkeypatch.setattr(
        sys,
        "argv",
        ["faturaiz", "--config", "configs/default.toml", "train", "--output-dir", str(tmp_path)],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"status": "tested"}


def test_cli_score_command(
    monkeypatch,
    capsys,
    tmp_path: Path,
    trained_bundle,
    settings,  # type: ignore[no-untyped-def]
) -> None:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame(
        [
            {
                "invoice_id": "A",
                "vendor_account_id": "SUP-1",
                "invoice_number": "INV-1",
                "invoice_date": "2026-01-01",
                "amount": 100.0,
                "currency": "TRY",
                "po_number": "PO-1",
                "bank_fingerprint": "BANK-1",
                "description": "service",
            }
        ]
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(cli, "load_verified_bundle", lambda artifact, manifest: trained_bundle)
    monkeypatch.setattr(cli, "triage_batch", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        sys,
        "argv",
        ["faturaiz", "--config", "configs/default.toml", "score", str(csv_path)],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out) == []
