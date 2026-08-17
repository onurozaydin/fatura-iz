from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from faturaiz.candidates import generate_candidate_pairs, temporal_group_split
from faturaiz.config import load_settings
from faturaiz.features import FEATURE_NAMES, build_pair_features
from faturaiz.model import ModelBundle, train_model
from faturaiz.synthetic import generate_invoices


@pytest.fixture(scope="session")
def settings():  # type: ignore[no-untyped-def]
    return load_settings(Path("configs/default.toml"))


@pytest.fixture(scope="session")
def invoices() -> pd.DataFrame:
    return generate_invoices(seed=17, base_invoices=500, duplicate_rate=0.2)


@pytest.fixture(scope="session")
def trained_bundle(invoices: pd.DataFrame, settings) -> ModelBundle:  # type: ignore[no-untyped-def]
    train = temporal_group_split(invoices)["train"]
    pairs = generate_candidate_pairs(train, settings.candidates)
    features = build_pair_features(train, pairs)
    labels = pairs["label"].to_numpy(dtype=int)
    pipeline = train_model(features, labels, seed=17)
    return ModelBundle(pipeline, 0.5, FEATURE_NAMES)


@pytest.fixture()
def small_batch() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "invoice_id": "A",
                "group_id": "G1",
                "vendor_account_id": "SUP-1",
                "invoice_number": "INV-100",
                "invoice_date": "2026-01-01",
                "amount": 1000.0,
                "currency": "TRY",
                "po_number": "PO-1",
                "bank_fingerprint": "BANK-1",
                "description": "cloud service",
            },
            {
                "invoice_id": "B",
                "group_id": "G1",
                "vendor_account_id": "SUP-1",
                "invoice_number": "inv100",
                "invoice_date": "2026-01-03",
                "amount": 1000.0,
                "currency": "TRY",
                "po_number": "PO-1",
                "bank_fingerprint": "BANK-1",
                "description": "cloud service",
            },
            {
                "invoice_id": "C",
                "group_id": "G2",
                "vendor_account_id": "SUP-1",
                "invoice_number": "INV-999",
                "invoice_date": "2026-01-04",
                "amount": 1000.0,
                "currency": "TRY",
                "po_number": "PO-2",
                "bank_fingerprint": "BANK-1",
                "description": "office supplies",
            },
        ]
    )
