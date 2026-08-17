"""Privacy-safe deterministic invoice scenario generator."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd

_DESCRIPTIONS = (
    "cloud hosting monthly service",
    "office supplies delivery",
    "equipment preventive maintenance",
    "freight and logistics service",
    "software subscription renewal",
    "facility cleaning service",
    "professional consulting hours",
    "packaging material purchase",
)


def _mutate_invoice_number(value: str, rng: np.random.Generator) -> str:
    choices = (
        value.replace("-", ""),
        value.lower(),
        f"{value}-R",
        value.replace("INV", "I NV"),
    )
    return str(rng.choice(choices))


def generate_invoices(
    *, seed: int, base_invoices: int = 3600, duplicate_rate: float = 0.16
) -> pd.DataFrame:
    """Generate synthetic originals, legitimate repeats, and duplicate resubmissions."""
    if base_invoices < 100:
        raise ValueError("base_invoices must be at least 100")
    if not 0 < duplicate_rate < 1:
        raise ValueError("duplicate_rate must be between zero and one")

    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)
    rows: list[dict[str, object]] = []
    vendor_count = max(40, base_invoices // 18)
    recurring_amounts = rng.choice(np.arange(500, 5001, 250), size=vendor_count)
    previous_by_vendor: dict[int, dict[str, object]] = {}

    for index in range(base_invoices):
        vendor_index = int(rng.integers(0, vendor_count))
        vendor = f"SUP-{vendor_index:04d}"
        anchor_date = start + timedelta(days=int(rng.integers(0, 540)))
        recurring = bool(rng.random() < 0.22)
        amount = (
            float(recurring_amounts[vendor_index])
            if recurring
            else float(np.round(np.exp(rng.normal(7.35, 0.85)), 2))
        )
        group_id = f"G-{index:06d}"
        invoice_id = f"F-{index:06d}"
        invoice_number = f"INV-{vendor_index:04d}-{index:06d}"
        po_number = f"PO-{int(rng.integers(1, 900)):05d}" if rng.random() < 0.78 else ""
        bank_fingerprint = f"BANK-{vendor_index:04d}-{vendor_index % 7}"
        description = str(rng.choice(_DESCRIPTIONS))
        previous = previous_by_vendor.get(vendor_index)
        if previous is not None and rng.random() < 0.18:
            # A legitimate recurring invoice: intentionally similar to a duplicate so the
            # benchmark contains operationally ambiguous hard negatives.
            previous_date = date.fromisoformat(str(previous["invoice_date"]))
            anchor_date = previous_date + timedelta(days=int(rng.integers(18, 56)))
            amount = float(cast(Any, previous["amount"]))
            description = str(previous["description"])
            po_number = str(previous["po_number"]) if rng.random() < 0.55 else po_number
        rows.append(
            {
                "invoice_id": invoice_id,
                "group_id": group_id,
                "vendor_account_id": vendor,
                "invoice_number": invoice_number,
                "invoice_date": anchor_date.isoformat(),
                "amount": amount,
                "currency": "TRY",
                "po_number": po_number,
                "bank_fingerprint": bank_fingerprint,
                "description": description,
                "is_duplicate_record": False,
                "duplicate_type": "original",
            }
        )
        previous_by_vendor[vendor_index] = rows[-1]

        if rng.random() < duplicate_rate:
            duplicate_type = str(
                rng.choice(
                    ("exact", "format_shift", "amount_shift", "weak_resubmission"),
                    p=(0.25, 0.35, 0.20, 0.20),
                )
            )
            duplicate_number = invoice_number
            duplicate_amount = amount
            duplicate_po = po_number
            if duplicate_type in {"format_shift", "amount_shift"}:
                duplicate_number = _mutate_invoice_number(invoice_number, rng)
            if duplicate_type == "amount_shift":
                duplicate_amount = float(np.round(amount * rng.uniform(0.985, 1.015), 2))
                duplicate_po = "" if rng.random() < 0.5 else po_number
            if duplicate_type == "weak_resubmission":
                duplicate_number = f"DOC-{vendor_index:04d}-{int(rng.integers(1000, 9999))}"
                duplicate_amount = float(np.round(amount * rng.uniform(0.97, 1.03), 2))
                duplicate_po = ""
            rows.append(
                {
                    "invoice_id": f"{invoice_id}-D",
                    "group_id": group_id,
                    "vendor_account_id": vendor,
                    "invoice_number": duplicate_number,
                    "invoice_date": (
                        anchor_date + timedelta(days=int(rng.integers(1, 22)))
                    ).isoformat(),
                    "amount": duplicate_amount,
                    "currency": "TRY",
                    "po_number": duplicate_po,
                    "bank_fingerprint": bank_fingerprint,
                    "description": description,
                    "is_duplicate_record": True,
                    "duplicate_type": duplicate_type,
                }
            )

    frame = pd.DataFrame(rows).sort_values(["invoice_date", "invoice_id"]).reset_index(drop=True)
    return frame
