"""Strict HTTP boundary schemas."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

SafeId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


class InvoiceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_id: SafeId
    vendor_account_id: SafeId
    invoice_number: SafeId
    invoice_date: date
    amount: Decimal = Field(gt=0, le=1_000_000_000, decimal_places=2)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    po_number: str = Field(default="", max_length=80)
    bank_fingerprint: SafeId
    description: str = Field(default="", max_length=500)

    @field_validator("invoice_id", "vendor_account_id", "bank_fingerprint")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(ord(char) < 32 for char in value):
            raise ValueError("control characters are not allowed")
        return value


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoices: list[InvoiceInput] = Field(min_length=2, max_length=500)

    @field_validator("invoices")
    @classmethod
    def require_unique_ids(cls, invoices: list[InvoiceInput]) -> list[InvoiceInput]:
        ids = [invoice.invoice_id for invoice in invoices]
        if len(ids) != len(set(ids)):
            raise ValueError("invoice_id values must be unique")
        return invoices


class EvidenceItem(BaseModel):
    feature: str
    value: float
    contribution: float


class MatchResult(BaseModel):
    left_invoice_id: str
    right_invoice_id: str
    vendor_ref: str
    risk_score: float
    state: Literal["REVIEW", "HOLD"]
    action: str
    evidence: list[EvidenceItem]


class TriageResponse(BaseModel):
    synthetic_training_data: bool = True
    candidates_reviewed: int
    matches: list[MatchResult]
    disclaimer: str = "Decision support only. A flagged pair is not proof of a duplicate or fraud."
