from fastapi.testclient import TestClient

from faturaiz.api import create_app


def _payload() -> dict[str, object]:
    base = {
        "vendor_account_id": "SUP-1",
        "invoice_number": "INV-100",
        "amount": "1000.00",
        "currency": "TRY",
        "po_number": "PO-1",
        "bank_fingerprint": "BANK-1",
        "description": "cloud service",
    }
    return {
        "invoices": [
            {**base, "invoice_id": "A", "invoice_date": "2026-01-01"},
            {**base, "invoice_id": "B", "invoice_date": "2026-01-02"},
        ]
    }


def test_api_health_and_triage(settings, trained_bundle) -> None:  # type: ignore[no-untyped-def]
    with TestClient(create_app(settings=settings, bundle=trained_bundle)) as client:
        health = client.get("/health")
        response = client.post("/v1/triage", json=_payload())
    assert health.status_code == 200
    assert response.status_code == 200
    body = response.json()
    assert body["synthetic_training_data"] is True
    assert body["matches"]
    assert "proof" in body["disclaimer"]


def test_api_rejects_duplicate_ids(settings, trained_bundle) -> None:  # type: ignore[no-untyped-def]
    payload = _payload()
    payload["invoices"][1]["invoice_id"] = "A"  # type: ignore[index]
    with TestClient(create_app(settings=settings, bundle=trained_bundle)) as client:
        response = client.post("/v1/triage", json=payload)
    assert response.status_code == 422
