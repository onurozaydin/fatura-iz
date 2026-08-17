import pandas as pd

from faturaiz.service import triage_batch


def test_service_masks_vendor_and_returns_human_action(
    small_batch: pd.DataFrame,
    settings,
    trained_bundle,  # type: ignore[no-untyped-def]
) -> None:
    results = triage_batch(
        small_batch,
        bundle=trained_bundle,
        candidate_config=settings.candidates,
        reference_pepper="test",
    )
    assert results
    assert results[0]["vendor_ref"].startswith("ref_")
    assert "SUP-1" not in str(results)
    assert results[0]["state"] in {"HOLD", "REVIEW"}
