"""FastAPI application with bounded, privacy-aware request handling."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from faturaiz.config import Settings, load_settings
from faturaiz.model import ModelBundle, load_verified_bundle
from faturaiz.schemas import MatchResult, TriageRequest, TriageResponse
from faturaiz.service import triage_batch

LOGGER = logging.getLogger("faturaiz.api")


def _records(request: TriageRequest) -> pd.DataFrame:
    return pd.DataFrame([invoice.model_dump(mode="json") for invoice in request.invoices]).assign(
        group_id=lambda frame: frame["invoice_id"]
    )


def create_app(*, settings: Settings | None = None, bundle: ModelBundle | None = None) -> FastAPI:
    resolved = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved
        app.state.bundle = bundle or load_verified_bundle(
            resolved.service.artifact_path, resolved.service.manifest_path
        )
        LOGGER.info("model_loaded threshold=%.3f", app.state.bundle.threshold)
        yield

    app = FastAPI(
        title="FaturaIz API",
        version="0.1.0",
        description="Explainable duplicate-invoice triage; human review is required.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        loaded: ModelBundle = request.app.state.bundle
        return {"status": "ok", "model_loaded": True, "threshold": loaded.threshold}

    @app.post("/v1/triage", response_model=TriageResponse)
    def triage(payload: TriageRequest, request: Request) -> TriageResponse:
        if len(payload.invoices) > resolved.service.max_batch_size:
            raise HTTPException(status_code=413, detail="batch exceeds configured limit")
        LOGGER.info("triage_request invoice_count=%d", len(payload.invoices))
        try:
            rows = triage_batch(
                _records(payload),
                bundle=request.app.state.bundle,
                candidate_config=resolved.candidates,
                reference_pepper=os.getenv("FATURAIZ_REFERENCE_PEPPER", "demo-only-pepper"),
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return TriageResponse(
            candidates_reviewed=len(rows), matches=[MatchResult(**row) for row in rows]
        )

    return app


def app_from_environment() -> FastAPI:
    config_path = Path(os.getenv("FATURAIZ_CONFIG", "configs/default.toml"))
    return create_app(settings=load_settings(config_path))


app = app_from_environment()
