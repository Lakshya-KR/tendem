"""
app/api.py
──────────
FastAPI application exposing pipeline status, decisions, and a trigger
endpoint for running the prediction pipeline on demand.

Endpoints
---------
GET  /status            – Health check + decision count
GET  /decisions         – Last N pipeline decisions
POST /run/{asset}       – Trigger a single-asset pipeline run
POST /run               – Trigger full pipeline (BTC + ETH)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from app.config import settings

DEFAULT_PATH = Path(settings.prediction_history_path)
DECISIONS_PATH = Path(os.getenv("DECISIONS_PATH", str(DEFAULT_PATH)))

app = FastAPI(
    title="CrowdWisdomTrading Predictions Agent API",
    version="1.0.0",
    description=(
        "Multi-agent crypto prediction pipeline using Polymarket, Kalshi, "
        "Apify OHLCV data, Kronos-style forecasting, and Kelly position sizing."
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _read_jsonl_tail(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    selected = lines[-limit:] if limit > 0 else lines
    out: List[Dict[str, Any]] = []
    for ln in selected:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/status", tags=["Health"])
def status() -> JSONResponse:
    """Health check. Returns decision file stats."""
    exists = DECISIONS_PATH.exists()
    count = 0
    if exists:
        try:
            with DECISIONS_PATH.open("r", encoding="utf-8") as fh:
                count = sum(1 for ln in fh if ln.strip())
        except Exception:
            count = 0
    return JSONResponse({
        "ok": True,
        "decisions_path": str(DECISIONS_PATH),
        "decisions_file_exists": exists,
        "decisions_count": count,
        "supported_assets": ["BTC", "ETH"],
        "openrouter_model": settings.openrouter_model,
        "apify_configured": settings.apify_token != "MISSING",
        "llm_configured": settings.openrouter_api_key != "MISSING",
    })


@app.get("/decisions", tags=["Data"])
def decisions(limit: int = Query(20, ge=1, le=500)) -> JSONResponse:
    """Return the last `limit` pipeline decisions."""
    if not DECISIONS_PATH.exists():
        return JSONResponse({"items": [], "count": 0})
    items = _read_jsonl_tail(DECISIONS_PATH, limit)
    return JSONResponse({"items": items, "count": len(items)})


@app.post("/run/{asset}", tags=["Pipeline"])
def run_asset(asset: str, background_tasks: BackgroundTasks) -> JSONResponse:
    """
    Trigger the prediction pipeline for a single asset (BTC or ETH).
    Runs synchronously and returns the decision.
    """
    asset = asset.upper()
    if asset not in ("BTC", "ETH"):
        raise HTTPException(status_code=400, detail=f"Unsupported asset '{asset}'. Use BTC or ETH.")

    # Import here to avoid circular imports at module load time
    from app.pipeline import run_asset_pipeline
    try:
        decision = run_asset_pipeline(asset)
        return JSONResponse({"ok": True, "decision": json.loads(decision.model_dump_json())})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/run", tags=["Pipeline"])
def run_all() -> JSONResponse:
    """
    Trigger the full prediction pipeline for all supported assets (BTC + ETH).
    Returns all decisions.
    """
    from app.pipeline import run_full_pipeline
    try:
        decisions_list = run_full_pipeline()
        return JSONResponse({
            "ok": True,
            "decisions": [json.loads(d.model_dump_json()) for d in decisions_list],
            "count": len(decisions_list),
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Run locally:
#   pip install -r requirements.txt
#   cp .env.example .env  # then fill in your keys
#   python -m uvicorn app.api:app --reload
