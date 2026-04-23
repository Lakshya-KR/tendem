"""
app/pipeline.py
───────────────
Main Hermes agent pipeline orchestrator.

Flow for each asset (BTC, ETH):
  1. PolymarketAgent  – fetch 5-min prediction market YES probability
  2. KalshiAgent      – fetch 5-min prediction market YES probability
  3. DataAgent        – fetch last 1000 OHLCV bars via Apify + LLM commentary
  4. KronosAgent      – directional forecast (UP/DOWN) for next 5 min
  5. KellyAgent       – position sizing via Kelly Criterion
  6. FeedbackAgent    – evaluate & generate calibration notes
  7. Persist result to JSONL + return PipelineDecision

Scaling ideas already wired in:
  - Multi-asset loop (BTC, ETH – easily extended)
  - Arbitrage signal: compare Polymarket vs Kalshi YES probabilities
  - 15-min vs 3×5-min decomposition hook (see _arbitrage_signal)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.agents.data_agent import DataAgent
from app.agents.feedback_agent import FeedbackAgent
from app.agents.kalshi_agent import KalshiAgent
from app.agents.kelly_agent import KellyAgent
from app.agents.kronos_agent import KronosAgent
from app.agents.polymarket_agent import PolymarketAgent
from app.config import settings
from app.core.logger import get_logger
from app.core.models import MarketPrediction, PipelineDecision

log = get_logger(__name__)

SUPPORTED_ASSETS: List[str] = ["BTC", "ETH"]


# ── Internal helpers ──────────────────────────────────────────────────────

def _arbitrage_signal(poly: Optional[MarketPrediction], kalshi: Optional[MarketPrediction]) -> str:
    """
    Detect cross-platform arbitrage opportunity.
    If Polymarket YES >> Kalshi YES (or vice-versa), flag it.
    """
    if poly is None or kalshi is None:
        return ""
    diff = abs(poly.yes_probability - kalshi.yes_probability)
    if diff > 0.10:
        higher = "Polymarket" if poly.yes_probability > kalshi.yes_probability else "Kalshi"
        lower = "Kalshi" if higher == "Polymarket" else "Polymarket"
        return (
            f"⚡ ARBITRAGE SIGNAL: {higher} YES={max(poly.yes_probability, kalshi.yes_probability):.2f} "
            f"vs {lower} YES={min(poly.yes_probability, kalshi.yes_probability):.2f} "
            f"(spread={diff:.2f}) – investigate cross-platform edge."
        )
    return ""


def _persist_decision(decision: PipelineDecision) -> None:
    path = Path(settings.prediction_history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(decision.model_dump_json() + "\n")
    log.info("Decision persisted → %s", path)


# ── Agents (singletons, instantiated once per process) ───────────────────

_polymarket_agent = PolymarketAgent()
_kalshi_agent = KalshiAgent()
_data_agent = DataAgent()
_kronos_agent = KronosAgent()
_kelly_agent = KellyAgent()
_feedback_agent = FeedbackAgent()


# ── Public API ─────────────────────────────────────────────────────────────

def run_asset_pipeline(asset: str) -> PipelineDecision:
    """
    Execute the full prediction pipeline for a single asset.
    All agent errors are caught and logged; the pipeline never crashes.
    """
    run_id = str(uuid.uuid4())[:8]
    log.info("=" * 60)
    log.info("Pipeline START  run_id=%s  asset=%s", run_id, asset)
    log.info("=" * 60)

    decision = PipelineDecision(run_id=run_id, asset=asset)

    # ── 1. Market predictions ─────────────────────────────────────────────
    poly_pred: Optional[MarketPrediction] = None
    kalshi_pred: Optional[MarketPrediction] = None

    try:
        poly_pred = _polymarket_agent.get_prediction(asset)
        if poly_pred:
            decision.market_predictions.append(poly_pred)
    except Exception as exc:
        log.error("PolymarketAgent error: %s", exc)

    try:
        kalshi_pred = _kalshi_agent.get_prediction(asset)
        if kalshi_pred:
            decision.market_predictions.append(kalshi_pred)
    except Exception as exc:
        log.error("KalshiAgent error: %s", exc)

    # Arbitrage check
    arb = _arbitrage_signal(poly_pred, kalshi_pred)
    if arb:
        log.warning(arb)

    # ── 2. Data fetching ──────────────────────────────────────────────────
    bars = []
    commentary = ""
    try:
        bars, commentary = _data_agent.fetch_and_summarise(asset)
    except Exception as exc:
        log.error("DataAgent error: %s", exc)

    # ── 3. Kronos forecast ────────────────────────────────────────────────
    forecast = None
    try:
        if bars:
            forecast = _kronos_agent.forecast(asset, bars, commentary)
            decision.price_forecast = forecast
    except Exception as exc:
        log.error("KronosAgent error: %s", exc)

    # ── 4. Kelly sizing ───────────────────────────────────────────────────
    if forecast is not None:
        # Use whichever market prediction has higher confidence (lower spread = more liquid)
        best_market = poly_pred or kalshi_pred
        try:
            kelly_pos = _kelly_agent.compute(asset, forecast, best_market)
            decision.kelly_position = kelly_pos
        except Exception as exc:
            log.error("KellyAgent error: %s", exc)

    # ── 5. Build final recommendation ────────────────────────────────────
    parts = []
    if decision.price_forecast:
        f = decision.price_forecast
        parts.append(f"Kronos: {f.direction} (conf={f.confidence:.2f}, method={f.method})")
    for mp in decision.market_predictions:
        parts.append(f"{mp.source.title()}: YES={mp.yes_probability:.2f} '{mp.market_title[:50]}'")
    if decision.kelly_position:
        k = decision.kelly_position
        parts.append(
            f"Kelly: bet {k.direction} @ {k.recommended_fraction*100:.1f}% bankroll "
            f"(edge={k.edge:.3f})"
        )
    if arb:
        parts.append(arb)

    decision.final_recommendation = " | ".join(parts) if parts else "Insufficient data."
    log.info("RECOMMENDATION [%s]: %s", asset, decision.final_recommendation)

    # ── 6. Feedback loop ──────────────────────────────────────────────────
    try:
        feedback = _feedback_agent.evaluate(decision)
        decision.feedback_notes = feedback
    except Exception as exc:
        log.error("FeedbackAgent error: %s", exc)
        decision.feedback_notes = "Feedback unavailable."

    # ── 7. Persist ────────────────────────────────────────────────────────
    _persist_decision(decision)

    log.info("Pipeline END  run_id=%s  asset=%s", run_id, asset)
    return decision


def run_full_pipeline() -> List[PipelineDecision]:
    """Run the pipeline for all supported assets."""
    decisions = []
    for asset in SUPPORTED_ASSETS:
        try:
            d = run_asset_pipeline(asset)
            decisions.append(d)
        except Exception as exc:
            log.error("Pipeline crashed for %s: %s", asset, exc)
    return decisions
