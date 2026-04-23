"""app/core/models.py – Pydantic models shared across all agents."""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ── Market-prediction snapshot ─────────────────────────────────────────────

class MarketPrediction(BaseModel):
    """A single YES-probability reading from a prediction market."""
    source: Literal["polymarket", "kalshi"]
    asset: str                          # e.g. "BTC", "ETH"
    market_title: str
    yes_probability: float = Field(..., ge=0.0, le=1.0)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


# ── OHLCV bar ──────────────────────────────────────────────────────────────

class OHLCVBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# ── Price forecast ─────────────────────────────────────────────────────────

class PriceForecast(BaseModel):
    asset: str
    direction: Literal["UP", "DOWN"]
    confidence: float = Field(..., ge=0.0, le=1.0)    # model certainty
    horizon_minutes: int = 5
    method: str = "kronos"


# ── Kelly position ─────────────────────────────────────────────────────────

class KellyPosition(BaseModel):
    asset: str
    direction: Literal["UP", "DOWN"]
    kelly_fraction: float               # raw Kelly
    recommended_fraction: float         # half-Kelly or capped
    edge: float                         # expected edge
    win_prob: float
    odds: float


# ── Pipeline decision (written to JSONL) ──────────────────────────────────

class PipelineDecision(BaseModel):
    run_id: str
    asset: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    market_predictions: List[MarketPrediction] = []
    price_forecast: Optional[PriceForecast] = None
    kelly_position: Optional[KellyPosition] = None

    feedback_notes: str = ""
    final_recommendation: str = ""
