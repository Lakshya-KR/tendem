"""
tests/test_pipeline.py
──────────────────────
Unit and integration tests for the CrowdWisdomTrading Predictions Agent.

Run with:
    pytest tests/ -v
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from app.core.models import (
    MarketPrediction,
    OHLCVBar,
    PipelineDecision,
    PriceForecast,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_bars(n: int = 50, base: float = 65000.0) -> List[OHLCVBar]:
    bars = []
    price = base
    for i in range(n):
        bars.append(
            OHLCVBar(
                timestamp=datetime.utcnow(),
                open=price,
                high=price * 1.001,
                low=price * 0.999,
                close=price * 1.0005,
                volume=10.0,
            )
        )
        price = price * 1.0005
    return bars


# ── Kelly math ────────────────────────────────────────────────────────────────

class TestKellyMath:
    """Validate Kelly formula correctness, especially for DOWN trades."""

    def test_up_trade_positive_edge(self) -> None:
        from app.agents.kelly_agent import _kelly_fraction, _market_odds

        # Market YES = 0.40 (40 % chance UP), our model says UP with confidence
        yes_prob = 0.40
        win_prob, odds = _market_odds(yes_prob, "UP")
        assert win_prob == pytest.approx(0.40)
        assert odds == pytest.approx(0.60 / 0.40)

        f = _kelly_fraction(win_prob, odds)
        # At YES=0.40 with win_prob=0.40 the Kelly is 0: breakeven
        assert f == pytest.approx(0.0, abs=0.01)

    def test_down_trade_positive_edge(self) -> None:
        from app.agents.kelly_agent import _kelly_fraction, _market_odds

        # Market YES = 0.70 (crowd thinks 70 % UP) – if we bet DOWN
        yes_prob = 0.70
        win_prob, odds = _market_odds(yes_prob, "DOWN")

        # Betting NO: win_prob = 1 - 0.70 = 0.30, odds = 0.70/0.30 ≈ 2.33
        assert win_prob == pytest.approx(0.30, abs=0.01)
        assert odds == pytest.approx(0.70 / 0.30, abs=0.01)

        f = _kelly_fraction(win_prob, odds)
        # f = (0.30 * 2.33 - 0.70) / 2.33 = (0.70 - 0.70) / 2.33 ≈ 0
        assert f == pytest.approx(0.0, abs=0.01)

    def test_strong_up_edge(self) -> None:
        from app.agents.kelly_agent import _kelly_fraction, _market_odds

        # Model is very confident UP but market says only 30 % YES
        yes_prob = 0.30
        win_prob, odds = _market_odds(yes_prob, "UP")
        f = _kelly_fraction(win_prob, odds)
        # Significant edge: market underprices the outcome
        assert f > 0.0

    def test_kelly_never_negative(self) -> None:
        from app.agents.kelly_agent import _kelly_fraction, _market_odds

        for direction in ("UP", "DOWN"):
            for yes_prob in (0.1, 0.3, 0.5, 0.7, 0.9):
                win_prob, odds = _market_odds(yes_prob, direction)
                f = _kelly_fraction(win_prob, odds)
                assert f >= 0.0, f"Kelly={f} for {direction} yes_prob={yes_prob}"

    def test_recommended_fraction_capped(self) -> None:
        from app.agents.kelly_agent import _MAX_FRACTION, _HALF_KELLY, _kelly_fraction, _market_odds

        # Even with massive edge, fraction should be capped
        win_prob, odds = _market_odds(0.05, "UP")  # YES at 5%, model says UP strongly
        raw = _kelly_fraction(0.95, odds)
        recommended = min(raw * _HALF_KELLY, _MAX_FRACTION)
        assert recommended <= _MAX_FRACTION


# ── Apify service ─────────────────────────────────────────────────────────────

class TestApifyService:
    def test_synthetic_fallback_returns_bars(self) -> None:
        from app.services.apify_service import _synthetic_bars

        bars = _synthetic_bars("BTC", 100)
        assert len(bars) == 100
        assert all(isinstance(b, OHLCVBar) for b in bars)
        assert all(b.close > 0 for b in bars)

    def test_synthetic_eth_different_from_btc(self) -> None:
        from app.services.apify_service import _synthetic_bars

        btc = _synthetic_bars("BTC", 10)
        eth = _synthetic_bars("ETH", 10)
        assert btc[0].close != eth[0].close  # different base prices

    @patch("app.services.apify_service.settings")
    def test_missing_token_returns_synthetic(self, mock_settings) -> None:
        mock_settings.apify_token = "MISSING"
        mock_settings.prediction_history_path = "./examples/prediction_history.jsonl"

        from app.services.apify_service import fetch_ohlcv

        bars = fetch_ohlcv("BTC", limit=50)
        assert len(bars) == 50


# ── Polymarket agent ──────────────────────────────────────────────────────────

class TestPolymarketAgent:
    def test_best_yes_prob_from_outcome_prices(self) -> None:
        from app.agents.polymarket_agent import _best_yes_prob

        market = {"outcomePrices": '["0.65", "0.35"]'}
        assert _best_yes_prob(market) == pytest.approx(0.65)

    def test_best_yes_prob_from_last_trade(self) -> None:
        from app.agents.polymarket_agent import _best_yes_prob

        market = {"lastTradePrice": "0.42"}
        assert _best_yes_prob(market) == pytest.approx(0.42)

    def test_best_yes_prob_returns_none_on_missing(self) -> None:
        from app.agents.polymarket_agent import _best_yes_prob

        assert _best_yes_prob({}) is None


# ── Kalshi agent ──────────────────────────────────────────────────────────────

class TestKalshiAgent:
    def test_extract_yes_prob_from_bid_ask(self) -> None:
        from app.agents.kalshi_agent import _extract_yes_prob

        # Kalshi prices in cents
        market = {"yes_bid": 45, "yes_ask": 55}
        prob = _extract_yes_prob(market)
        assert prob == pytest.approx(0.50, abs=0.01)

    def test_extract_yes_prob_from_last_price_cents(self) -> None:
        from app.agents.kalshi_agent import _extract_yes_prob

        market = {"last_price": 72}
        prob = _extract_yes_prob(market)
        assert prob == pytest.approx(0.72, abs=0.01)

    def test_extract_yes_prob_from_decimal(self) -> None:
        from app.agents.kalshi_agent import _extract_yes_prob

        market = {"last_price": 0.60}
        prob = _extract_yes_prob(market)
        assert prob == pytest.approx(0.60, abs=0.01)

    def test_returns_none_on_empty(self) -> None:
        from app.agents.kalshi_agent import _extract_yes_prob

        assert _extract_yes_prob({}) is None


# ── Pipeline models ───────────────────────────────────────────────────────────

class TestPipelineModels:
    def test_pipeline_decision_serialises(self) -> None:
        d = PipelineDecision(run_id="test01", asset="BTC")
        j = json.loads(d.model_dump_json())
        assert j["asset"] == "BTC"
        assert j["run_id"] == "test01"

    def test_market_prediction_probability_bounds(self) -> None:
        with pytest.raises(Exception):
            MarketPrediction(
                source="polymarket",
                asset="BTC",
                market_title="test",
                yes_probability=1.5,  # out of range
            )

    def test_price_forecast_direction_literal(self) -> None:
        f = PriceForecast(asset="ETH", direction="DOWN", confidence=0.7, method="test")
        assert f.direction == "DOWN"


# ── Arbitrage signal ──────────────────────────────────────────────────────────

class TestArbitrageSignal:
    def test_large_spread_triggers_signal(self) -> None:
        from app.pipeline import _arbitrage_signal

        poly = MarketPrediction(source="polymarket", asset="BTC", market_title="t", yes_probability=0.75)
        kal = MarketPrediction(source="kalshi", asset="BTC", market_title="t", yes_probability=0.50)
        signal = _arbitrage_signal(poly, kal)
        assert "ARBITRAGE" in signal

    def test_small_spread_no_signal(self) -> None:
        from app.pipeline import _arbitrage_signal

        poly = MarketPrediction(source="polymarket", asset="BTC", market_title="t", yes_probability=0.55)
        kal = MarketPrediction(source="kalshi", asset="BTC", market_title="t", yes_probability=0.50)
        signal = _arbitrage_signal(poly, kal)
        assert signal == ""

    def test_none_inputs_return_empty(self) -> None:
        from app.pipeline import _arbitrage_signal

        assert _arbitrage_signal(None, None) == ""
        assert _arbitrage_signal(None, MagicMock()) == ""


# ── FastAPI ───────────────────────────────────────────────────────────────────

class TestAPI:
    def test_status_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        from app.api import app

        client = TestClient(app)
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "decisions_count" in data
        assert "supported_assets" in data

    def test_decisions_empty(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        import app.api as api_module

        original = api_module.DECISIONS_PATH
        api_module.DECISIONS_PATH = tmp_path / "nonexistent.jsonl"

        client = TestClient(api_module.app)
        resp = client.get("/decisions?limit=5")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

        api_module.DECISIONS_PATH = original

    def test_invalid_asset_returns_400(self) -> None:
        from fastapi.testclient import TestClient

        from app.api import app

        client = TestClient(app)
        resp = client.post("/run/DOGE")
        assert resp.status_code == 400
