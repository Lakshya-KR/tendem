"""
app/agents/kalshi_agent.py
──────────────────────────
Hermes AIAgent that fetches next-5-minute crypto prediction markets from
Kalshi using their public REST API v2 (no auth required for market data).
"""
from __future__ import annotations

import json
from typing import Optional

import httpx

from app.core.hermes_agent import AIAgent
from app.core.logger import get_logger
from app.core.models import MarketPrediction

log = get_logger(__name__)

from app.config import settings
_KALSHI_BASE = settings.kalshi_base_url

# Kalshi series tickers for BTC and ETH price prediction markets
_SERIES_MAP = {
    "BTC": ["KXBTC", "KXBTCD", "BTCUSD"],
    "ETH": ["KXETH", "KXETHD", "ETHUSD"],
}


def _fetch_kalshi_markets(asset: str) -> list:
    """
    Fetch open Kalshi markets for `asset` using the public /markets endpoint.
    No authentication required for read-only market data.
    """
    all_markets: list = []
    for series in _SERIES_MAP.get(asset.upper(), [asset.upper()]):
        try:
            resp = httpx.get(
                f"{_KALSHI_BASE}/markets",
                params={"series_ticker": series, "status": "open", "limit": 10},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            markets = data.get("markets", [])
            log.info("Kalshi series=%s returned %d markets", series, len(markets))
            all_markets.extend(markets)
        except Exception as exc:
            log.debug("Kalshi series=%s error: %s", series, exc)

    # Also try keyword search
    if not all_markets:
        keyword = "bitcoin" if asset.upper() == "BTC" else "ethereum"
        try:
            resp = httpx.get(
                f"{_KALSHI_BASE}/markets",
                params={"status": "open", "limit": 25},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            all_markets = [
                m for m in data.get("markets", [])
                if keyword in (m.get("title", "") + m.get("ticker", "")).lower()
            ]
            log.info("Kalshi keyword search for %s → %d markets", keyword, len(all_markets))
        except Exception as exc:
            log.warning("Kalshi keyword search failed: %s", exc)

    return all_markets


def _extract_yes_prob(market: dict) -> Optional[float]:
    """
    Kalshi prices are in cents (0-100). Normalise to 0-1 probability.
    """
    # yes_bid + yes_ask mid
    yes_bid = market.get("yes_bid")
    yes_ask = market.get("yes_ask")
    if yes_bid is not None and yes_ask is not None:
        mid = (float(yes_bid) + float(yes_ask)) / 2.0
        # Kalshi returns cents (1 = $0.01)
        if mid > 1:
            mid /= 100.0
        return mid
    # last_price fallback
    lp = market.get("last_price") or market.get("yes_price")
    if lp is not None:
        v = float(lp)
        return v / 100.0 if v > 1 else v
    return None


class KalshiAgent(AIAgent):
    """
    Hermes agent that queries Kalshi for crypto price prediction markets.

    The agent:
      1. Fetches open markets via the public REST v2 API.
      2. Uses the LLM to select the market closest to a 5-min horizon.
      3. Returns a MarketPrediction with the YES probability.
    """

    def __init__(self) -> None:
        super().__init__(
            
            ephemeral_system_prompt=(
                "You are a CFTC-regulated prediction market analyst specialising in Kalshi. "
                "You receive JSON market data and must select the market most relevant to "
                "the next 5-minute price direction. "
                "Reply with ONLY a JSON object: {\"market_index\": <int>, \"reasoning\": \"<str>\"}. "
                "market_index is the 0-based index into the provided list."
            ),
            quiet_mode=True,
        )

    def get_prediction(self, asset: str) -> Optional[MarketPrediction]:
        """Returns a MarketPrediction for `asset` from Kalshi, or None."""
        markets = _fetch_kalshi_markets(asset)
        if not markets:
            log.warning("Kalshi: no markets found for %s", asset)
            return None

        prompt = (
            f"Asset: {asset}\n"
            f"Available Kalshi markets (JSON list):\n{json.dumps(markets[:10], indent=2)}\n\n"
            "Which market (0-based index) is most relevant to the next 5-minute price move? "
            "Respond with only the JSON object described in your instructions."
        )

        try:
            raw = self.chat(prompt)
            raw = raw.strip().strip("```json").strip("```").strip()
            choice = json.loads(raw)
            idx = int(choice.get("market_index", 0))
            reasoning = choice.get("reasoning", "")
        except Exception as exc:
            log.warning("LLM Kalshi selection failed (%s) – defaulting to index 0", exc)
            idx = 0
            reasoning = "default"

        idx = min(idx, len(markets) - 1)
        chosen = markets[idx]
        yes_prob = _extract_yes_prob(chosen)

        if yes_prob is None:
            log.warning("Kalshi: could not extract YES prob from market %s", chosen.get("ticker"))
            return None

        log.info(
            "Kalshi [%s] → market=%r yes_prob=%.3f | %s",
            asset, chosen.get("title", chosen.get("ticker", "?"))[:60], yes_prob, reasoning[:80],
        )

        return MarketPrediction(
            source="kalshi",
            asset=asset,
            market_title=chosen.get("title", chosen.get("ticker", "unknown")),
            yes_probability=yes_prob,
        )
