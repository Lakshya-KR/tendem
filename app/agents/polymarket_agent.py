"""
app/agents/polymarket_agent.py
──────────────────────────────
Hermes AIAgent that finds the next-5-minute crypto prediction market on
Polymarket and returns the implied probability of an UP move.

Uses the public Gamma API (no auth required) to search for BTC/ETH markets.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

import httpx

from app.core.hermes_agent import AIAgent
from app.core.logger import get_logger
from app.core.models import MarketPrediction

log = get_logger(__name__)

from app.config import settings
_GAMMA_URL = settings.polymarket_base_url


def _fetch_crypto_markets(asset: str, limit: int = 20) -> list:
    """
    Query Polymarket Gamma API for open crypto markets matching `asset`.
    Returns raw market dicts.
    """
    keywords = [asset.upper(), asset.lower(), "bitcoin" if asset.upper() == "BTC" else "ethereum"]
    params = {
        "active": "true",
        "closed": "false",
        "limit": str(limit),
        "tag_slug": "crypto",
    }
    try:
        resp = httpx.get(f"{_GAMMA_URL}/markets", params=params, timeout=15)
        resp.raise_for_status()
        markets = resp.json()
        if isinstance(markets, dict):
            markets = markets.get("data", markets.get("markets", []))
    except Exception as exc:
        log.warning("Polymarket Gamma API error: %s", exc)
        return []

    # Filter to markets mentioning the asset
    matched = [
        m for m in markets
        if any(kw in (m.get("question", "") + m.get("slug", "")).lower() for kw in keywords)
    ]
    log.info("Polymarket: found %d markets for %s (total fetched: %d)", len(matched), asset, len(markets))
    return matched


def _best_yes_prob(market: dict) -> Optional[float]:
    """Extract the best YES probability from a market dict."""
    # outcome_prices is a JSON-encoded list e.g. '["0.72","0.28"]'
    prices_raw = market.get("outcomePrices") or market.get("outcomes")
    if prices_raw:
        try:
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            yes_price = float(prices[0]) if prices else None
            if yes_price is not None and 0 <= yes_price <= 1:
                return yes_price
        except Exception:
            pass
    # Fall back to lastTradePrice
    ltp = market.get("lastTradePrice") or market.get("price")
    if ltp is not None:
        try:
            v = float(ltp)
            return v if 0 <= v <= 1 else None
        except Exception:
            pass
    return None


class PolymarketAgent(AIAgent):
    """
    Hermes agent that queries Polymarket for crypto up/down predictions.

    The agent:
      1. Fetches live markets via the public Gamma API.
      2. Uses the LLM to pick the most relevant 5-min prediction market.
      3. Returns a MarketPrediction with the YES probability.
    """

    def __init__(self) -> None:
        super().__init__(
            
            ephemeral_system_prompt=(
                "You are a prediction market analyst specialising in crypto markets on Polymarket. "
                "You receive raw JSON market data and must identify the market most relevant to "
                "the next 5-minute price direction of the given crypto asset. "
                "Reply with ONLY a JSON object: {\"market_index\": <int>, \"reasoning\": \"<str>\"}. "
                "market_index is the 0-based index into the provided list."
            ),
            quiet_mode=True,
        )

    def get_prediction(self, asset: str) -> Optional[MarketPrediction]:
        """
        Returns a MarketPrediction for `asset` or None if no market found.
        """
        markets = _fetch_crypto_markets(asset)
        if not markets:
            log.warning("Polymarket: no markets found for %s", asset)
            return None

        # Ask the LLM to pick the most relevant market
        prompt = (
            f"Asset: {asset}\n"
            f"Available markets (JSON list):\n{json.dumps(markets[:10], indent=2)}\n\n"
            "Which market (0-based index) is most relevant to the next 5-minute price move? "
            "Respond with only the JSON object described in your instructions."
        )

        try:
            raw = self.chat(prompt)
            # Strip markdown fences if present
            raw = raw.strip().strip("```json").strip("```").strip()
            choice = json.loads(raw)
            idx = int(choice.get("market_index", 0))
            reasoning = choice.get("reasoning", "")
        except Exception as exc:
            log.warning("LLM market selection failed (%s) – defaulting to index 0", exc)
            idx = 0
            reasoning = "default"

        idx = min(idx, len(markets) - 1)
        chosen = markets[idx]
        yes_prob = _best_yes_prob(chosen)

        if yes_prob is None:
            log.warning("Polymarket: could not extract YES prob from market %s", chosen.get("slug"))
            return None

        log.info(
            "Polymarket [%s] → market=%r yes_prob=%.3f | %s",
            asset, chosen.get("question", "?")[:60], yes_prob, reasoning[:80],
        )

        return MarketPrediction(
            source="polymarket",
            asset=asset,
            market_title=chosen.get("question", chosen.get("slug", "unknown")),
            yes_probability=yes_prob,
        )
