"""
app/agents/data_agent.py
────────────────────────
Hermes AIAgent responsible for fetching OHLCV price history via Apify,
then asking the LLM to summarise key market features for the forecaster.
"""
from __future__ import annotations

import json
from typing import List, Tuple

from app.core.hermes_agent import AIAgent
from app.config import settings
from app.core.logger import get_logger
from app.core.models import OHLCVBar
from app.services.apify_service import fetch_ohlcv

log = get_logger(__name__)


def _bar_summary(bars: List[OHLCVBar]) -> dict:
    """Compute lightweight stats from recent bars for LLM context."""
    if not bars:
        return {}
    closes = [b.close for b in bars]
    recent = bars[-20:]
    r_closes = [b.close for b in recent]
    avg_vol = sum(b.volume for b in recent) / len(recent)

    pct_overall = round((closes[-1] - closes[0]) / closes[0] * 100, 4) if closes[0] != 0 else 0.0
    pct_recent = round((r_closes[-1] - r_closes[0]) / r_closes[0] * 100, 4) if r_closes[0] != 0 else 0.0

    return {
        "total_bars": len(bars),
        "first_close": round(closes[0], 2),
        "last_close": round(closes[-1], 2),
        "recent_20_high": round(max(b.high for b in recent), 2),
        "recent_20_low": round(min(b.low for b in recent), 2),
        "recent_20_avg_close": round(sum(r_closes) / len(r_closes), 2),
        "avg_volume_20": round(avg_vol, 4),
        "pct_change_overall": pct_overall,
        "pct_change_recent_20": pct_recent,
    }


class DataAgent(AIAgent):
    """
    Hermes agent that:
      1. Calls Apify to fetch last N OHLCV bars.
      2. Computes summary stats.
      3. Asks the LLM for a brief market-structure commentary.
    Returns the raw bars + LLM commentary for downstream agents.
    """

    def __init__(self) -> None:
        super().__init__(
            
            ephemeral_system_prompt=(
                "You are a quantitative crypto market analyst. "
                "Given OHLCV summary statistics, produce a concise 2-3 sentence "
                "market-structure commentary highlighting trend, momentum, and volatility. "
                "Be factual and brief."
            ),
            quiet_mode=True,
        )

    def fetch_and_summarise(self, asset: str, limit: int = 1000) -> Tuple[List[OHLCVBar], str]:
        """
        Fetch bars via Apify and generate an LLM commentary.

        Returns
        -------
        bars : List[OHLCVBar]
        commentary : str  (LLM-generated market structure notes)
        """
        bars = fetch_ohlcv(asset=asset, limit=limit, timeframe="5m")
        stats = _bar_summary(bars)

        prompt = (
            f"Asset: {asset}\n"
            f"OHLCV summary stats (last {limit} × 5-min bars):\n"
            f"{json.dumps(stats, indent=2)}\n\n"
            "Provide a concise market-structure commentary (2-3 sentences)."
        )

        try:
            commentary = self.chat(prompt)
        except Exception as exc:
            log.warning("DataAgent LLM commentary failed: %s", exc)
            commentary = f"Stats: {stats}"

        log.info("DataAgent [%s]: fetched %d bars; commentary=%d chars", asset, len(bars), len(commentary))
        return bars, commentary
