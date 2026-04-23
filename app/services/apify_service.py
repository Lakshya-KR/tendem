"""
app/services/apify_service.py
─────────────────────────────
Fetches OHLCV crypto bars via Apify actors.

Primary actor  : moving_beacon-owner1/my-actor-14 (KuCoin OHLCV scraper)
Fallback actor : knotless_cadence~crypto-price-scraper (real-time price only)

If APIFY_TOKEN is missing or the actor call fails, a synthetic dataset is
returned and a clear warning is logged so reviewers can see exactly what
happened.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import List, Optional

import httpx

from app.config import settings
from app.core.logger import get_logger
from app.core.models import OHLCVBar

log = get_logger(__name__)

from app.config import settings as _cfg
_APIFY_BASE = _cfg.apify_base_url

# ── Actor IDs ─────────────────────────────────────────────────────────────
_OHLCV_ACTOR = "moving_beacon-owner1~my-actor-14"   # KuCoin OHLCV
_PRICE_ACTOR = "knotless_cadence~crypto-price-scraper"


def _run_actor_sync(actor_id: str, run_input: dict, timeout: int = 120) -> list:
    """
    Run an Apify actor synchronously and return dataset items.
    Raises RuntimeError on failure.
    """
    token = settings.apify_token
    url = f"{_APIFY_BASE}/acts/{actor_id}/run-sync-get-dataset-items"
    headers = {}
    params = {}
    # If using Toloka proxy, pass Authorization header; else use Apify token param
    if "agents.toloka.ai" in _APIFY_BASE:
        headers["Authorization"] = f"Bearer {token}"
    else:
        params = {"token": token}

    log.info("Apify ▶ actor=%s input=%s", actor_id, run_input)
    try:
        resp = httpx.post(url, params=params, json=run_input, timeout=timeout, headers=headers)
        resp.raise_for_status()
        items = resp.json()
        log.info("Apify ◀ %d items returned", len(items))
        return items
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Apify actor {actor_id} HTTP {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Apify actor {actor_id} failed: {exc}") from exc


def _binance_public_bars(asset: str, n: int = 1000, timeframe: str = "5m") -> Optional[List[OHLCVBar]]:
    """
    Fallback: fetch real OHLCV from Binance public klines API (no auth, no quota).
    Used only when Apify fails, to keep the demo running on real data.
    Returns None on any error so the caller can fall through to synthetic.
    """
    interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
    interval = interval_map.get(timeframe, "5m")
    symbol = f"{asset.upper()}USDT"
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": min(n, 1000)}
    try:
        resp = httpx.get(url, params=params, timeout=20)
        resp.raise_for_status()
        raw = resp.json()
        bars = [
            OHLCVBar(
                timestamp=datetime.utcfromtimestamp(row[0] / 1000),
                open=float(row[1]), high=float(row[2]),
                low=float(row[3]),  close=float(row[4]),
                volume=float(row[5]),
            )
            for row in raw
        ]
        log.info("Binance public fallback: fetched %d real OHLCV bars for %s", len(bars), asset)
        return bars
    except Exception as exc:
        log.warning("Binance public fallback failed: %s", exc)
        return None


def _synthetic_bars(asset: str, n: int = 1000) -> List[OHLCVBar]:
    """Generate plausible-looking synthetic OHLCV bars for fallback/testing."""
    log.warning("⚠️  Using SYNTHETIC price data for %s (set APIFY_TOKEN for real data)", asset)
    import random, math
    random.seed(hash(asset) % 2**32)

    base_price = 65_000.0 if asset.upper() == "BTC" else 2_500.0
    bars: List[OHLCVBar] = []
    price = base_price
    now = datetime.utcnow()

    for i in range(n):
        ts = now - timedelta(minutes=(n - i) * 5)
        change = price * random.gauss(0, 0.003)
        open_ = price
        close = price + change
        high = max(open_, close) * (1 + abs(random.gauss(0, 0.001)))
        low = min(open_, close) * (1 - abs(random.gauss(0, 0.001)))
        vol = random.uniform(5, 50)
        bars.append(OHLCVBar(timestamp=ts, open=open_, high=high, low=low, close=close, volume=vol))
        price = close

    return bars


def fetch_ohlcv(asset: str = "BTC", limit: int = 1000, timeframe: str = "5m") -> List[OHLCVBar]:
    """
    Fetch the last `limit` OHLCV bars for `asset` using Apify.

    Falls back to synthetic data if:
      - APIFY_TOKEN is not configured, OR
      - The Apify actor call fails for any reason.
    """
    if settings.apify_token == "MISSING":
        log.warning("APIFY_TOKEN not set – using synthetic data.")
        return _synthetic_bars(asset, limit)

    symbol = f"{asset.upper()}/USDT"
    try:
        items = _run_actor_sync(
            _OHLCV_ACTOR,
            {"symbol": symbol, "timeframe": timeframe, "data_limit": limit},
        )
    except RuntimeError as exc:
        log.error("Apify OHLCV actor error: %s – trying Binance public fallback", exc)
        binance_bars = _binance_public_bars(asset, limit, timeframe)
        if binance_bars:
            return binance_bars
        return _synthetic_bars(asset, limit)

    bars: List[OHLCVBar] = []
    for item in items:
        try:
            bars.append(
                OHLCVBar(
                    timestamp=datetime.fromisoformat(str(item.get("timestamp", datetime.utcnow().isoformat()))),
                    open=float(item.get("open", item.get("o", 0))),
                    high=float(item.get("high", item.get("h", 0))),
                    low=float(item.get("low", item.get("l", 0))),
                    close=float(item.get("close", item.get("c", 0))),
                    volume=float(item.get("volume", item.get("v", 0))),
                )
            )
        except Exception as parse_err:
            log.debug("Skipping malformed Apify item: %s – %s", item, parse_err)

    if not bars:
        log.warning("Apify returned 0 parseable bars – falling back to synthetic")
        return _synthetic_bars(asset, limit)

    log.info("Fetched %d OHLCV bars for %s from Apify", len(bars), asset)
    return bars
