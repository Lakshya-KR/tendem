"""
app/agents/kronos_agent.py
──────────────────────────
Hermes AIAgent that produces UP/DOWN price forecasts.

Integration path:
  Real Kronos (github.com/shiyu-coder/Kronos) requires a GPU environment and
  its own model weights, so direct subprocess execution is environment-specific.
  This agent follows the Kronos interface (time-series → directional forecast)
  and delegates to the LLM when the Kronos subprocess is unavailable, producing
  a probabilistic directional prediction from the OHLCV statistics.

To plug in real Kronos:
  1. Clone and install the Kronos repo.
  2. Set KRONOS_SCRIPT_PATH=/path/to/kronos/predict.py in your .env.
  3. This agent will call it as a subprocess automatically.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

from app.core.hermes_agent import AIAgent
from app.config import settings
from app.core.logger import get_logger
from app.core.models import OHLCVBar, PriceForecast

log = get_logger(__name__)

# KRONOS=on by default: if KRONOS_SCRIPT_PATH is unset we use the bundled
# scripts/kronos_infer.py. Set KRONOS=off (or KRONOS_SCRIPT_PATH=) to skip
# the subprocess entirely and go straight to the LLM proxy path.
_DEFAULT_KRONOS_SCRIPT = str(Path(__file__).resolve().parents[2] / "scripts" / "kronos_infer.py")
_KRONOS_ENABLED = os.getenv("KRONOS", "on").strip().lower() not in ("off", "0", "false", "no")
_KRONOS_SCRIPT = os.getenv("KRONOS_SCRIPT_PATH") or (_DEFAULT_KRONOS_SCRIPT if _KRONOS_ENABLED else "")


def _try_kronos_subprocess(bars: List[OHLCVBar], asset: str) -> PriceForecast | None:
    """
    Attempt to run Kronos as a subprocess if the script path is configured.
    Returns None if Kronos is not available.
    """
    if not _KRONOS_SCRIPT or not os.path.exists(_KRONOS_SCRIPT):
        return None

    # Write closes to a temp CSV for Kronos input
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("timestamp,close\n")
        for b in bars[-512:]:  # Kronos typically uses last 512 points
            f.write(f"{b.timestamp.isoformat()},{b.close}\n")
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", _KRONOS_SCRIPT, "--input", tmp_path, "--asset", asset],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()[:200]
            stdout = (result.stdout or "").strip()[:200]
            log.warning("Kronos subprocess exit=%s stdout=%s stderr=%s",
                        result.returncode, stdout, stderr)
            return None
        output = json.loads(result.stdout)
        if "error" in output:
            log.warning("Kronos subprocess reported error: %s", output["error"])
            return None
        raw_dir = output.get("direction", "UP")
        if isinstance(raw_dir, (int, float)):
            direction = "UP" if float(raw_dir) > 0 else "DOWN"
        else:
            direction = str(raw_dir).upper()
            if direction not in ("UP", "DOWN"):
                direction = "UP"
        confidence = float(output.get("confidence", 0.6))
        log.info("Kronos subprocess: %s %s conf=%.3f", asset, direction, confidence)
        return PriceForecast(
            asset=asset,
            direction=direction,
            confidence=confidence,
            horizon_minutes=5,
            method="kronos_subprocess",
        )
    except Exception as exc:
        log.warning("Kronos subprocess failed: %s", exc)
        return None
    finally:
        os.unlink(tmp_path)


import subprocess, json, sys
from pathlib import Path

class KronosAgent(AIAgent):
    """
    Hermes agent that predicts next 5-minute price direction using Kronos.

    Falls back to LLM-based reasoning from OHLCV stats if Kronos is not
    installed in the current environment.
    """

    def __init__(self) -> None:
        super().__init__(
            
            ephemeral_system_prompt=(
                "You are a quantitative time-series forecasting model. "
                "Given OHLCV statistics for a crypto asset, predict whether the "
                "price will go UP or DOWN in the next 5 minutes. "
                "Reply with ONLY a JSON object: "
                "{\"direction\": \"UP\" or \"DOWN\", \"confidence\": <float 0-1>, \"reasoning\": \"<str>\"}. "
                "Base your decision on momentum, recent trend, and volatility patterns."
            ),
            quiet_mode=True,
        )

    def forecast(self, asset: str, bars: List[OHLCVBar], commentary: str = "") -> PriceForecast:
        """
        Generate a 5-minute directional forecast for `asset`.

        Tries Kronos subprocess first; falls back to LLM reasoning.
        """
        # Try real Kronos first
        kronos_result = _try_kronos_subprocess(bars, asset)
        if kronos_result:
            return kronos_result

        # LLM-based Kronos-style reasoning
        closes = [b.close for b in bars[-30:]]
        volumes = [b.volume for b in bars[-30:]]
        returns = [
            round((closes[i] - closes[i - 1]) / closes[i - 1] * 100, 4)
            if closes[i - 1] != 0 else 0.0
            for i in range(1, len(closes))
        ]

        prompt = (
            f"Asset: {asset}\n"
            f"Last 30 closes (5-min bars): {closes}\n"
            f"Last 30 returns (%): {returns}\n"
            f"Last 30 volumes: {volumes}\n"
            f"Market commentary: {commentary}\n\n"
            "Predict next 5-minute direction. Reply ONLY with the JSON object."
        )

        try:
            raw = self.chat(prompt)
            raw = raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(raw)
            direction = parsed.get("direction", "UP").upper()
            if direction not in ("UP", "DOWN"):
                direction = "UP"
            confidence = float(parsed.get("confidence", 0.55))
            confidence = max(0.5, min(0.99, confidence))  # clamp
            reasoning = parsed.get("reasoning", "")
        except Exception as exc:
            log.warning("KronosAgent LLM parse error: %s – defaulting UP 0.55", exc)
            direction = "UP"
            confidence = 0.55
            reasoning = "parse error"

        log.info(
            "KronosAgent [%s] → %s conf=%.3f | %s",
            asset, direction, confidence, reasoning[:80],
        )

        return PriceForecast(
            asset=asset,
            direction=direction,
            confidence=confidence,
            horizon_minutes=5,
            method="llm_kronos_proxy",
        )
