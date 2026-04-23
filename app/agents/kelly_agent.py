"""
app/agents/kelly_agent.py
─────────────────────────
Hermes AIAgent that applies the Kelly Criterion to size positions.

Kelly formula (binary bet):
    f* = (p * b - q) / b
where
    p  = probability of winning
    q  = 1 - p = probability of losing
    b  = net odds received on the bet (e.g. on a 50¢ YES market: b = (1-0.5)/0.5 = 1.0)

For prediction markets:
    YES bet at price P: b = (1 - P) / P
    DOWN/NO bet at price P: b = P / (1 - P)

We use HALF-Kelly (0.5 × f*) and cap at 25 % of bankroll for safety.
"""
from __future__ import annotations

import json

from app.core.hermes_agent import AIAgent
from app.config import settings
from app.core.logger import get_logger
from app.core.models import KellyPosition, MarketPrediction, PriceForecast

log = get_logger(__name__)

_MAX_FRACTION = 0.25   # never risk more than 25 % of bankroll
_HALF_KELLY = 0.5      # use half-Kelly for risk reduction


def _kelly_fraction(win_prob: float, odds: float) -> float:
    """
    Raw Kelly fraction.

    Parameters
    ----------
    win_prob : probability our bet wins (0-1)
    odds     : net payout per unit risked (b in the formula)

    Returns the fraction of bankroll to wager (clamped to [0, 1]).
    """
    q = 1.0 - win_prob
    if odds <= 0:
        return 0.0
    f = (win_prob * odds - q) / odds
    return max(0.0, min(1.0, f))


def _market_odds(yes_prob: float, direction: str) -> tuple[float, float]:
    """
    Derive win_prob and net odds for the bet implied by `direction`.

    For UP → we bet YES:  win_prob = yes_prob,  odds = (1 - P) / P
    For DOWN → we bet NO: win_prob = 1 - yes_prob, odds = P / (1 - P)
    """
    yes_prob = max(0.01, min(0.99, yes_prob))  # avoid div by zero

    if direction == "UP":
        win_prob = yes_prob
        odds = (1.0 - yes_prob) / yes_prob
    else:  # DOWN → bet NO
        win_prob = 1.0 - yes_prob
        odds = yes_prob / (1.0 - yes_prob)

    return win_prob, odds


class KellyAgent(AIAgent):
    """
    Hermes agent that computes Kelly position sizing.

    The agent:
      1. Derives market odds from the prediction market YES probability.
      2. Calculates the raw Kelly fraction.
      3. Applies half-Kelly scaling and a hard cap.
      4. Asks the LLM to validate / flag any risk concerns.
    """

    def __init__(self) -> None:
        super().__init__(
            
            ephemeral_system_prompt=(
                "You are a quantitative risk manager. "
                "Given a Kelly position sizing calculation, validate it and flag any concerns. "
                "Reply with ONLY a JSON object: "
                "{\"approved\": true/false, \"notes\": \"<str>\", \"adjusted_fraction\": <float or null>}. "
                "approved=false means the position should be skipped entirely. "
                "adjusted_fraction overrides the recommended fraction if you have a strong reason."
            ),
            quiet_mode=True,
        )

    def compute(
        self,
        asset: str,
        forecast: PriceForecast,
        market_prediction: MarketPrediction | None,
    ) -> KellyPosition:
        """
        Compute Kelly position for `asset` given a directional forecast and market price.

        Falls back to a conservative 1 % fraction if no market data available.
        """
        # Use market YES probability if available, otherwise derive from model confidence
        if market_prediction is not None:
            yes_prob = market_prediction.yes_probability
        else:
            # Approximate: if model is 70 % confident UP, treat market YES ≈ 0.5 (no edge from mkt)
            yes_prob = 0.50
            log.warning("KellyAgent: no market prediction – using yes_prob=0.50 as prior")

        direction = forecast.direction
        win_prob, odds = _market_odds(yes_prob, direction)

        raw_kelly = _kelly_fraction(win_prob, odds)
        edge = win_prob - (1.0 / (1.0 + odds))  # expected value per unit risked

        recommended = min(raw_kelly * _HALF_KELLY, _MAX_FRACTION)

        log.info(
            "Kelly [%s %s]: yes_prob=%.3f win_prob=%.3f odds=%.3f raw=%.4f → rec=%.4f edge=%.4f",
            asset, direction, yes_prob, win_prob, odds, raw_kelly, recommended, edge,
        )

        # Ask LLM to validate
        prompt = (
            f"Asset: {asset}\n"
            f"Direction: {direction}\n"
            f"Market YES probability: {yes_prob:.3f}\n"
            f"Model confidence: {forecast.confidence:.3f}\n"
            f"Win probability: {win_prob:.3f}\n"
            f"Net odds: {odds:.3f}\n"
            f"Raw Kelly fraction: {raw_kelly:.4f}\n"
            f"Recommended (half-Kelly, capped at 25%): {recommended:.4f}\n"
            f"Expected edge: {edge:.4f}\n\n"
            "Validate this position sizing. Reply ONLY with the JSON object."
        )

        approved = True
        notes = "Auto-approved"
        adjusted: float | None = None

        try:
            raw = self.chat(prompt)
            raw = raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(raw)
            approved = bool(parsed.get("approved", True))
            notes = parsed.get("notes", "")
            adjusted = parsed.get("adjusted_fraction")
            if adjusted is not None:
                adjusted = float(adjusted)
        except Exception as exc:
            log.warning("KellyAgent LLM validation failed: %s", exc)

        final_fraction = recommended
        if not approved:
            final_fraction = 0.0
            log.info("KellyAgent: position REJECTED by LLM validator – %s", notes)
        elif adjusted is not None:
            final_fraction = max(0.0, min(_MAX_FRACTION, adjusted))
            log.info("KellyAgent: LLM adjusted fraction to %.4f", final_fraction)

        return KellyPosition(
            asset=asset,
            direction=direction,
            kelly_fraction=raw_kelly,
            recommended_fraction=final_fraction,
            edge=edge,
            win_prob=win_prob,
            odds=odds,
        )
