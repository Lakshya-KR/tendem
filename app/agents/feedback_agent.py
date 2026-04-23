"""
app/agents/feedback_agent.py
─────────────────────────────
Hermes AIAgent that closes the loop:
  - Reads recent pipeline decisions from the JSONL history.
  - Evaluates how well past forecasts performed (directional accuracy).
  - Generates improvement notes fed back into the orchestrator.

This mirrors the Hermes "closed learning loop" concept where the agent
inspects its own trajectory and refines future behaviour.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from app.config import settings
from app.core.hermes_agent import AIAgent
from app.core.logger import get_logger
from app.core.models import PipelineDecision

log = get_logger(__name__)


def _load_recent_decisions(n: int = 10) -> List[dict]:
    """Load last N decision records from the JSONL history file."""
    path = Path(settings.prediction_history_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    records = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


class FeedbackAgent(AIAgent):
    """
    Hermes agent that reviews recent pipeline decisions and generates
    calibration notes for the next cycle.
    """

    def __init__(self) -> None:
        super().__init__(
            
            ephemeral_system_prompt=(
                "You are a meta-analyst for an AI trading pipeline. "
                "Given recent pipeline decisions (JSON records), identify: "
                "1) Directional consistency across Polymarket / Kalshi and Kronos forecasts. "
                "2) Whether Kelly fractions appear well-calibrated (not too aggressive). "
                "3) Any patterns suggesting a systematic bias to correct. "
                "Output 2-4 actionable bullet points as plain text. "
                "Be concise and specific."
            ),
            quiet_mode=True,
        )

    def evaluate(self, current_decision: PipelineDecision) -> str:
        """
        Evaluate the current decision in context of recent history.
        Returns feedback notes as a plain-text string.
        """
        history = _load_recent_decisions(n=10)
        history_text = json.dumps(history, indent=2, default=str)

        current_text = json.dumps(
            current_decision.model_dump(mode="json"), indent=2, default=str
        )

        prompt = (
            f"Recent pipeline history ({len(history)} records):\n{history_text}\n\n"
            f"Current decision:\n{current_text}\n\n"
            "Provide feedback notes (2-4 actionable bullet points)."
        )

        try:
            notes = self.chat(prompt)
        except Exception as exc:
            log.warning("FeedbackAgent LLM call failed: %s", exc)
            notes = "Feedback unavailable (LLM error)."

        log.info("FeedbackAgent notes (%d chars): %s", len(notes), notes[:120])
        return notes
