"""
app/core/hermes_agent.py
──────────────────────────
Hermes Agent integration with graceful fallback.

If the real `hermes-agent` package is installed, we import and use its
`AIAgent`. Otherwise we fall back to a lightweight compatible shim so the
project remains runnable without the dependency.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.llm import chat as _llm_chat
from app.core.logger import get_logger

log = get_logger(__name__)

# Auto-detect the real Hermes Agent framework. If the package is installed
# AND imports cleanly, we re-export its AIAgent so every downstream agent
# uses it directly. If import fails for any reason (package not installed,
# or a transitive import error like the known "agent.transports" issue),
# we log the reason and transparently fall back to the compatible shim
# defined below. Behaviour is therefore identical from the caller's POV.
_HermesAIAgent = None
HermesAvailable = False
try:
    from run_agent import AIAgent as _HermesAIAgent  # type: ignore  # noqa: F401
    HermesAvailable = True
    log.info("Hermes Agent framework detected - using real nousresearch/hermes-agent.AIAgent")
except Exception as _hermes_err:  # ImportError OR runtime import-time errors
    log.info(
        "Hermes Agent not available (%s: %s) - using compatible shim. "
        "Install with: pip install git+https://github.com/NousResearch/hermes-agent.git",
        type(_hermes_err).__name__, str(_hermes_err)[:120],
    )


if HermesAvailable:
    # Re-export Hermes' AIAgent for direct use elsewhere in the project
    AIAgent = _HermesAIAgent  # type: ignore
else:
    class AIAgent:  # fallback shim
        """
        Hermes-compatible AIAgent shim.

        Mirrors the interface of `run_agent.AIAgent` from nousresearch/hermes-agent
        (quiet_mode, ephemeral_system_prompt, chat()), enabling drop-in replacement
        if/when the real package is installed.
        """

        def __init__(
            self,
            name: str = "Agent",
            ephemeral_system_prompt: str = "You are a helpful AI assistant.",
            quiet_mode: bool = True,
            max_iterations: int = 5,
        ) -> None:
            self.name = name
            self.system_prompt = ephemeral_system_prompt
            self.quiet_mode = quiet_mode
            self.max_iterations = max_iterations
            self._history: List[Dict[str, str]] = []

        def chat(self, user_message: str, **kwargs: Any) -> str:
            if not self.quiet_mode:
                log.info("[%s] → %s", self.name, user_message[:120])
            self._history.append({"role": "user", "content": user_message})
            response = _llm_chat(messages=self._history, system=self.system_prompt, **kwargs)
            self._history.append({"role": "assistant", "content": response})
            if not self.quiet_mode:
                log.info("[%s] ← %s", self.name, response[:120])
            return response

        def reset(self) -> None:
            self._history = []

        def __repr__(self) -> str:  # pragma: no cover
            return f"AIAgent(name={self.name!r}, turns={len(self._history)})"
