"""
app/core/llm.py
───────────────
Thin OpenRouter wrapper used by all agents.
Uses the OpenAI-compatible endpoint so it works identically to how
hermes-agent's AIAgent sends requests (OPENROUTER_API_KEY + base_url).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings
from app.core.logger import get_logger

log = get_logger(__name__)

from app.config import settings as _cfg
OPENROUTER_BASE = _cfg.openrouter_base_url


def chat(
    messages: List[Dict[str, str]],
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    model: Optional[str] = None,
) -> str:
    """
    Send a chat request to OpenRouter and return the assistant text.
    Raises RuntimeError on HTTP errors or missing API key.
    """
    if settings.openrouter_api_key == "MISSING":
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to your .env file.")

    payload: Dict[str, Any] = {
        "model": model or settings.openrouter_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": (
            [{"role": "system", "content": system}] + messages if system else messages
        ),
    }

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://crowdwisdomtrading.com",
        "X-Title": "CrowdWisdomTrading Predictions Agent",
    }

    log.debug("LLM request → model=%s messages=%d", payload["model"], len(payload["messages"]))

    try:
        resp = httpx.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers=headers,
            content=json.dumps(payload),
            timeout=60,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        log.error("OpenRouter HTTP error %s: %s", exc.response.status_code, exc.response.text)
        raise RuntimeError(f"OpenRouter error {exc.response.status_code}: {exc.response.text}") from exc

    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
        log.debug("LLM response length=%d chars", len(text))
        return text
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {data}") from exc
