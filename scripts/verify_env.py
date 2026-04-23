"""
verify_env.py — print a masked report of detected env vars.

Run before the demo so the viewer can see the pipeline is configured
without exposing real tokens on screen.

Usage:
  python scripts/verify_env.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass


SECRET_KEYS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "APIFY_TOKEN",
)
URL_KEYS = (
    "OPENROUTER_BASE_URL",
    "OPENAI_BASE_URL",
    "APIFY_BASE_URL",
    "KALSHI_BASE_URL",
    "POLYMARKET_BASE_URL",
)
OTHER_KEYS = (
    "OPENROUTER_MODEL",
    "PREDICTION_HISTORY_PATH",
    "LOG_LEVEL",
    "KRONOS",
    "KRONOS_SCRIPT_PATH",
)


def mask(value: str) -> str:
    if not value or value == "MISSING":
        return "<MISSING>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]} (len={len(value)})"


def main() -> int:
    print("CrowdWisdom env check")
    print("=" * 40)

    print("\nSecrets:")
    missing = 0
    for key in SECRET_KEYS:
        val = os.getenv(key, "")
        status = "OK " if val and val != "MISSING" else "MISS"
        if status == "MISS":
            missing += 1
        print(f"  [{status}] {key:20s} = {mask(val)}")

    print("\nURLs:")
    for key in URL_KEYS:
        val = os.getenv(key, "")
        print(f"  {key:24s} = {val or '<default>'}")

    print("\nOther:")
    for key in OTHER_KEYS:
        val = os.getenv(key, "")
        print(f"  {key:24s} = {val or '<default>'}")

    print("\n" + ("All secrets present." if missing == 0 else f"WARNING: {missing} secret(s) missing."))
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
