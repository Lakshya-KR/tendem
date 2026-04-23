#!/usr/bin/env python3
"""
main.py – CLI entrypoint for the CrowdWisdomTrading Predictions Agent.

Usage
-----
# Run full pipeline once (BTC + ETH):
    python main.py

# Run for a specific asset:
    python main.py --asset BTC

# Run in loop mode (every N minutes):
    python main.py --loop --interval 5

# Dry-run (skip LLM calls, use synthetic data):
    python main.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
import time

from app.core.logger import get_logger
from app.pipeline import SUPPORTED_ASSETS, run_asset_pipeline, run_full_pipeline

log = get_logger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CrowdWisdomTrading Predictions Agent – multi-agent crypto pipeline"
    )
    p.add_argument(
        "--asset", choices=SUPPORTED_ASSETS + [a.lower() for a in SUPPORTED_ASSETS],
        default=None, help="Single asset to run (default: all)"
    )
    p.add_argument(
        "--loop", action="store_true",
        help="Run continuously on a fixed interval"
    )
    p.add_argument(
        "--interval", type=int, default=5,
        help="Interval in minutes between pipeline runs (only with --loop)"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print config and exit without running the pipeline"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from app.config import settings
    log.info("CrowdWisdomTrading Predictions Agent")
    log.info("  Model       : %s", settings.openrouter_model)
    log.info("  Apify       : %s", "configured" if settings.apify_token != "MISSING" else "MISSING – synthetic data will be used")
    log.info("  LLM         : %s", "configured" if settings.openrouter_api_key != "MISSING" else "MISSING – set OPENROUTER_API_KEY")
    log.info("  History file: %s", settings.prediction_history_path)

    if args.dry_run:
        log.info("Dry-run mode – exiting without running pipeline.")
        return 0

    asset = args.asset.upper() if args.asset else None

    def _run_once() -> None:
        if asset:
            run_asset_pipeline(asset)
        else:
            run_full_pipeline()

    if args.loop:
        log.info("Loop mode: running every %d minute(s). Ctrl-C to stop.", args.interval)
        while True:
            try:
                _run_once()
            except KeyboardInterrupt:
                log.info("Interrupted – stopping.")
                return 0
            except Exception as exc:
                log.error("Pipeline error: %s – will retry next cycle", exc)
            log.info("Sleeping %d minute(s)...", args.interval)
            time.sleep(args.interval * 60)
    else:
        try:
            _run_once()
        except Exception as exc:
            log.error("Pipeline failed: %s", exc)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
