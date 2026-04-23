# Submission — CrowdWisdomTrading Predictions Agent

- **Repository URL:** https://github.com/tolokaexpert-tech/crowdwisdom.git
- **APIFY tokens used (for evaluation only):**
  `tlk_aidGeMy59jBZU4-maDdk3Q.045e._rBWkROtBbLDAZtvegHG88NSAysJBRsWtGNIzpJ9-AL73IyK3JZbDpREf8bt2MwDgiEeDRG5DYMovsXvYm8ZIqZeeHvN2B06_vPnJbUEvTA`
- **Demo video:** [TO-BE-ADDED — will be inserted before sending the submission email]

## What's included

A Python backend using a Hermes-compatible agent framework: Polymarket and Kalshi market-discovery agents for 5-minute BTC/ETH predictions, an Apify-backed data agent that fetches the last ~1000 OHLCV bars, a Kronos forecasting path (`scripts/setup_kronos.*` installer + `scripts/kronos_infer.py`, with an LLM-proxy fallback when real Kronos is unavailable), a Kelly-criterion risk agent, a feedback loop that persists decisions to `examples/prediction_history.jsonl`, and a FastAPI service exposing `GET /status`, `GET /decisions`, and `POST /run` / `POST /run/{asset}`. Configuration is via `.env` only — no hardcoded keys or endpoints.
