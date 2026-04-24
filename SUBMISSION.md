# Submission — CrowdWisdomTrading Predictions Agent

- **Repository URL:** https://github.com/Lakshya-KR/tendem
- **APIFY token used (for evaluation):** `apify_api_34ntdb87U3rUYiJCJPhQEehCjmdtNE0mJP8K`
  - Type: personal free-tier token (not a platform/proxy credential)
  - Base URL: `https://api.apify.com/v2` (standard Apify endpoint)
  - Note: this token will be rotated after the evaluation window closes. If it returns HTTP 403 (quota exhausted) during review, the pipeline's automatic Binance public-API fallback keeps the demo operational with real OHLCV data.
- **Demo video:** https://drive.google.com/file/d/1EXyd-XT0RhY-7hVsOTe9VMfiBtA49VlS/view?usp=sharing

## What's included

A Python backend using a Hermes-compatible agent framework: Polymarket and Kalshi market-discovery agents for 5-minute BTC/ETH predictions, an Apify-backed data agent that fetches the last ~1000 OHLCV bars, a real Kronos forecasting path wrapping `shiyu-coder/Kronos` via `scripts/kronos_infer.py` (with an installer script and LLM-proxy fallback), a Kelly-criterion risk agent with secondary LLM validation, a feedback loop that persists decisions to `examples/prediction_history.jsonl`, and a FastAPI service exposing `GET /status`, `GET /decisions`, and `POST /run` / `POST /run/{asset}`. Configuration is via `.env` only — no hardcoded keys or endpoints. The evaluation run was performed with `method=kronos_subprocess` (real Kronos inference, not the LLM fallback).
