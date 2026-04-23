# CrowdWisdomTrading Predictions Agent

A sophisticated multi-agent pipeline built with a **Hermes-compatible agent framework** to predict short-horizon (5-minute) crypto price movements and optimize position sizing.

## 🚀 Overview

This project implements a complete backend for crypto market research and prediction. It coordinates multiple specialized agents to analyze market sentiment, fetch historical data, forecast directional moves, and manage risk using the Kelly Criterion.

### Key Features
- **Multi-Agent Orchestration**: Specialized agents for Polymarket, Kalshi, Data Scraper, Kronos Forecasting, and Kelly Risk Management.
- **Hermes Closed-Loop Learning**: A `FeedbackAgent` that reviews past performance and generates calibration notes for future cycles.
- **Cross-Platform Arbitrage**: Automatically detects spread opportunities between Polymarket and Kalshi predictions.
- **Graceful Fallbacks**: Built-in synthetic data generation and LLM-proxy forecasting if external services (like Apify or real Kronos) are unavailable.
- **Developer-First Design**: Comprehensive logging, error handling, and a FastAPI-based monitoring API.

---

## 🛠 Architecture

The pipeline flows as follows for each asset (BTC, ETH):
1. **Polymarket/Kalshi Agents**: Discover the most relevant 5-minute prediction markets and extract "YES" probabilities.
2. **Data Agent**: Fetches the last 1000 OHLCV bars via Apify and generates market-structure commentary.
3. **Kronos Agent**: Produces a directional forecast (UP/DOWN) for the next 5-minute horizon.
4. **Kelly Agent**: Sizes the position based on edge and win probability, capped at 25% bankroll for safety.
5. **Feedback Agent**: Analyzes the pipeline's trajectory and persists decisions to `prediction_history.jsonl`.

---

## 🚦 Quick Start

### 1. Prerequisites
- Python 3.10+ (tested on 3.11, 3.12, 3.13)
- **`git` on your system PATH** — `requirements.txt` installs the Hermes Agent framework from its GitHub repository (there is no PyPI wheel), so pip shells out to git during resolution. Install from [git-scm.com](https://git-scm.com/downloads) if missing.
- *Optional:* GPU for the fastest real Kronos execution (CPU works too, just slower).

> If your environment genuinely cannot provide `git` and you still want a working pipeline, comment out the `hermes-agent` line in `requirements.txt` and run `pip install -r requirements.txt`. The code's runtime try/except will catch the missing import and engage the bundled shim automatically (same `AIAgent` interface, no code changes needed).

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/Lakshya-KR/tendem.git
cd tendem

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate

# Install dependencies (no git required — the Hermes Agent framework is optional)
pip install -r requirements.txt
```

### 2a. Hermes Agent framework — how it's integrated
The client spec requires the Hermes Agent framework from NousResearch. Integration is handled at two layers:

1. **Install** — `requirements.txt` declares `hermes-agent` as a git-URL dependency, so `pip install -r requirements.txt` fetches and installs the real framework. `git` must be on PATH (see section 1).

2. **Runtime binding** — `app/core/hermes_agent.py` performs a `try/except` import of `run_agent.AIAgent`. If the package loaded cleanly, its class is bound to the `AIAgent` symbol at module level, and every downstream agent (`PolymarketAgent`, `KalshiAgent`, `DataAgent`, `KronosAgent`, `KellyAgent`, `FeedbackAgent`) inherits from the real Hermes class. If the import raises *for any reason* (the framework has a known runtime error `ModuleNotFoundError: No module named 'agent.transports'` in some setups), a Hermes-compatible shim with the same interface (`chat()`, `reset()`, `quiet_mode`, `ephemeral_system_prompt`) is bound instead, keeping the pipeline operational while a single INFO log line explains why.

After starting the pipeline, the first log line from `app.core.hermes_agent` tells you which path is active:

- `Hermes Agent framework detected - using real nousresearch/hermes-agent.AIAgent` — real framework, production path.
- `Hermes Agent not available (<ExceptionType>: <reason>) - using compatible shim.` — shim path; the bracketed reason identifies the blocker.

Either path produces identical behaviour from every caller's perspective — the shim is a safety net for environments where the upstream package cannot be imported, not a replacement for it.

### 3. Configuration
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

Environment variables:

| Key | Required | Default / Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | Your OpenRouter key |
| `OPENROUTER_BASE_URL` | no | `https://openrouter.ai/api/v1` (override for a proxy) |
| `OPENROUTER_MODEL` | no | `deepseek/deepseek-chat-v3-0324:free` |
| `OPENAI_API_KEY` | no | Defaults to `OPENROUTER_API_KEY` (for Hermes) |
| `OPENAI_BASE_URL` | no | Defaults to `OPENROUTER_BASE_URL` |
| `APIFY_TOKEN` | yes | Your Apify token |
| `APIFY_BASE_URL` | no | `https://api.apify.com/v2` (override for a proxy) |
| `KALSHI_BASE_URL` | no | `https://api.kalshi.com/trade-api/v2` |
| `POLYMARKET_BASE_URL` | no | `https://gamma-api.polymarket.com` |
| `PREDICTION_HISTORY_PATH` | no | `./examples/prediction_history.jsonl` |
| `LOG_LEVEL` | no | `INFO` |
| `KRONOS` | no | `on` — set `off` to skip Kronos subprocess |
| `KRONOS_SCRIPT_PATH` | no | Defaults to `scripts/kronos_infer.py` |

Verify what the process sees (values are masked):
```bash
python scripts/verify_env.py
```

### 4. (Optional) Real Kronos inference
`KRONOS=on` is the default. The agent will call `scripts/kronos_infer.py` as a subprocess and fall back to an LLM proxy if Kronos isn't installed.

To install Kronos for real inference:
```bash
# Linux/macOS
bash scripts/setup_kronos.sh
# Windows
scripts\setup_kronos.bat
```
Set `KRONOS=off` to skip the subprocess entirely and always use the LLM proxy path.

### 5. Running the Agent
**Run once (CLI):**
```bash
python main.py --asset BTC
```

**Run in continuous loop (every 5 mins):**
```bash
python main.py --loop --interval 5
```

**Start the Monitoring API:**
```bash
python -m uvicorn app.api:app --reload
```
> On Windows, use `python -m uvicorn` rather than bare `uvicorn` — the `uvicorn.exe` shim isn't always on PATH after `pip install`.
- **API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- `GET /status` — pipeline health
- `GET /decisions` — decision history
- `POST /run` — trigger a full run (both assets)
- `POST /run/{asset}` — trigger a run for one asset (e.g. `BTC`)

> Use POST (not GET) for `/run` endpoints — GET will return 405.

### 6. Troubleshooting
- **Hermes resolves `https://openrouter.ai/api/v1` despite a custom base URL.** Clear stale provider config: delete `%USERPROFILE%\.hermes\provider.json` (Windows) or `~/.hermes/provider.json`, then re-run with `OPENAI_BASE_URL` set.
- **APIFY returns 404/503.** `DataAgent` falls back to synthetic OHLCV automatically; the pipeline keeps flowing. Check `APIFY_BASE_URL` if running through a proxy.
- **Kronos subprocess fails.** The fallback LLM proxy path will be used; see logs for the underlying error. Re-run `scripts/setup_kronos.*` if `.kronos/` is missing.

---

## 🧠 Scaling & "Outside the Box" Thinking

- **Multi-Asset Support**: Easily extended to any asset supported by Polymarket/Kalshi by adding to `SUPPORTED_ASSETS` in `app/pipeline.py`.
- **Arbitrage Hooks**: The `_arbitrage_signal` function implements logic to compare multi-platform sentiment, highlighting market inefficiencies.
- **Interface Compatibility**: The custom `AIAgent` implementation provides a stable, dependency-free interface for Hermes agents while maintaining full compatibility with the Hermes "Closed Loop" philosophy.

---

## 📝 Submission Metadata

See [`SUBMISSION.md`](./SUBMISSION.md) for the repo URL, demo video, and the exact APIFY token(s) used for the evaluation run.

Developed using **Antigravity AI Coding Assistant** as part of the CrowdWisdomTrading assessment.