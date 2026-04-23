"""
Kronos inference wrapper called as a subprocess by KronosAgent.

Contract with KronosAgent:
  stdout: single JSON object -> {"direction": "UP"|"DOWN", "confidence": <0..1>}
  exit 0 on success, non-zero + JSON {"error": "..."} on failure.
  KronosAgent falls back to its LLM proxy on any non-zero exit.

Usage:
  python scripts/kronos_infer.py --asset BTC --input path/to/closes.csv

Input CSV format (written by KronosAgent):
  timestamp,close
  2026-01-01T00:00:00,42100.5
  ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KRONOS_DIR = ROOT / ".kronos"


def _fail(msg: str, code: int = 2) -> None:
    print(json.dumps({"error": msg}))
    sys.exit(code)


def _load_closes(csv_path: Path) -> list[float]:
    closes: list[float] = []
    with csv_path.open() as f:
        header = f.readline().strip().split(",")
        try:
            close_idx = header.index("close")
        except ValueError:
            _fail("input CSV missing 'close' column")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= close_idx:
                continue
            try:
                closes.append(float(parts[close_idx]))
            except ValueError:
                continue
    return closes


def _run_kronos(asset: str, closes: list[float]) -> dict:
    """
    Attempt real Kronos inference. Requires `.kronos` cloned and
    model weights fetchable from HuggingFace. Any failure -> raise.
    """
    if not KRONOS_DIR.exists():
        raise RuntimeError(
            f"Kronos not installed at {KRONOS_DIR}. "
            "Run scripts/setup_kronos.sh (or .bat on Windows)."
        )

    # Make Kronos importable.
    sys.path.insert(0, str(KRONOS_DIR))

    import pandas as pd  # type: ignore
    from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model=model, tokenizer=tokenizer, device="cpu", max_context=512)

    window = closes[-512:] if len(closes) > 512 else closes
    if len(window) < 32:
        raise RuntimeError(f"not enough bars for inference: {len(window)}")

    df = pd.DataFrame({"close": window})
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    df["volume"] = 0.0

    # Kronos expects pandas Series (accesses .dt), not raw DatetimeIndex
    _x_idx = pd.date_range(end=pd.Timestamp.utcnow().tz_localize(None), periods=len(df), freq="5min")
    x_timestamp = pd.Series(_x_idx)
    y_timestamp = pd.Series(pd.date_range(start=_x_idx[-1] + pd.Timedelta("5min"), periods=1, freq="5min"))

    pred = predictor.predict(
        df=df[["open", "high", "low", "close", "volume"]],
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=1,
        T=1.0, top_p=0.9, sample_count=1,
    )

    last_close = float(window[-1])
    pred_close = float(pred["close"].iloc[0])
    direction = "UP" if pred_close >= last_close else "DOWN"
    # Confidence from relative move, clamped to [0.5, 0.99]
    rel = abs(pred_close - last_close) / max(abs(last_close), 1e-9)
    confidence = max(0.5, min(0.99, 0.5 + rel * 20.0))
    return {"direction": direction, "confidence": round(confidence, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--input", required=True, help="CSV with timestamp,close columns")
    args = ap.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        _fail(f"input CSV not found: {csv_path}")

    closes = _load_closes(csv_path)
    if len(closes) < 32:
        _fail(f"need at least 32 bars, got {len(closes)}")

    try:
        result = _run_kronos(args.asset, closes)
    except ModuleNotFoundError as exc:
        _fail(f"kronos deps missing ({exc}); run scripts/setup_kronos.(sh|bat)")
    except Exception as exc:  # noqa: BLE001 -- surface any runtime failure to caller
        _fail(f"kronos inference failed: {type(exc).__name__}: {exc}")

    print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()
