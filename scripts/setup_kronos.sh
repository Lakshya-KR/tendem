#!/usr/bin/env bash
set -euo pipefail
# Simple installer for Kronos repo in a subdir .kronos
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KRONOS_DIR="$ROOT_DIR/.kronos"
if [ ! -d "$KRONOS_DIR" ]; then
  git clone --depth 1 https://github.com/shiyu-coder/Kronos "$KRONOS_DIR"
fi
python -m pip install --upgrade pip
# Minimal deps; full requirements may be heavy depending on hardware
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu || true
python -m pip install -r "$KRONOS_DIR/requirements.txt" || true

echo "Kronos installed under $KRONOS_DIR"
