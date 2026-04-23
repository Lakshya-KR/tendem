#!/bin/bash
# Auto-activate venv and run CrowdWisdomTrading

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run: python -m venv venv"
    exit 1
fi

source venv/Scripts/activate

# Run the application
python main.py "$@"
