"""tests/conftest.py – Pytest configuration."""
import sys
from pathlib import Path

# Ensure the project root is on the path so `app` imports work without install
sys.path.insert(0, str(Path(__file__).parent.parent))
