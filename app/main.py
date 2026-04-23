"""
app/main.py
────────────
Thin alias for the root `main.py` entry-point. Exists so that both
invocations work identically:

    python main.py --asset BTC
    python -m app.main --asset BTC

The real implementation lives at the repository root in main.py; this
module simply forwards to it. Argparse reads from sys.argv directly, so
no argument marshalling is needed here.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root is the parent of this file's `app/` directory. Prepend it
# to sys.path so the root-level `main` module is importable regardless of
# the current working directory at invocation time.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from main import main as _root_main  # type: ignore  # noqa: E402


def main() -> int:
    return _root_main()


if __name__ == "__main__":
    sys.exit(main())
