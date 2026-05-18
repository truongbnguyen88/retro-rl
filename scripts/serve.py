#!/usr/bin/env python
"""Serve the retro-rl FastAPI backend via uvicorn.

Usage
-----
    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8000
    python scripts/serve.py --checkpoint-root outputs/checkpoints \\
                            --tensorboard-root outputs/tensorboard

The backend is a read-only consumer of training artifacts plus an in-process
episode runtime. Frontend (Streamlit) talks to it over HTTP only.

No ``--reload`` flag: backend changes are infrequent (the seam is HTTP), and
hot-reload + the app factory + uvicorn worker spawn semantics interact
poorly. Restart the process by hand after backend edits.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# sys.path + PYTHONPATH shim: macOS auto-applies UF_HIDDEN to files in
# `.venv/lib/.../site-packages/`, and CPython 3.12.5+ skips hidden .pth files
# for security. Seed sys.path here so the editable install isn't required;
# propagate via PYTHONPATH so uvicorn-spawned workers inherit it (if/when we
# move past the single-process server).
_repo_root = Path(__file__).resolve().parents[1]
_src_path = str(_repo_root / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
_existing_pp = os.environ.get("PYTHONPATH", "")
if _src_path not in _existing_pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        _src_path + (os.pathsep + _existing_pp if _existing_pp else "")
    )

import uvicorn  # noqa: E402

from retro_rl.backend.api import create_app  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve the retro-rl FastAPI backend.")
    p.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1; use 0.0.0.0 to expose to LAN)",
    )
    p.add_argument(
        "--port", type=int, default=8000,
        help="Bind port (default: 8000)",
    )
    p.add_argument(
        "--checkpoint-root", type=Path, default=Path("outputs/checkpoints"),
        help="Directory with run subdirectories (default: outputs/checkpoints)",
    )
    p.add_argument(
        "--tensorboard-root", type=Path, default=Path("outputs/tensorboard"),
        help="Directory with TB log subdirectories (default: outputs/tensorboard)",
    )
    p.add_argument(
        "--agent-cache-size", type=int, default=4,
        help="LRU cap on the AgentRegistry (default: 4)",
    )
    p.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="uvicorn log level (default: info)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    app = create_app(
        checkpoint_root=args.checkpoint_root,
        tensorboard_root=args.tensorboard_root,
        agent_cache_size=args.agent_cache_size,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
