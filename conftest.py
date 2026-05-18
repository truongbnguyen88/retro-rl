"""Pytest top-level conftest.

Ensures ``src/`` is on ``sys.path`` regardless of whether the editable install's
``.pth`` file is currently honored. On macOS, the system occasionally sets the
``UF_HIDDEN`` flag on files inside the venv's ``site-packages`` (e.g. after
filesystem rescans), and CPython 3.12.5+ silently skips hidden ``.pth`` files
for security — which breaks ``import retro_rl`` until the flag is cleared.

This conftest sidesteps the issue entirely: it only mutates ``sys.path``, runs
once per pytest session, and adds nothing in production.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
