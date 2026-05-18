"""Structured logging factory.

:func:`get_logger` returns a configured stdlib logger:

* Console handler via ``rich.logging.RichHandler`` (pretty tracebacks, colored
  levels) when ``rich`` is importable; falls back to a plain StreamHandler
  otherwise.
* Optional rotating file handler at ``<run_dir>/run.log`` when ``run_dir`` is
  given. File logs are plain text (no ANSI), one line per record.

The factory is idempotent per-name: re-calling returns the same logger with
handlers re-bound rather than duplicated.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FILE_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def get_logger(
    name: str = "retro_rl",
    run_dir: Path | None = None,
    level: str = "INFO",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Reset to avoid duplicate handlers on re-import / re-call.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    logger.addHandler(_console_handler(level))

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            run_dir / "run.log", maxBytes=10_000_000, backupCount=3
        )
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(_FILE_FMT))
        logger.addHandler(fh)

    return logger


def _console_handler(level: str) -> logging.Handler:
    try:
        from rich.logging import RichHandler
    except ImportError:
        h: logging.Handler = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_FILE_FMT))
    else:
        h = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    h.setLevel(level)
    return h


__all__ = ["get_logger"]
