"""Checkpoint manager with atomic save, JSON sidecars, last-K pruning, best-tracking.

One :class:`CheckpointManager` per run. Files written under ``<dir>/<run_name>/``:

* ``step-<N>.zip`` + ``step-<N>.json`` — periodic checkpoints (last-K kept).
* ``best.zip`` + ``best.json`` — best-by-eval-return (overwritten in place).
* ``config_snapshot.json`` — written separately by the trainer; sidecars only
  reference it by path.

Atomicity: every write goes to ``<path>.tmp`` first, then ``os.replace`` swaps
it into place. Readers (backend) never see a half-written zip.

Resume: on ``__init__``, scans the directory and restores ``_best_return`` from
``best.json`` if present. Mid-run crashes leave the manager state recoverable
from disk alone.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_save_zip(model: PPO, path: Path) -> None:
    """Save SB3 model to ``path`` atomically via tmp + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    model.save(str(tmp))
    os.replace(tmp, path)


def _atomic_save_vecnormalize(model: PPO, path: Path) -> bool:
    """Save VecNormalize running stats next to a checkpoint, atomically.

    Returns True if stats were written (i.e. the model's env is a VecNormalize),
    False otherwise. The ``.pkl`` is a sidecar to ``step-<N>.zip`` / ``best.zip``
    so resume can restore the exact reward-normalization state for that
    checkpoint. ``get_vec_normalize_env`` only walks wrapper references (no
    subprocess IPC), so this is safe even after the train VecEnv is closed.
    """
    vn = model.get_vec_normalize_env()
    if vn is None:
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    vn.save(str(tmp))
    os.replace(tmp, path)
    return True


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


class CheckpointManager:
    """Manages periodic + best checkpoints for one training run.

    Parameters
    ----------
    root
        Parent directory (e.g. ``outputs/checkpoints``). Run subdirectory is
        created as ``root / run_name``.
    run_name
        Identifies the run; used as the subdirectory name.
    keep_last_k
        Retain only the K most recent ``step-<N>.zip`` files; older are deleted.
        ``best.zip`` is excluded from this rotation.
    keep_best
        If True, ``save`` will write ``best.zip`` whenever ``eval_return``
        beats the current best.
    config_snapshot_path
        Path to the run's config snapshot JSON (written separately by the
        trainer). Recorded in every sidecar for reproducibility.
    """

    def __init__(
        self,
        root: Path,
        run_name: str,
        keep_last_k: int = 5,
        keep_best: bool = True,
        config_snapshot_path: Path | None = None,
    ) -> None:
        if keep_last_k < 1:
            raise ValueError(f"keep_last_k must be >= 1, got {keep_last_k}")
        self.run_name = run_name
        self.keep_last_k = keep_last_k
        self.keep_best = keep_best
        self.config_snapshot_path = config_snapshot_path

        self.dir = Path(root) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)

        self._best_return: float | None = self._read_best_return()

    # ------------------------------------------------------------------ I/O

    def save(
        self,
        model: PPO,
        step: int,
        eval_return: float | None = None,
    ) -> Path:
        """Write step checkpoint + sidecar; optionally update ``best``.

        Returns the path to the ``step-<N>.zip`` just written.
        """
        zip_path = self.dir / f"step-{step}.zip"
        _atomic_save_zip(model, zip_path)
        _atomic_save_vecnormalize(model, zip_path.with_suffix(".pkl"))
        _atomic_write_json(
            zip_path.with_suffix(".json"),
            self._sidecar_payload(step, eval_return, kind="step"),
        )

        if self.keep_best and eval_return is not None and self._is_new_best(eval_return):
            self._write_best(model, step, eval_return)

        self._prune()
        return zip_path

    def latest(self) -> Path | None:
        """Path to the highest-step ``step-<N>.zip``, or None if none exist."""
        ckpts = self._step_checkpoints()
        return ckpts[-1] if ckpts else None

    def best(self) -> Path | None:
        """Path to ``best.zip`` if it exists, else None."""
        p = self.dir / "best.zip"
        return p if p.exists() else None

    @property
    def best_return(self) -> float | None:
        return self._best_return

    # -------------------------------------------------------------- helpers

    def _sidecar_payload(self, step: int, eval_return: float | None, kind: str) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "step": step,
            "eval_return": eval_return,
            "kind": kind,
            "config_snapshot_path": (
                str(self.config_snapshot_path) if self.config_snapshot_path is not None else None
            ),
            "timestamp": _utc_iso(),
        }

    def _is_new_best(self, eval_return: float) -> bool:
        return self._best_return is None or eval_return > self._best_return

    def _write_best(self, model: PPO, step: int, eval_return: float) -> None:
        best_zip = self.dir / "best.zip"
        _atomic_save_zip(model, best_zip)
        _atomic_save_vecnormalize(model, best_zip.with_suffix(".pkl"))
        _atomic_write_json(
            self.dir / "best.json",
            self._sidecar_payload(step, eval_return, kind="best"),
        )
        self._best_return = eval_return

    def _read_best_return(self) -> float | None:
        meta = self.dir / "best.json"
        if not meta.exists():
            return None
        try:
            data = json.loads(meta.read_text())
            ret = data.get("eval_return")
            return float(ret) if ret is not None else None
        except (json.JSONDecodeError, ValueError, OSError):
            # Corrupt sidecar — ignore, treat as no prior best.
            return None

    def _step_checkpoints(self) -> list[Path]:
        """Return ``step-*.zip`` paths sorted ascending by step."""
        files = []
        for p in self.dir.glob("step-*.zip"):
            try:
                step = int(p.stem.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            files.append((step, p))
        files.sort()
        return [p for _, p in files]

    def _prune(self) -> None:
        ckpts = self._step_checkpoints()
        excess = len(ckpts) - self.keep_last_k
        if excess <= 0:
            return
        for old in ckpts[:excess]:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)
            old.with_suffix(".pkl").unlink(missing_ok=True)


__all__ = ["CheckpointManager"]
