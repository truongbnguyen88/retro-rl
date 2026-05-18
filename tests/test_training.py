"""Milestone-3 CheckpointManager tests.

CheckpointManager is the only piece of M3 that's pure logic — atomic writes,
JSON sidecars, last-K rotation, best-tracking. Tested here in isolation with
a stub model.

The callbacks and trainer are exercised end-to-end by the smoke run
(``python scripts/train.py --config configs/ppo_smoke.yaml``), which produces
disk artifacts that prove the wiring works. No separate slow integration
test in this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retro_rl.training.checkpoint import CheckpointManager


class _FakeModel:
    """Minimal stand-in for an SB3 PPO: only ``save(path)`` is exercised."""

    def __init__(self, payload: str = "weights-v1") -> None:
        self.payload = payload

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.payload)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_creates_run_subdirectory(tmp_path: Path):
    CheckpointManager(root=tmp_path, run_name="abc")
    assert (tmp_path / "abc").is_dir()


def test_rejects_zero_keep_last_k(tmp_path: Path):
    with pytest.raises(ValueError):
        CheckpointManager(root=tmp_path, run_name="abc", keep_last_k=0)


# ---------------------------------------------------------------------------
# Save: files + sidecar
# ---------------------------------------------------------------------------


def test_save_writes_zip_and_sidecar(tmp_path: Path):
    mgr = CheckpointManager(root=tmp_path, run_name="r", keep_last_k=5)
    out = mgr.save(_FakeModel(), step=100, eval_return=None)

    assert out == tmp_path / "r" / "step-100.zip"
    assert out.exists()
    side = out.with_suffix(".json")
    assert side.exists()
    meta = json.loads(side.read_text())
    assert meta["run_name"] == "r"
    assert meta["step"] == 100
    assert meta["eval_return"] is None
    assert meta["kind"] == "step"
    assert "timestamp" in meta


def test_save_records_config_snapshot_path(tmp_path: Path):
    snap = tmp_path / "snapshot.json"
    snap.write_text("{}")
    mgr = CheckpointManager(root=tmp_path, run_name="r", config_snapshot_path=snap)
    mgr.save(_FakeModel(), step=10)
    meta = json.loads((tmp_path / "r" / "step-10.json").read_text())
    assert meta["config_snapshot_path"] == str(snap)


def test_save_leaves_no_tmp_files(tmp_path: Path):
    mgr = CheckpointManager(root=tmp_path, run_name="r")
    mgr.save(_FakeModel(), step=10, eval_return=1.0)
    leftover = list((tmp_path / "r").glob("*.tmp"))
    assert leftover == []


# ---------------------------------------------------------------------------
# Best-tracking
# ---------------------------------------------------------------------------


def test_best_updates_on_higher_return(tmp_path: Path):
    mgr = CheckpointManager(root=tmp_path, run_name="r")
    mgr.save(_FakeModel("v1"), step=10, eval_return=5.0)
    mgr.save(_FakeModel("v2"), step=20, eval_return=10.0)

    best = mgr.best()
    assert best is not None
    assert best.name == "best.zip"
    assert best.read_text() == "v2"
    assert mgr.best_return == 10.0
    meta = json.loads(mgr.dir.joinpath("best.json").read_text())
    assert meta["kind"] == "best"
    assert meta["step"] == 20


def test_best_unchanged_on_lower_return(tmp_path: Path):
    mgr = CheckpointManager(root=tmp_path, run_name="r")
    mgr.save(_FakeModel("v1"), step=10, eval_return=10.0)
    mgr.save(_FakeModel("v2"), step=20, eval_return=5.0)

    assert mgr.best().read_text() == "v1"
    assert mgr.best_return == 10.0


def test_best_not_written_when_eval_return_none(tmp_path: Path):
    mgr = CheckpointManager(root=tmp_path, run_name="r")
    mgr.save(_FakeModel(), step=10, eval_return=None)
    assert mgr.best() is None
    assert mgr.best_return is None


def test_best_disabled_when_keep_best_false(tmp_path: Path):
    mgr = CheckpointManager(root=tmp_path, run_name="r", keep_best=False)
    mgr.save(_FakeModel(), step=10, eval_return=100.0)
    assert mgr.best() is None


def test_best_return_restored_from_existing_sidecar(tmp_path: Path):
    mgr1 = CheckpointManager(root=tmp_path, run_name="r")
    mgr1.save(_FakeModel("v1"), step=10, eval_return=42.0)

    mgr2 = CheckpointManager(root=tmp_path, run_name="r")
    assert mgr2.best_return == 42.0
    # A new save below the restored best must not overwrite best.zip.
    mgr2.save(_FakeModel("v2"), step=20, eval_return=1.0)
    assert mgr2.best().read_text() == "v1"


# ---------------------------------------------------------------------------
# Last-K rotation
# ---------------------------------------------------------------------------


def test_prunes_to_keep_last_k_step_checkpoints(tmp_path: Path):
    mgr = CheckpointManager(root=tmp_path, run_name="r", keep_last_k=2)
    for s in (10, 20, 30, 40):
        mgr.save(_FakeModel(), step=s, eval_return=None)

    steps_on_disk = sorted(int(p.stem.split("-")[1]) for p in (tmp_path / "r").glob("step-*.zip"))
    assert steps_on_disk == [30, 40]
    # Sidecars pruned in lockstep.
    sides = sorted(int(p.stem.split("-")[1]) for p in (tmp_path / "r").glob("step-*.json"))
    assert sides == [30, 40]


def test_prune_excludes_best_zip(tmp_path: Path):
    mgr = CheckpointManager(root=tmp_path, run_name="r", keep_last_k=1)
    mgr.save(_FakeModel("v1"), step=10, eval_return=100.0)  # becomes best
    for s in (20, 30, 40):
        mgr.save(_FakeModel(), step=s, eval_return=None)

    # best.zip survives all the way through pruning.
    assert mgr.best() is not None
    assert mgr.best().read_text() == "v1"


# ---------------------------------------------------------------------------
# Latest accessor
# ---------------------------------------------------------------------------


def test_latest_returns_highest_step(tmp_path: Path):
    mgr = CheckpointManager(root=tmp_path, run_name="r", keep_last_k=5)
    # Out-of-order save calls: highest step wins regardless.
    for s in (50, 10, 30, 20, 40):
        mgr.save(_FakeModel(), step=s, eval_return=None)
    latest = mgr.latest()
    assert latest is not None
    assert latest.name == "step-50.zip"


def test_latest_none_when_no_checkpoints(tmp_path: Path):
    mgr = CheckpointManager(root=tmp_path, run_name="r")
    assert mgr.latest() is None
    assert mgr.best() is None
