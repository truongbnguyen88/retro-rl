"""Milestone-3 CheckpointManager tests + entropy-schedule callback tests.

CheckpointManager is the only piece of M3 that's pure logic — atomic writes,
JSON sidecars, last-K rotation, best-tracking. Tested here in isolation with
a stub model.

EntCoefLinearSchedule (added in v6) drives ``model.ent_coef`` from the rollout
hook. It is also pure logic given a stub model, so it sits here next to the
other isolated callback-class tests rather than the slow smoke run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retro_rl.training.callbacks import EntCoefLinearSchedule
from retro_rl.training.checkpoint import CheckpointManager


class _FakeModel:
    """Minimal stand-in for an SB3 PPO: ``save(path)`` plus the
    ``get_vec_normalize_env`` accessor the manager calls to decide whether to
    write a VecNormalize stats sidecar. ``vecnormalize`` lets a test inject a
    stub normalization env; default None mirrors a run without it.
    """

    def __init__(self, payload: str = "weights-v1", vecnormalize: object | None = None) -> None:
        self.payload = payload
        self._vecnormalize = vecnormalize

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.payload)

    def get_vec_normalize_env(self) -> object | None:
        return self._vecnormalize


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
# VecNormalize stats sidecar (resume-safe normalization)
# ---------------------------------------------------------------------------


class _FakeVecNormalize:
    """Stub with the one method the manager calls: ``save(path)``."""

    def __init__(self, payload: str = "vecnorm-stats") -> None:
        self.payload = payload

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.payload)


def test_no_pkl_written_without_vecnormalize(tmp_path: Path):
    mgr = CheckpointManager(root=tmp_path, run_name="r")
    mgr.save(_FakeModel(), step=10, eval_return=None)
    assert list((tmp_path / "r").glob("*.pkl")) == []


def test_pkl_sidecar_written_for_step_and_best(tmp_path: Path):
    mgr = CheckpointManager(root=tmp_path, run_name="r")
    model = _FakeModel("w", vecnormalize=_FakeVecNormalize("stats-1"))
    mgr.save(model, step=10, eval_return=5.0)  # also becomes best

    step_pkl = tmp_path / "r" / "step-10.pkl"
    best_pkl = tmp_path / "r" / "best.pkl"
    assert step_pkl.read_text() == "stats-1"
    assert best_pkl.read_text() == "stats-1"


def test_pkl_sidecar_pruned_with_zip(tmp_path: Path):
    mgr = CheckpointManager(root=tmp_path, run_name="r", keep_last_k=2)
    for s in (10, 20, 30, 40):
        mgr.save(_FakeModel(vecnormalize=_FakeVecNormalize()), step=s, eval_return=None)

    pkls = sorted(int(p.stem.split("-")[1]) for p in (tmp_path / "r").glob("step-*.pkl"))
    assert pkls == [30, 40]


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


# ---------------------------------------------------------------------------
# EntCoefLinearSchedule
# ---------------------------------------------------------------------------


class _StubLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, float]] = []

    def record(self, key: str, value: float) -> None:
        self.records.append((key, float(value)))


class _StubPPO:
    """Minimal PPO surface used by SB3's BaseCallback.

    SB3's ``BaseCallback.logger`` is a property that reads ``self.model.logger``,
    so the stub must expose one.
    """

    def __init__(self, logger: _StubLogger | None = None) -> None:
        self.num_timesteps = 0
        self.ent_coef: float = 0.0
        self.logger = logger or _StubLogger()


def _bind(cb: EntCoefLinearSchedule, model: _StubPPO):
    """Mimic SB3's BaseCallback.init_callback hand-off without importing SB3."""
    cb.model = model  # type: ignore[assignment]
    return cb


def test_ent_coef_schedule_endpoints():
    model = _StubPPO()
    cb = _bind(
        EntCoefLinearSchedule(initial=0.02, final=0.001, total_timesteps=1_000_000),
        model,
    )
    # At step 0 the callback should set ent_coef to the initial value.
    model.num_timesteps = 0
    cb._on_training_start()
    assert model.ent_coef == pytest.approx(0.02)

    # At full progress, ent_coef matches the final value.
    model.num_timesteps = 1_000_000
    cb._on_rollout_end()
    assert model.ent_coef == pytest.approx(0.001)

    # Past full progress, the schedule is clamped — never below the final.
    model.num_timesteps = 2_000_000
    cb._on_rollout_end()
    assert model.ent_coef == pytest.approx(0.001)


def test_ent_coef_schedule_midpoint_is_linear():
    model = _StubPPO()
    cb = _bind(
        EntCoefLinearSchedule(initial=0.02, final=0.0, total_timesteps=100),
        model,
    )
    model.num_timesteps = 25
    cb._on_rollout_end()
    assert model.ent_coef == pytest.approx(0.02 * 0.75)
    model.num_timesteps = 50
    cb._on_rollout_end()
    assert model.ent_coef == pytest.approx(0.01)
    model.num_timesteps = 75
    cb._on_rollout_end()
    assert model.ent_coef == pytest.approx(0.005)


def test_ent_coef_schedule_logs_to_tb():
    logger = _StubLogger()
    model = _StubPPO(logger=logger)
    cb = _bind(
        EntCoefLinearSchedule(initial=0.05, final=0.0, total_timesteps=10),
        model,
    )
    model.num_timesteps = 5
    cb._on_rollout_end()
    assert ("train/ent_coef", pytest.approx(0.025)) in logger.records


def test_ent_coef_schedule_rejects_invalid_args():
    with pytest.raises(ValueError):
        EntCoefLinearSchedule(initial=0.02, final=0.001, total_timesteps=0)
    with pytest.raises(ValueError):
        EntCoefLinearSchedule(initial=-0.01, final=0.0, total_timesteps=1000)
    with pytest.raises(ValueError):
        EntCoefLinearSchedule(initial=0.02, final=-0.001, total_timesteps=1000)
