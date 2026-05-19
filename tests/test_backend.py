"""Milestone-5 backend tests.

Strategy
--------
We bypass the FastAPI lifespan by populating ``app.state`` ourselves and use
``app.dependency_overrides`` to inject fakes:

* :class:`_FakeResolver` — disk-free :class:`CheckpointResolver` stand-in
* :class:`_FakeAgent`    — implements the ``Agent`` protocol (just ``predict``)
* :class:`_FakeRuntime`  — fully-controllable :class:`EpisodeRuntime` substitute

For the happy-path ``POST /episodes`` test we monkey-patch
``retro_rl.backend.api.EpisodeRuntime`` so :func:`EpisodeRuntime.start` doesn't
build a real stable-retro env. The fake runtime is registered into a *real*
:class:`EpisodeRegistry`, so the registry's invariants (duplicate detection,
removal, close-on-shutdown) are exercised end-to-end.

The TB-metrics test writes a tiny ``events.out.tfevents.*`` file with
``tensorflow.summary`` to a tmp dir, then asserts the route parses it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from retro_rl.backend import api
from retro_rl.backend.api import (
    create_app,
    get_agents,
    get_episodes,
    get_resolver,
    get_started_at,
    get_tb_root,
)
from retro_rl.backend.inference import (
    AgentRegistry,
    CheckpointNotFoundError,
    EpisodeNotFoundError,
    EpisodeRegistry,
)
from retro_rl.backend.models import CheckpointInfo, EpisodeState
from retro_rl.utils.config import EnvConfig

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResolver:
    """Minimal stand-in for :class:`CheckpointResolver`.

    Mirrors the real resolver's failure modes so route tests exercise the
    full validation path: ill-formed ids and missing-on-disk both raise
    :class:`CheckpointNotFoundError`.
    """

    def __init__(
        self,
        items: list[CheckpointInfo] | None = None,
        root: Path | None = None,
        env_cfg: EnvConfig | None = None,
        bad_id: str = "no_such_run/best",
    ) -> None:
        self._items = items or []
        self.root = root or Path("outputs/checkpoints")
        self._env_cfg = env_cfg or EnvConfig()
        self._bad_id = bad_id

    def list_all(self) -> list[CheckpointInfo]:
        return list(self._items)

    def resolve_path(self, checkpoint_id: str) -> Path:
        # Reuse the real id-format validator so malformed ids fail like prod.
        from retro_rl.backend.inference import _parse_checkpoint_id

        _parse_checkpoint_id(checkpoint_id)
        if checkpoint_id == self._bad_id:
            raise CheckpointNotFoundError(f"checkpoint not found on disk: {checkpoint_id}")
        return Path(f"/fake/{checkpoint_id}.zip")

    def env_config_for(self, checkpoint_id: str) -> EnvConfig:
        from retro_rl.backend.inference import _parse_checkpoint_id

        _parse_checkpoint_id(checkpoint_id)
        if checkpoint_id == self._bad_id:
            raise CheckpointNotFoundError(f"checkpoint not found on disk: {checkpoint_id}")
        return self._env_cfg


class _FakeAgent:
    """Implements the Agent protocol used by EpisodeRuntime."""

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[int, None]:
        return 0, None


class _FakeAgentRegistry:
    """No-load, no-cache substitute for :class:`AgentRegistry`."""

    def __init__(self) -> None:
        self.agent = _FakeAgent()
        self.get_calls: list[str] = []

    def get(self, checkpoint_id: str) -> _FakeAgent:
        self.get_calls.append(checkpoint_id)
        return self.agent

    def clear(self) -> None:
        pass


class _FakeRuntime:
    """Stand-in for :class:`EpisodeRuntime`.

    Steps a synthetic episode of length ``ends_after`` then sets ``terminated``.
    Each step adds 1.0 to total reward. Frame is a fixed RGB stub.
    """

    def __init__(
        self,
        *,
        episode_id: str = "ep-fake-1",
        checkpoint_id: str = "ppo_x/best",
        ends_after: int = 3,
    ) -> None:
        self.episode_id = episode_id
        self.checkpoint_id = checkpoint_id
        self._ends_after = ends_after
        self._step = 0
        self._reward = 0.0
        self._terminated = False
        self._truncated = False
        self._closed = False
        self._started_at = "2026-05-18T13:00:00+00:00"

    @classmethod
    def start(
        cls,
        *,
        checkpoint_id: str,
        agent: Any,
        env_cfg: Any,
        seed: int | None = None,
        deterministic: bool = True,
        max_steps: int | None = None,
    ) -> _FakeRuntime:
        # signature mirrors EpisodeRuntime.start so the route can call it unchanged
        return cls(checkpoint_id=checkpoint_id)

    def step(self) -> None:
        if self._terminated or self._truncated:
            from retro_rl.backend.inference import EpisodeFinishedError

            raise EpisodeFinishedError("done")
        self._step += 1
        self._reward += 1.0
        if self._step >= self._ends_after:
            self._terminated = True

    def state(self) -> EpisodeState:
        return EpisodeState(
            episode_id=self.episode_id,
            step=self._step,
            total_reward=self._reward,
            terminated=self._terminated,
            truncated=self._truncated,
            done=self._terminated or self._truncated,
            last_action=0 if self._step > 0 else None,
            info={},
        )

    def frame_png(self) -> bytes:
        # Encode a 4×4 RGB stub via the same path the real runtime uses.
        from retro_rl.backend.inference import _frame_to_png

        return _frame_to_png(np.zeros((4, 4, 3), dtype=np.uint8))

    def close(self) -> None:
        self._closed = True

    @property
    def done(self) -> bool:
        return self._terminated or self._truncated

    @property
    def started_at(self) -> str:
        return self._started_at


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_resolver() -> _FakeResolver:
    items = [
        CheckpointInfo(
            id="ppo_x/best",
            run_name="ppo_x",
            kind="best",
            step=200_000,
            eval_return=42.0,
            timestamp="2026-05-18T13:00:00+00:00",
            path="outputs/checkpoints/ppo_x/best.zip",
        ),
        CheckpointInfo(
            id="ppo_x/step-100000",
            run_name="ppo_x",
            kind="step",
            step=100_000,
            eval_return=20.0,
            timestamp="2026-05-18T12:00:00+00:00",
            path="outputs/checkpoints/ppo_x/step-100000.zip",
        ),
        CheckpointInfo(
            id="ppo_y/step-50000",
            run_name="ppo_y",
            kind="step",
            step=50_000,
            eval_return=None,
            timestamp="2026-05-18T11:00:00+00:00",
            path="outputs/checkpoints/ppo_y/step-50000.zip",
        ),
    ]
    return _FakeResolver(items=items)


@pytest.fixture
def fake_agents() -> _FakeAgentRegistry:
    return _FakeAgentRegistry()


@pytest.fixture
def episodes_registry() -> EpisodeRegistry:
    """Real registry — exercises duplicate detection, removal, close_all."""
    return EpisodeRegistry()


@pytest.fixture
def client(
    tmp_path: Path,
    fake_resolver: _FakeResolver,
    fake_agents: _FakeAgentRegistry,
    episodes_registry: EpisodeRegistry,
) -> TestClient:
    """Build a TestClient with all dependencies overridden.

    Lifespan is bypassed because the overrides supply the registries; we
    manually populate ``app.state.started_at`` so ``/health`` works.
    """
    app = create_app(checkpoint_root=tmp_path / "ckpt", tensorboard_root=tmp_path / "tb")
    app.state.started_at = time.monotonic()
    app.dependency_overrides[get_resolver] = lambda: fake_resolver
    app.dependency_overrides[get_agents] = lambda: fake_agents
    app.dependency_overrides[get_episodes] = lambda: episodes_registry
    app.dependency_overrides[get_started_at] = lambda: app.state.started_at
    app.dependency_overrides[get_tb_root] = lambda: tmp_path / "tb"
    # Don't enter lifespan — overrides cover all dependencies and we don't
    # want create_app's CheckpointResolver to walk a tmp dir we never populated.
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["version"], str)
    assert body["uptime_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# /checkpoints
# ---------------------------------------------------------------------------


def test_checkpoints_returns_resolver_items(client: TestClient):
    r = client.get("/checkpoints")
    assert r.status_code == 200
    data = r.json()
    assert [c["id"] for c in data["checkpoints"]] == [
        "ppo_x/best",
        "ppo_x/step-100000",
        "ppo_y/step-50000",
    ]


def test_checkpoints_empty(tmp_path: Path):
    app = create_app(checkpoint_root=tmp_path, tensorboard_root=tmp_path)
    app.state.started_at = time.monotonic()
    app.dependency_overrides[get_resolver] = lambda: _FakeResolver(items=[])
    app.dependency_overrides[get_agents] = lambda: _FakeAgentRegistry()
    app.dependency_overrides[get_episodes] = lambda: EpisodeRegistry()
    app.dependency_overrides[get_started_at] = lambda: app.state.started_at
    app.dependency_overrides[get_tb_root] = lambda: tmp_path
    c = TestClient(app)
    r = c.get("/checkpoints")
    assert r.status_code == 200
    assert r.json() == {"checkpoints": []}


# ---------------------------------------------------------------------------
# /runs
# ---------------------------------------------------------------------------


def test_runs_aggregates_per_run(client: TestClient, tmp_path: Path):
    # The resolver's root is used for config_snapshot_path detection; create one
    # for ppo_x but not for ppo_y to exercise both branches.
    ckpt_root = tmp_path / "ckpt"
    (ckpt_root / "ppo_x").mkdir(parents=True)
    (ckpt_root / "ppo_x" / "config_snapshot.json").write_text("{}")
    # Rebuild fake resolver pointed at the ckpt_root so the config-snapshot
    # existence check resolves correctly for ppo_x (but not ppo_y).
    new_resolver = _FakeResolver(
        items=[
            CheckpointInfo(
                id="ppo_x/best",
                run_name="ppo_x",
                kind="best",
                step=200_000,
                eval_return=42.0,
                timestamp="t",
                path=str(ckpt_root / "ppo_x" / "best.zip"),
            ),
            CheckpointInfo(
                id="ppo_x/step-100000",
                run_name="ppo_x",
                kind="step",
                step=100_000,
                eval_return=20.0,
                timestamp="t",
                path=str(ckpt_root / "ppo_x" / "step-100000.zip"),
            ),
            CheckpointInfo(
                id="ppo_y/step-50000",
                run_name="ppo_y",
                kind="step",
                step=50_000,
                eval_return=None,
                timestamp="t",
                path=str(ckpt_root / "ppo_y" / "step-50000.zip"),
            ),
        ],
        root=ckpt_root,
    )
    client.app.dependency_overrides[get_resolver] = lambda: new_resolver

    r = client.get("/runs")
    assert r.status_code == 200
    runs = {row["run_name"]: row for row in r.json()["runs"]}
    assert runs["ppo_x"]["has_best"] is True
    assert runs["ppo_x"]["best_return"] == 42.0
    assert runs["ppo_x"]["latest_step"] == 100_000  # max of step kind only
    assert runs["ppo_x"]["checkpoint_count"] == 2
    assert runs["ppo_x"]["config_snapshot_path"].endswith("config_snapshot.json")

    assert runs["ppo_y"]["has_best"] is False
    assert runs["ppo_y"]["best_return"] is None
    assert runs["ppo_y"]["latest_step"] == 50_000
    assert runs["ppo_y"]["config_snapshot_path"] is None  # file doesn't exist


# ---------------------------------------------------------------------------
# /runs/{name}/metrics — synth TB events + parse
# ---------------------------------------------------------------------------


def _write_tb_scalars(log_dir: Path, scalars: dict[str, list[tuple[int, float]]]) -> None:
    """Emit minimal TB scalar events. Uses torch.utils.tensorboard.SummaryWriter
    (already a dep via stable-baselines3 → torch)."""
    from torch.utils.tensorboard import SummaryWriter

    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    try:
        for tag, points in scalars.items():
            for step, value in points:
                writer.add_scalar(tag, value, global_step=step)
    finally:
        writer.flush()
        writer.close()


def test_metrics_route_parses_scalars(client: TestClient, tmp_path: Path):
    tb_root = tmp_path / "tb"
    _write_tb_scalars(
        tb_root / "ppo_x_1",
        {"eval/mean_return": [(1000, 5.0), (2000, 10.0)], "train/loss": [(1000, 0.5)]},
    )
    client.app.dependency_overrides[get_tb_root] = lambda: tb_root

    r = client.get("/runs/ppo_x/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["run_name"] == "ppo_x"
    series = {s["name"]: s["points"] for s in body["series"]}
    assert "eval/mean_return" in series
    assert "train/loss" in series
    assert series["eval/mean_return"] == [
        {"step": 1000, "value": 5.0},
        {"step": 2000, "value": 10.0},
    ]


def test_metrics_route_404_when_no_tb_dir(client: TestClient, tmp_path: Path):
    # tb_root must exist but contain no matching subdir for the "no log dir" branch.
    tb_root = tmp_path / "tb_empty"
    tb_root.mkdir()
    client.app.dependency_overrides[get_tb_root] = lambda: tb_root
    r = client.get("/runs/missing/metrics")
    assert r.status_code == 404
    assert "no TB log dir" in r.json()["detail"]


def test_metrics_route_404_when_tb_root_absent(client: TestClient, tmp_path: Path):
    """Distinct 404 branch: tb_root doesn't exist at all."""
    client.app.dependency_overrides[get_tb_root] = lambda: tmp_path / "never_created"
    r = client.get("/runs/anything/metrics")
    assert r.status_code == 404
    assert "does not exist" in r.json()["detail"]


def test_metrics_route_picks_latest_log_dir(client: TestClient, tmp_path: Path):
    """When two TB dirs exist for the same run, pick the most-recently-modified."""
    tb_root = tmp_path / "tb"
    _write_tb_scalars(tb_root / "ppo_x_1", {"eval/mean_return": [(1, 1.0)]})
    time.sleep(0.05)
    _write_tb_scalars(tb_root / "ppo_x_2", {"eval/mean_return": [(2, 2.0)]})
    client.app.dependency_overrides[get_tb_root] = lambda: tb_root

    r = client.get("/runs/ppo_x/metrics")
    assert r.status_code == 200
    series = {s["name"]: s["points"] for s in r.json()["series"]}
    assert series["eval/mean_return"] == [{"step": 2, "value": 2.0}]


# ---------------------------------------------------------------------------
# POST /episodes — happy path + error branches
# ---------------------------------------------------------------------------


def test_post_episodes_happy_path(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """End-to-end: route calls EpisodeRuntime.start (monkey-patched to fake)."""
    monkeypatch.setattr(api, "EpisodeRuntime", _FakeRuntime)
    r = client.post("/episodes", json={"checkpoint_id": "ppo_x/best", "seed": 42})
    assert r.status_code == 201
    body = r.json()
    assert body["checkpoint_id"] == "ppo_x/best"
    assert body["episode_id"]
    assert body["started_at"]


def test_post_episodes_unknown_checkpoint(client: TestClient):
    r = client.post("/episodes", json={"checkpoint_id": "no_such_run/best"})
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "no_such_run" in detail  # specific id surfaces in error message


def test_post_episodes_bad_id_format(client: TestClient):
    r = client.post("/episodes", json={"checkpoint_id": "no-slash"})
    assert r.status_code == 404
    assert "must be" in r.json()["detail"]


def test_post_episodes_extra_field_rejected(client: TestClient):
    r = client.post("/episodes", json={"checkpoint_id": "ppo_x/best", "bogus": True})
    assert r.status_code == 422


def test_post_episodes_missing_checkpoint_id(client: TestClient):
    r = client.post("/episodes", json={"seed": 1})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /episodes/{id}/state + /frame + DELETE
# ---------------------------------------------------------------------------


def test_get_state_missing_404(client: TestClient):
    r = client.get("/episodes/unknown/state")
    assert r.status_code == 404


def test_get_state_returns_snapshot(
    client: TestClient,
    episodes_registry: EpisodeRegistry,
):
    ep = _FakeRuntime(episode_id="ep-1")
    episodes_registry.register(ep)
    r = client.get("/episodes/ep-1/state")
    assert r.status_code == 200
    body = r.json()
    assert body["episode_id"] == "ep-1"
    assert body["step"] == 0
    assert body["done"] is False
    assert body["last_action"] is None


def test_get_frame_advances_state(
    client: TestClient,
    episodes_registry: EpisodeRegistry,
):
    ep = _FakeRuntime(episode_id="ep-2", ends_after=3)
    episodes_registry.register(ep)

    # 1st frame: step 0 → 1
    r = client.get("/episodes/ep-2/frame")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    state = client.get("/episodes/ep-2/state").json()
    assert state["step"] == 1
    assert state["total_reward"] == 1.0
    assert state["done"] is False


def test_get_frame_idempotent_after_done(
    client: TestClient,
    episodes_registry: EpisodeRegistry,
):
    ep = _FakeRuntime(episode_id="ep-3", ends_after=2)
    episodes_registry.register(ep)
    # Drive to done.
    client.get("/episodes/ep-3/frame")
    client.get("/episodes/ep-3/frame")
    state = client.get("/episodes/ep-3/state").json()
    assert state["done"] is True
    # Subsequent frames return last image, no error, no extra step.
    r = client.get("/episodes/ep-3/frame")
    assert r.status_code == 200
    state2 = client.get("/episodes/ep-3/state").json()
    assert state2["step"] == state["step"]


def test_get_frame_missing_404(client: TestClient):
    r = client.get("/episodes/unknown/frame")
    assert r.status_code == 404


def test_delete_episode_releases(
    client: TestClient,
    episodes_registry: EpisodeRegistry,
):
    ep = _FakeRuntime(episode_id="ep-4")
    episodes_registry.register(ep)
    assert "ep-4" in episodes_registry.list_ids()

    r = client.delete("/episodes/ep-4")
    assert r.status_code == 204
    assert "ep-4" not in episodes_registry.list_ids()
    assert ep._closed is True

    # Idempotent re-delete? Spec is 404 on missing — verify.
    r2 = client.delete("/episodes/ep-4")
    assert r2.status_code == 404


# ---------------------------------------------------------------------------
# AgentRegistry — exercised separately since its semantics are independent
# ---------------------------------------------------------------------------


def test_agent_registry_lru_eviction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """LRU cap evicts oldest entry when capacity is exceeded."""
    from retro_rl.backend import inference as inf

    loads: list[str] = []

    class _StubResolver:
        def resolve_path(self, cid: str) -> Path:
            return tmp_path / f"{cid.replace('/', '_')}.zip"

    def _fake_load(path: str, env=None):
        loads.append(path)
        return object()

    monkeypatch.setattr(inf, "PPO", type("PPO", (), {"load": staticmethod(_fake_load)}))

    reg = AgentRegistry(_StubResolver(), max_cached=2)  # type: ignore[arg-type]
    reg.get("a/best")
    reg.get("b/best")
    reg.get("c/best")  # evicts 'a' (least-recently-used)

    cached = reg.cached_ids()
    assert "a/best" not in cached
    assert "b/best" in cached
    assert "c/best" in cached
    assert len(loads) == 3  # each was a miss


def test_agent_registry_cache_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from retro_rl.backend import inference as inf

    loads: list[str] = []

    class _StubResolver:
        def resolve_path(self, cid: str) -> Path:
            return tmp_path / f"{cid.replace('/', '_')}.zip"

    def _fake_load(path: str, env=None):
        loads.append(path)
        return object()

    monkeypatch.setattr(inf, "PPO", type("PPO", (), {"load": staticmethod(_fake_load)}))

    reg = AgentRegistry(_StubResolver(), max_cached=4)  # type: ignore[arg-type]
    reg.get("a/best")
    reg.get("a/best")  # hit
    reg.get("a/best")  # hit
    assert len(loads) == 1


# ---------------------------------------------------------------------------
# EpisodeRegistry semantics
# ---------------------------------------------------------------------------


def test_episode_registry_duplicate_id_raises():
    reg = EpisodeRegistry()
    ep = _FakeRuntime(episode_id="dup")
    reg.register(ep)
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(_FakeRuntime(episode_id="dup"))


def test_episode_registry_missing_raises():
    reg = EpisodeRegistry()
    with pytest.raises(EpisodeNotFoundError):
        reg.get("missing")


def test_episode_registry_close_all():
    reg = EpisodeRegistry()
    eps = [_FakeRuntime(episode_id=f"e{i}") for i in range(3)]
    for ep in eps:
        reg.register(ep)
    reg.close_all()
    assert reg.list_ids() == []
    assert all(ep._closed for ep in eps)
