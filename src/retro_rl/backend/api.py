"""FastAPI app for the retro-rl backend.

Routes (per TASKS.md M5)
------------------------
* ``GET  /health``                          — liveness + version + uptime
* ``GET  /checkpoints``                     — flat list of all checkpoints on disk
* ``POST /episodes``                        — start a rollout (envs constructed eagerly)
* ``GET  /episodes/{id}/state``             — snapshot (does not advance)
* ``GET  /episodes/{id}/frame``             — advance one env step + return PNG
* ``DELETE /episodes/{id}``                 — close env, drop from registry
* ``GET  /runs``                            — per-run summary (best/latest)
* ``GET  /runs/{run_name}/metrics``         — all scalar series from the latest TB log dir

Design
------
Registries (resolver / agents / episodes) live on ``app.state`` so each
:func:`create_app` call gets fresh state — tests can spin a clean app per case.
Routes pull them via FastAPI's ``Depends`` so tests can also use
``app.dependency_overrides`` to inject mocks.

Async-ness: routes are sync. The hot path is CPU-bound (PPO predict, env.step,
PNG encode) — FastAPI runs sync routes in a threadpool, which is exactly what
we want here. No coroutines, no event-loop blocking concerns.
"""

from __future__ import annotations

import contextlib
import logging
import time
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from retro_rl.backend.inference import (
    AgentRegistry,
    CheckpointNotFoundError,
    CheckpointResolver,
    EpisodeFinishedError,
    EpisodeNotFoundError,
    EpisodeRegistry,
    EpisodeRuntime,
)
from retro_rl.backend.models import (
    CheckpointInfo,
    CheckpointList,
    EpisodeStartRequest,
    EpisodeStartResponse,
    EpisodeState,
    HealthResponse,
    MetricPoint,
    MetricSeries,
    RunInfo,
    RunList,
    RunMetrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependencies (Request → singleton on app.state)
# ---------------------------------------------------------------------------


def get_resolver(request: Request) -> CheckpointResolver:
    return request.app.state.resolver


def get_agents(request: Request) -> AgentRegistry:
    return request.app.state.agents


def get_episodes(request: Request) -> EpisodeRegistry:
    return request.app.state.episodes


def get_tb_root(request: Request) -> Path:
    return request.app.state.tensorboard_root


def get_started_at(request: Request) -> float:
    return request.app.state.started_at


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _resolve_version() -> str:
    try:
        return version("retro-rl")
    except PackageNotFoundError:  # pragma: no cover — only hit outside an install
        return "0.0.0+unknown"


def create_app(
    *,
    checkpoint_root: Path | str = "outputs/checkpoints",
    tensorboard_root: Path | str = "outputs/tensorboard",
    agent_cache_size: int = 4,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Parameters
    ----------
    checkpoint_root
        Directory containing run subdirectories with checkpoint .zips and sidecars.
    tensorboard_root
        Directory containing one TB log subdir per training invocation.
    agent_cache_size
        LRU cap on the :class:`AgentRegistry`. 4 PPOs ≈ 80 MB resident.
    cors_origins
        Origins allowed for CORS. Defaults to Streamlit localhost ports.
    """
    ckpt_root = Path(checkpoint_root)
    tb_root = Path(tensorboard_root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolver = CheckpointResolver(ckpt_root)
        agents = AgentRegistry(resolver, max_cached=agent_cache_size)
        episodes = EpisodeRegistry()
        app.state.resolver = resolver
        app.state.agents = agents
        app.state.episodes = episodes
        app.state.tensorboard_root = tb_root
        app.state.started_at = time.monotonic()
        logger.info(
            "retro-rl backend up: checkpoints=%s tensorboard=%s",
            ckpt_root,
            tb_root,
        )
        try:
            yield
        finally:
            episodes.close_all()
            agents.clear()
            logger.info("retro-rl backend shut down")

    app = FastAPI(
        title="retro-rl backend",
        description=(
            "Read-only consumer of training artifacts plus an in-process "
            "episode runtime for client-driven rollouts. Frontend talks to "
            "this service over HTTP only — no shared Python imports."
        ),
        version=_resolve_version(),
        lifespan=lifespan,
    )

    origins = cors_origins or [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    _register_routes(app)
    return app


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(
        started_at: Annotated[float, Depends(get_started_at)],
    ) -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=_resolve_version(),
            uptime_seconds=max(0.0, time.monotonic() - started_at),
        )

    # ---- checkpoints / runs (read-only, derived from disk) ---------------

    @app.get("/checkpoints", response_model=CheckpointList, tags=["catalog"])
    def list_checkpoints(
        resolver: Annotated[CheckpointResolver, Depends(get_resolver)],
    ) -> CheckpointList:
        return CheckpointList(checkpoints=resolver.list_all())

    @app.get("/runs", response_model=RunList, tags=["catalog"])
    def list_runs(
        resolver: Annotated[CheckpointResolver, Depends(get_resolver)],
    ) -> RunList:
        return RunList(runs=_build_run_list(resolver))

    @app.get(
        "/runs/{run_name}/metrics",
        response_model=RunMetrics,
        tags=["catalog"],
    )
    def get_run_metrics(
        run_name: str,
        tb_root: Annotated[Path, Depends(get_tb_root)],
    ) -> RunMetrics:
        try:
            return _load_run_metrics(run_name, tb_root)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    # ---- episodes (stateful, client-driven) ------------------------------

    @app.post(
        "/episodes",
        response_model=EpisodeStartResponse,
        status_code=201,
        tags=["episodes"],
    )
    def start_episode(
        req: EpisodeStartRequest,
        resolver: Annotated[CheckpointResolver, Depends(get_resolver)],
        agents: Annotated[AgentRegistry, Depends(get_agents)],
        episodes: Annotated[EpisodeRegistry, Depends(get_episodes)],
    ) -> EpisodeStartResponse:
        try:
            agent = agents.get(req.checkpoint_id)
            env_cfg = resolver.env_config_for(req.checkpoint_id)
        except CheckpointNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        ep = EpisodeRuntime.start(
            checkpoint_id=req.checkpoint_id,
            agent=agent,
            env_cfg=env_cfg,
            seed=req.seed,
            deterministic=req.deterministic,
            max_steps=req.max_steps,
        )
        episodes.register(ep)
        logger.info(
            "started episode %s on %s (seed=%s deterministic=%s)",
            ep.episode_id,
            req.checkpoint_id,
            req.seed,
            req.deterministic,
        )
        return EpisodeStartResponse(
            episode_id=ep.episode_id,
            checkpoint_id=ep.checkpoint_id,
            started_at=ep.started_at,
        )

    @app.get(
        "/episodes/{episode_id}/state",
        response_model=EpisodeState,
        tags=["episodes"],
    )
    def get_episode_state(
        episode_id: str,
        episodes: Annotated[EpisodeRegistry, Depends(get_episodes)],
    ) -> EpisodeState:
        try:
            ep = episodes.get(episode_id)
        except EpisodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return ep.state()

    @app.get(
        "/episodes/{episode_id}/frame",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
        tags=["episodes"],
    )
    def get_episode_frame(
        episode_id: str,
        episodes: Annotated[EpisodeRegistry, Depends(get_episodes)],
    ) -> Response:
        """Advance one env step (if not done) and return the current frame as PNG.

        Once the episode ends, repeated calls return the final frame without
        stepping — clients should poll ``GET /state`` to detect ``done`` and
        stop polling.
        """
        try:
            ep = episodes.get(episode_id)
        except EpisodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        if not ep.done:
            # Race: another thread may finish the episode between the `done`
            # check and `step()`. Treat as done; serve the last frame.
            with contextlib.suppress(EpisodeFinishedError):
                ep.step()
        return Response(content=ep.frame_png(), media_type="image/png")

    @app.delete(
        "/episodes/{episode_id}",
        status_code=204,
        response_class=Response,
        tags=["episodes"],
    )
    def end_episode(
        episode_id: str,
        episodes: Annotated[EpisodeRegistry, Depends(get_episodes)],
    ) -> Response:
        try:
            episodes.get(episode_id)
        except EpisodeNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        episodes.remove(episode_id)
        return Response(status_code=204)


# ---------------------------------------------------------------------------
# Helpers — run summary + TB scalar parsing
# ---------------------------------------------------------------------------


def _build_run_list(resolver: CheckpointResolver) -> list[RunInfo]:
    """Aggregate per-run summary from the resolver's checkpoint enumeration."""
    by_run: dict[str, list[CheckpointInfo]] = {}
    for ci in resolver.list_all():
        by_run.setdefault(ci.run_name, []).append(ci)

    out: list[RunInfo] = []
    for run_name, ckpts in by_run.items():
        best = next((c for c in ckpts if c.kind == "best"), None)
        step_ckpts = [c for c in ckpts if c.kind == "step"]
        latest_step = max((c.step for c in step_ckpts), default=None)
        config_path = resolver.root / run_name / "config_snapshot.json"
        out.append(
            RunInfo(
                run_name=run_name,
                has_best=best is not None,
                best_return=best.eval_return if best is not None else None,
                latest_step=latest_step,
                checkpoint_count=len(ckpts),
                config_snapshot_path=str(config_path) if config_path.exists() else None,
            )
        )
    out.sort(key=lambda r: r.run_name)
    return out


def _load_run_metrics(run_name: str, tb_root: Path) -> RunMetrics:
    """Parse all scalar series from the latest TB log dir for *run_name*.

    SB3 appends ``_<N>`` to the configured ``tb_log_name`` for each
    ``learn()`` invocation. We pick the most-recently-modified match — the
    "current" or "latest" run is what the dashboard cares about. Merging
    across re-trains is a future enhancement if it shows up as a need.
    """
    if not tb_root.is_dir():
        raise FileNotFoundError(f"tensorboard root does not exist: {tb_root}")

    candidates = [p for p in tb_root.glob(f"{run_name}_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no TB log dir matching {run_name!r} in {tb_root}")
    tb_dir = max(candidates, key=lambda p: p.stat().st_mtime)

    # Lazy import — tensorboard is heavy and only this route needs it.
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={event_accumulator.SCALARS: 0},  # 0 = load all
    )
    ea.Reload()

    series: list[MetricSeries] = []
    for tag in sorted(ea.Tags().get("scalars", [])):
        events = ea.Scalars(tag)
        points = [MetricPoint(step=int(e.step), value=float(e.value)) for e in events]
        series.append(MetricSeries(name=tag, points=points))

    return RunMetrics(run_name=run_name, series=series)


__all__ = [
    "create_app",
    "get_resolver",
    "get_agents",
    "get_episodes",
    "get_tb_root",
    "get_started_at",
]
