"""Pydantic request/response schemas for the FastAPI backend.

The backend's public contract lives here. Routes in :mod:`retro_rl.backend.api`
serialize/deserialize through these models — no untyped dicts cross the seam.

Conventions
-----------
* ``extra="forbid"`` on every model — typos in client payloads should 400, not
  silently drop.
* Timestamps are ISO-8601 strings (e.g. ``"2026-05-18T13:52:58+00:00"``). They
  originate as strings in checkpoint sidecars; re-parsing to ``datetime`` buys
  nothing and forces the frontend to re-stringify.
* Checkpoint identity = ``"<run_name>/<kind>"`` where ``kind`` is ``"best"`` or
  ``"step-<N>"``. This is the stable client-facing handle, decoupled from disk
  paths (which can move).
* Binary endpoints (``GET /episodes/{id}/frame``) have no schema — they emit
  ``image/png`` bytes directly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """``GET /health`` — liveness + minimal version handshake."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    version: str  # retro_rl package version
    uptime_seconds: float = Field(..., ge=0.0)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


class CheckpointInfo(BaseModel):
    """One checkpoint, materialised from its JSON sidecar."""

    model_config = ConfigDict(extra="forbid")

    id: str  # "<run_name>/<kind>" — client-facing handle
    run_name: str
    kind: Literal["step", "best"]
    step: int = Field(..., ge=0)
    eval_return: float | None  # None when sidecar has no eval signal
    timestamp: str  # ISO-8601 from sidecar
    path: str  # repo-relative .zip path (frontend may display)


class CheckpointList(BaseModel):
    """``GET /checkpoints`` — flat list across all runs."""

    model_config = ConfigDict(extra="forbid")

    checkpoints: list[CheckpointInfo]


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


class EpisodeStartRequest(BaseModel):
    """``POST /episodes`` body — start a new rollout."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str  # e.g. "ppo_airstriker/best"
    seed: int | None = None
    deterministic: bool = True
    max_steps: int | None = Field(default=None, ge=1)  # override env max_episode_steps


class EpisodeStartResponse(BaseModel):
    """``POST /episodes`` response."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str  # uuid4 hex
    checkpoint_id: str
    started_at: str  # ISO-8601


class EpisodeState(BaseModel):
    """``GET /episodes/{id}/state`` — current rollout state.

    ``done = terminated or truncated``. ``info`` carries the last env-step
    info dict (RAM-derived variables, shaping breakdown, etc.) — coerced to
    JSON-safe types upstream of this model.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    step: int = Field(..., ge=0)
    total_reward: float
    terminated: bool
    truncated: bool
    done: bool
    last_action: int | None  # None before any step has happened
    info: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class RunInfo(BaseModel):
    """One training run's summary, derived from its checkpoint directory."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    has_best: bool
    best_return: float | None
    best_length: float | None  # max eval/mean_ep_length seen in TB; None if not available
    latest_step: int | None
    checkpoint_count: int = Field(..., ge=0)
    config_snapshot_path: str | None  # repo-relative path to config_snapshot.json


class RunList(BaseModel):
    """``GET /runs`` — all training runs visible to the backend."""

    model_config = ConfigDict(extra="forbid")

    runs: list[RunInfo]


class MetricPoint(BaseModel):
    """One (step, value) sample from a TB scalar series."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(..., ge=0)
    value: float


class MetricSeries(BaseModel):
    """A single named scalar series (e.g. ``eval/mean_return``)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    points: list[MetricPoint]


class RunMetrics(BaseModel):
    """``GET /runs/{run_name}/metrics`` — all scalar series for a run.

    Keyed by series name (the TB tag, e.g. ``eval/mean_return``,
    ``train/loss``). Frontend decides which to plot.
    """

    model_config = ConfigDict(extra="forbid")

    run_name: str
    series: list[MetricSeries]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """FastAPI default ``{"detail": ...}`` shape, modelled for OpenAPI docs."""

    model_config = ConfigDict(extra="forbid")

    detail: str


__all__ = [
    "HealthResponse",
    "CheckpointInfo",
    "CheckpointList",
    "EpisodeStartRequest",
    "EpisodeStartResponse",
    "EpisodeState",
    "RunInfo",
    "RunList",
    "MetricPoint",
    "MetricSeries",
    "RunMetrics",
    "ErrorResponse",
]
