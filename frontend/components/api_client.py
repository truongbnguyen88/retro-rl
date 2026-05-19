"""HTTP client for the retro-rl backend.

Single source of HTTP truth for the Streamlit frontend. Per CLAUDE.md, the
frontend never imports backend Python — every interaction goes through these
functions, which speak the backend's REST API.

Caching strategy
----------------
Catalog endpoints (``/health``, ``/runs``, ``/checkpoints``, metrics) are
wrapped with ``st.cache_data`` so repeated reruns don't hammer the backend.
TTLs are short (5–15 s) so a live training run still surfaces fresh metrics
on the next page refresh.

Episode endpoints are **never** cached — they are stateful and every call has
side effects (``/frame`` advances the rollout).

Errors
------
HTTP errors propagate as :class:`BackendError` with the response body when
available; callers in pages can ``st.error`` them directly.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


DEFAULT_BACKEND_URL = "http://localhost:8000"


def backend_url() -> str:
    """Resolve backend URL from env or default."""
    return os.environ.get("RETRO_RL_BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")


class BackendError(RuntimeError):
    """Raised when a backend call fails. Carries status + body for display."""

    def __init__(self, status: int, body: str, url: str) -> None:
        super().__init__(f"backend {status} at {url}: {body}")
        self.status = status
        self.body = body
        self.url = url


def _get(path: str, *, timeout: float = 5.0, params: dict[str, Any] | None = None) -> Any:
    url = f"{backend_url()}{path}"
    try:
        r = requests.get(url, timeout=timeout, params=params)
    except requests.RequestException as e:
        raise BackendError(0, str(e), url) from e
    if not r.ok:
        raise BackendError(r.status_code, r.text, url)
    return r.json()


def _post(path: str, *, json_body: dict[str, Any], timeout: float = 10.0) -> Any:
    url = f"{backend_url()}{path}"
    try:
        r = requests.post(url, json=json_body, timeout=timeout)
    except requests.RequestException as e:
        raise BackendError(0, str(e), url) from e
    if not r.ok:
        raise BackendError(r.status_code, r.text, url)
    return r.json()


def _delete(path: str, *, timeout: float = 5.0) -> int:
    url = f"{backend_url()}{path}"
    try:
        r = requests.delete(url, timeout=timeout)
    except requests.RequestException as e:
        raise BackendError(0, str(e), url) from e
    if r.status_code not in (200, 204, 404):
        raise BackendError(r.status_code, r.text, url)
    return r.status_code


# ---------------------------------------------------------------------------
# Catalog endpoints — cached
# ---------------------------------------------------------------------------


@st.cache_data(ttl=5, show_spinner=False)
def get_health() -> dict[str, Any]:
    """Cached health probe — used by the sidebar status indicator."""
    return _get("/health")


@st.cache_data(ttl=10, show_spinner=False)
def list_runs() -> list[dict[str, Any]]:
    return _get("/runs")["runs"]


@st.cache_data(ttl=10, show_spinner=False)
def list_checkpoints() -> list[dict[str, Any]]:
    return _get("/checkpoints")["checkpoints"]


@st.cache_data(ttl=15, show_spinner=False)
def get_run_metrics(run_name: str) -> dict[str, list[dict[str, float]]]:
    """Return ``{series_name: [{step, value}, ...]}`` for a run.

    Cache TTL 15 s so live training runs surface new points within one
    page-refresh interval without overloading the backend.
    """
    body = _get(f"/runs/{run_name}/metrics")
    return {s["name"]: s["points"] for s in body["series"]}


# ---------------------------------------------------------------------------
# Episode endpoints — never cached (stateful)
# ---------------------------------------------------------------------------


def start_episode(
    *,
    checkpoint_id: str,
    seed: int | None = None,
    deterministic: bool = True,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """POST /episodes — returns episode_id + started_at."""
    body: dict[str, Any] = {"checkpoint_id": checkpoint_id, "deterministic": deterministic}
    if seed is not None:
        body["seed"] = seed
    if max_steps is not None:
        body["max_steps"] = max_steps
    return _post("/episodes", json_body=body)


def get_episode_state(episode_id: str) -> dict[str, Any]:
    return _get(f"/episodes/{episode_id}/state")


def get_episode_frame(episode_id: str) -> bytes:
    """Advance one step and return PNG bytes. Raw response — no JSON decode."""
    url = f"{backend_url()}/episodes/{episode_id}/frame"
    try:
        r = requests.get(url, timeout=10.0)
    except requests.RequestException as e:
        raise BackendError(0, str(e), url) from e
    if not r.ok:
        raise BackendError(r.status_code, r.text, url)
    return r.content


def end_episode(episode_id: str) -> None:
    """DELETE /episodes/{id} — idempotent from the client's perspective."""
    _delete(f"/episodes/{episode_id}")


def clear_catalog_cache() -> None:
    """Invalidate cached catalog endpoints. Useful after a manual refresh."""
    get_health.clear()
    list_runs.clear()
    list_checkpoints.clear()
    get_run_metrics.clear()


__all__ = [
    "BackendError",
    "backend_url",
    "get_health",
    "list_runs",
    "list_checkpoints",
    "get_run_metrics",
    "start_episode",
    "get_episode_state",
    "get_episode_frame",
    "end_episode",
    "clear_catalog_cache",
]
