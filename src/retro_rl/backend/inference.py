"""Lazy-loaded agent registry + in-process episode runtime for the FastAPI backend.

Two cooperating layers, both thread-safe (FastAPI runs sync routes in a
threadpool, so concurrent requests on the same episode are possible):

:class:`CheckpointResolver`
    Walks ``outputs/checkpoints/<run>/`` and maps ``checkpoint_id`` strings
    (``"<run_name>/best"`` or ``"<run_name>/step-<N>"``) to disk paths and to
    the :class:`~retro_rl.utils.config.EnvConfig` snapshotted with the run.

:class:`AgentRegistry`
    LRU-bounded cache of ``PPO`` instances keyed by ``checkpoint_id``. Loading
    a 20-MB SB3 zip takes ~1 s; we never want to do that on the hot path.

:class:`EpisodeRuntime`
    One in-flight rollout: env + agent + (step, reward, done, last frame).
    The unit of advance is ``step()`` (one env step). Per-instance lock makes
    concurrent ``/frame`` and ``/state`` calls safe.

:class:`EpisodeRegistry`
    Thread-safe ``episode_id -> EpisodeRuntime`` map. No automatic GC; the
    backend may call :meth:`EpisodeRegistry.remove` when an episode ends or
    when the client disconnects. (Abandonment-cleanup is a future polish.)

Path conventions
----------------
Disk layout owned by :class:`~retro_rl.training.checkpoint.CheckpointManager`:
    outputs/checkpoints/<run_name>/
        ├── best.zip          (if any eval set a best)
        ├── best.json         (sidecar; same schema as step-*.json)
        ├── step-<N>.zip      (last-K rotated)
        ├── step-<N>.json
        └── config_snapshot.json   (TrainConfig dumped at run start)

The sidecar JSON is the authoritative source for checkpoint metadata; we never
re-derive timestamps or eval returns from the .zip itself.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from stable_baselines3 import PPO

from retro_rl.backend.models import CheckpointInfo, EpisodeState
from retro_rl.env import make_env
from retro_rl.utils.config import EnvConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InferenceError(Exception):
    """Base class for backend-inference errors."""


class CheckpointNotFoundError(InferenceError):
    """Raised when a ``checkpoint_id`` does not resolve to a .zip on disk."""


class EpisodeNotFoundError(InferenceError):
    """Raised when an ``episode_id`` is not in the registry."""


class EpisodeFinishedError(InferenceError):
    """Raised when ``step()`` is called on an episode that already ended."""


# ---------------------------------------------------------------------------
# CheckpointResolver — disk → checkpoint metadata
# ---------------------------------------------------------------------------


def _parse_checkpoint_id(checkpoint_id: str) -> tuple[str, str]:
    """Split ``"<run_name>/<kind>"``. ``kind`` is ``"best"`` or ``"step-<N>"``."""
    if "/" not in checkpoint_id:
        raise CheckpointNotFoundError(
            f"checkpoint_id must be '<run_name>/<kind>', got {checkpoint_id!r}"
        )
    run_name, kind = checkpoint_id.split("/", 1)
    if not (kind == "best" or kind.startswith("step-")):
        raise CheckpointNotFoundError(
            f"checkpoint kind must be 'best' or 'step-<N>', got {kind!r}"
        )
    return run_name, kind


class CheckpointResolver:
    """Disk walker for checkpoint discovery + EnvConfig resolution.

    Parameters
    ----------
    checkpoint_root
        Directory containing one subdirectory per run. Defaults to
        ``outputs/checkpoints`` (matches :class:`TrainConfig.checkpoint_dir`).
    """

    def __init__(self, checkpoint_root: Path = Path("outputs/checkpoints")) -> None:
        self.root = Path(checkpoint_root)

    # ---- resolution ------------------------------------------------------

    def resolve_path(self, checkpoint_id: str) -> Path:
        """Return the .zip path for *checkpoint_id*, raising if missing."""
        run_name, kind = _parse_checkpoint_id(checkpoint_id)
        path = self.root / run_name / f"{kind}.zip"
        if not path.is_file():
            raise CheckpointNotFoundError(
                f"checkpoint not found on disk: {path}"
            )
        return path

    def env_config_for(self, checkpoint_id: str) -> EnvConfig:
        """Load the env config snapshotted alongside the run."""
        run_name, _ = _parse_checkpoint_id(checkpoint_id)
        snap_path = self.root / run_name / "config_snapshot.json"
        if not snap_path.is_file():
            raise CheckpointNotFoundError(
                f"config_snapshot.json missing for run {run_name!r}: {snap_path}"
            )
        data = json.loads(snap_path.read_text())
        env_data = data.get("env")
        if env_data is None:
            raise CheckpointNotFoundError(
                f"config_snapshot.json for {run_name!r} has no 'env' key"
            )
        return EnvConfig.model_validate(env_data)

    # ---- enumeration -----------------------------------------------------

    def list_all(self) -> list[CheckpointInfo]:
        """Return all checkpoints across all runs, derived from JSON sidecars."""
        if not self.root.is_dir():
            return []
        out: list[CheckpointInfo] = []
        for run_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for sidecar in sorted(run_dir.glob("*.json")):
                # config_snapshot.json is not a checkpoint sidecar
                if sidecar.name == "config_snapshot.json":
                    continue
                zip_path = sidecar.with_suffix(".zip")
                if not zip_path.exists():
                    continue
                info = self._sidecar_to_info(sidecar, zip_path)
                if info is not None:
                    out.append(info)
        return out

    def _sidecar_to_info(
        self, sidecar: Path, zip_path: Path
    ) -> CheckpointInfo | None:
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("skipping unreadable sidecar %s: %s", sidecar, e)
            return None
        try:
            run_name = meta["run_name"]
            kind = meta["kind"]
            step = int(meta["step"])
        except KeyError as e:
            logger.warning("sidecar %s missing field %s; skipping", sidecar, e)
            return None
        return CheckpointInfo(
            id=f"{run_name}/{sidecar.stem}",
            run_name=run_name,
            kind=kind,
            step=step,
            eval_return=meta.get("eval_return"),
            timestamp=meta.get("timestamp", ""),
            path=str(zip_path),
        )


# ---------------------------------------------------------------------------
# AgentRegistry — lazy-loaded, LRU-bounded PPO cache
# ---------------------------------------------------------------------------


class AgentRegistry:
    """LRU cache of loaded ``PPO`` instances keyed by ``checkpoint_id``.

    ``get`` is the only hot path. A miss triggers ``PPO.load`` (slow, ~1 s for
    a 20-MB checkpoint); a hit returns the cached instance in O(1).

    Thread-safety: every public method acquires a single registry-level lock.
    The lock is held across ``PPO.load`` on a miss — concurrent first-requests
    for the same checkpoint will serialize, but each one runs once.

    Parameters
    ----------
    resolver
        Used to translate ``checkpoint_id`` → disk path.
    max_cached
        Maximum simultaneously-cached PPOs. When the cap is exceeded the LRU
        entry is evicted. 4 × ~50 MB resident is a comfortable default.
    """

    def __init__(self, resolver: CheckpointResolver, max_cached: int = 4) -> None:
        if max_cached < 1:
            raise ValueError(f"max_cached must be >= 1, got {max_cached}")
        self._resolver = resolver
        self._max = max_cached
        self._cache: OrderedDict[str, PPO] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, checkpoint_id: str) -> PPO:
        """Return a loaded PPO for *checkpoint_id*; loads on first request."""
        with self._lock:
            if checkpoint_id in self._cache:
                self._cache.move_to_end(checkpoint_id)
                return self._cache[checkpoint_id]
            path = self._resolver.resolve_path(checkpoint_id)
            logger.info("loading PPO checkpoint: %s", path)
            # env=None — inference-only; SB3 allows this for predict.
            model = PPO.load(str(path), env=None)
            self._cache[checkpoint_id] = model
            while len(self._cache) > self._max:
                evicted_id, _ = self._cache.popitem(last=False)
                logger.info("evicted PPO from cache: %s", evicted_id)
            return model

    def evict(self, checkpoint_id: str) -> None:
        """Drop a single entry. No-op when absent."""
        with self._lock:
            self._cache.pop(checkpoint_id, None)

    def clear(self) -> None:
        """Drop all cached agents."""
        with self._lock:
            self._cache.clear()

    def cached_ids(self) -> list[str]:
        """Return currently-cached ids (MRU last). Diagnostic, not API surface."""
        with self._lock:
            return list(self._cache.keys())


# ---------------------------------------------------------------------------
# Helpers — JSON-safe info dict + frame → PNG
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Coerce numpy scalars / arrays inside an info dict to JSON-serializable types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        v = float(value)
        # Pydantic rejects nan/inf by default; coerce to None for safety.
        if not np.isfinite(v):
            return None
        return v
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _frame_to_png(frame_rgb: np.ndarray) -> bytes:
    """Encode an RGB uint8 (H, W, 3) frame as PNG bytes."""
    if frame_rgb.dtype != np.uint8:
        frame_rgb = frame_rgb.astype(np.uint8)
    img = Image.fromarray(frame_rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _action_to_int(action: Any) -> int | None:
    """Coerce SB3's action output (np scalar / 0-d array / 1-d array) to a Python int."""
    if action is None:
        return None
    if isinstance(action, np.ndarray):
        if action.size != 1:
            # Multi-discrete or vector action — fall back to first elem for the
            # state model. The action was already applied; this is just for
            # display. Caller can override if they need full fidelity later.
            return int(action.flat[0])
        return int(action.item())
    if isinstance(action, (np.integer, int)):
        return int(action)
    return None


# ---------------------------------------------------------------------------
# EpisodeRuntime — single rollout, client-driven
# ---------------------------------------------------------------------------


class EpisodeRuntime:
    """One in-flight rollout. Advance with :meth:`step`; observe with
    :meth:`state` / :meth:`frame_png`.

    The env is constructed in :meth:`start` with ``render_mode='rgb_array'``
    so frames are always available. ``deterministic`` is forwarded to
    ``agent.predict`` at every step.

    Thread-safety
    -------------
    A per-instance ``threading.Lock`` serializes ``step`` / ``state`` /
    ``frame_png`` / ``close``. Concurrent calls from FastAPI's threadpool are
    safe but will queue.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        checkpoint_id: str,
        agent: PPO,
        env,
        deterministic: bool,
        max_steps: int | None,
    ) -> None:
        self.episode_id = episode_id
        self.checkpoint_id = checkpoint_id
        self._agent = agent
        self._env = env
        self._deterministic = deterministic
        self._max_steps = max_steps

        self._lock = threading.Lock()
        self._obs: np.ndarray | None = None  # populated by start()
        self._step_count = 0
        self._total_reward = 0.0
        self._terminated = False
        self._truncated = False
        self._last_action: int | None = None
        self._last_info: dict[str, Any] = {}
        self._last_frame: np.ndarray | None = None
        self._started_at: str = ""
        self._closed = False

    # ---- lifecycle -------------------------------------------------------

    @classmethod
    def start(
        cls,
        *,
        episode_id: str | None = None,
        checkpoint_id: str,
        agent: PPO,
        env_cfg: EnvConfig,
        seed: int | None = None,
        deterministic: bool = True,
        max_steps: int | None = None,
    ) -> "EpisodeRuntime":
        """Build env, reset, capture initial frame. Returns a ready runtime."""
        env = make_env(env_cfg, seed=seed, render_mode="rgb_array")
        runtime = cls(
            episode_id=episode_id or uuid.uuid4().hex,
            checkpoint_id=checkpoint_id,
            agent=agent,
            env=env,
            deterministic=deterministic,
            max_steps=max_steps,
        )
        obs, info = env.reset(seed=seed)
        runtime._obs = obs
        runtime._last_info = _json_safe(info) if info else {}
        runtime._last_frame = runtime._render()
        runtime._started_at = datetime.now(tz=timezone.utc).isoformat()
        return runtime

    def close(self) -> None:
        """Release the underlying env. Idempotent."""
        with self._lock:
            if self._closed:
                return
            try:
                self._env.close()
            except Exception as e:  # pragma: no cover — env teardown is fire-and-forget
                logger.warning("error closing env for episode %s: %s",
                               self.episode_id, e)
            finally:
                self._closed = True

    # ---- step / observe --------------------------------------------------

    def step(self) -> None:
        """Advance one env step. No-op if already done. Raises if closed.

        Raises
        ------
        EpisodeFinishedError
            If the episode already ended (`done == True`).
        InferenceError
            If the episode has been closed.
        """
        with self._lock:
            if self._closed:
                raise InferenceError(f"episode {self.episode_id} is closed")
            if self._terminated or self._truncated:
                raise EpisodeFinishedError(
                    f"episode {self.episode_id} already finished at step {self._step_count}"
                )
            assert self._obs is not None
            action, _ = self._agent.predict(self._obs, deterministic=self._deterministic)
            obs, reward, terminated, truncated, info = self._env.step(action)

            self._obs = obs
            self._step_count += 1
            self._total_reward += float(reward)
            self._terminated = bool(terminated)
            self._truncated = bool(truncated)
            self._last_action = _action_to_int(action)
            self._last_info = _json_safe(info) if info else {}
            self._last_frame = self._render()

            if (
                self._max_steps is not None
                and self._step_count >= self._max_steps
                and not (self._terminated or self._truncated)
            ):
                self._truncated = True

    def state(self) -> EpisodeState:
        """Snapshot the current state — does not step."""
        with self._lock:
            return EpisodeState(
                episode_id=self.episode_id,
                step=self._step_count,
                total_reward=self._total_reward,
                terminated=self._terminated,
                truncated=self._truncated,
                done=self._terminated or self._truncated,
                last_action=self._last_action,
                info=self._last_info,
            )

    def frame_png(self) -> bytes:
        """Return the most-recently-rendered frame as PNG bytes."""
        with self._lock:
            if self._last_frame is None:
                raise InferenceError(
                    f"episode {self.episode_id} has no frame "
                    "(env render_mode may not be 'rgb_array')"
                )
            return _frame_to_png(self._last_frame)

    # ---- properties ------------------------------------------------------

    @property
    def done(self) -> bool:
        with self._lock:
            return self._terminated or self._truncated

    @property
    def started_at(self) -> str:
        return self._started_at

    # ---- internals -------------------------------------------------------

    def _render(self) -> np.ndarray | None:
        frame = self._env.render()
        if frame is None:
            return None
        return np.asarray(frame)


# ---------------------------------------------------------------------------
# EpisodeRegistry — episode_id -> EpisodeRuntime
# ---------------------------------------------------------------------------


class EpisodeRegistry:
    """Thread-safe ``episode_id -> EpisodeRuntime`` map.

    No automatic GC; callers (api.py) are expected to ``remove`` episodes when
    the client signals done or to call ``close_all`` on app shutdown.
    """

    def __init__(self) -> None:
        self._episodes: dict[str, EpisodeRuntime] = {}
        self._lock = threading.Lock()

    def register(self, ep: EpisodeRuntime) -> None:
        """Add an episode. Raises ValueError on duplicate id."""
        with self._lock:
            if ep.episode_id in self._episodes:
                raise ValueError(f"duplicate episode_id: {ep.episode_id}")
            self._episodes[ep.episode_id] = ep

    def get(self, episode_id: str) -> EpisodeRuntime:
        """Return the episode for *episode_id*; raises if absent."""
        with self._lock:
            ep = self._episodes.get(episode_id)
        if ep is None:
            raise EpisodeNotFoundError(f"unknown episode_id: {episode_id}")
        return ep

    def remove(self, episode_id: str) -> None:
        """Remove and close an episode. No-op if absent."""
        with self._lock:
            ep = self._episodes.pop(episode_id, None)
        if ep is not None:
            ep.close()

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._episodes.keys())

    def close_all(self) -> None:
        """Close and forget every episode. For app shutdown."""
        with self._lock:
            episodes = list(self._episodes.values())
            self._episodes.clear()
        for ep in episodes:
            ep.close()


__all__ = [
    "InferenceError",
    "CheckpointNotFoundError",
    "EpisodeNotFoundError",
    "EpisodeFinishedError",
    "CheckpointResolver",
    "AgentRegistry",
    "EpisodeRuntime",
    "EpisodeRegistry",
]
