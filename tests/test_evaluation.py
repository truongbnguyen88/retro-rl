"""Milestone-4 evaluation tests.

Covers:
  * compute_metrics — pure unit tests, no env
  * evaluate — mock env + mock agent, no ROM required
  * write_mp4 — atomic write, empty-frames guard
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from retro_rl.evaluation.evaluator import evaluate
from retro_rl.evaluation.metrics import EpisodeResult, compute_metrics
from retro_rl.utils.video import write_mp4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FixedAgent:
    """Always predicts action 0. Conforms to the Agent protocol (accepts and
    returns recurrent state); records episode_start flags so tests can assert
    the evaluator threads/resets state correctly."""

    def __init__(self):
        self.episode_starts: list[bool] = []

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        if episode_start is not None:
            self.episode_starts.append(bool(np.asarray(episode_start).ravel()[0]))
        return 0, state


class _MockEnv:
    """Configurable single-episode mock — replays a fixed step sequence.

    Parameters
    ----------
    steps_per_episode
        How many calls to ``step()`` before the episode ends.
    reward_per_step
        Reward returned at every step.
    terminate_at_end
        If True the last step sets ``terminated=True``; else ``truncated=True``.
    stage_clear_at_step
        If not None, ``info["stage_clear"]`` is set to 1 at that step index
        (0-based, relative to episode start).
    render_frame
        If not None, ``render()`` returns this array; else returns None.
    """

    def __init__(
        self,
        steps_per_episode: int = 5,
        reward_per_step: float = 1.0,
        terminate_at_end: bool = True,
        stage_clear_at_step: int | None = None,
        render_frame: np.ndarray | None = None,
    ) -> None:
        self._steps_per_ep = steps_per_episode
        self._reward = reward_per_step
        self._terminate = terminate_at_end
        self._stage_clear_step = stage_clear_at_step
        self._render_frame = render_frame
        self._obs = np.zeros((84, 84, 4), dtype=np.uint8)
        self._step_count = 0

    def reset(self, seed=None, options=None):
        self._step_count = 0
        return self._obs, {}

    def step(self, action):
        step_idx = self._step_count
        self._step_count += 1
        done = self._step_count >= self._steps_per_ep
        terminated = done and self._terminate
        truncated = done and not self._terminate
        info: dict = {}
        if self._stage_clear_step is not None and step_idx == self._stage_clear_step:
            info["stage_clear"] = 1
        return self._obs, self._reward, terminated, truncated, info

    def render(self):
        return self._render_frame

    def close(self):
        pass


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_basic():
    episodes = [
        EpisodeResult(return_=10.0, length=50, stage_cleared=True, deaths=1),
        EpisodeResult(return_=20.0, length=100, stage_cleared=False, deaths=0),
    ]
    m = compute_metrics(episodes)
    assert m.n_episodes == 2
    assert m.mean_return == pytest.approx(15.0)
    assert m.std_return == pytest.approx(5.0)
    assert m.min_return == pytest.approx(10.0)
    assert m.max_return == pytest.approx(20.0)
    assert m.mean_length == pytest.approx(75.0)
    assert m.stage_clear_rate == pytest.approx(0.5)
    assert m.mean_deaths == pytest.approx(0.5)


def test_compute_metrics_single_episode():
    ep = EpisodeResult(return_=42.0, length=30, stage_cleared=False, deaths=2)
    m = compute_metrics([ep])
    assert m.n_episodes == 1
    assert m.mean_return == pytest.approx(42.0)
    assert m.std_return == pytest.approx(0.0)
    assert m.min_return == pytest.approx(42.0)
    assert m.max_return == pytest.approx(42.0)
    assert m.stage_clear_rate == pytest.approx(0.0)
    assert m.mean_deaths == pytest.approx(2.0)


def test_compute_metrics_all_cleared():
    episodes = [
        EpisodeResult(return_=5.0, length=10, stage_cleared=True, deaths=0),
        EpisodeResult(return_=7.0, length=10, stage_cleared=True, deaths=0),
    ]
    m = compute_metrics(episodes)
    assert m.stage_clear_rate == pytest.approx(1.0)


def test_compute_metrics_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        compute_metrics([])


def test_eval_metrics_is_frozen():
    m = compute_metrics([EpisodeResult(return_=1.0, length=5, stage_cleared=False, deaths=0)])
    # Frozen dataclass / pydantic model: assignment raises AttributeError
    # (FrozenInstanceError subclasses AttributeError in 3.11+).
    with pytest.raises(AttributeError):
        m.mean_return = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# evaluate — episode count + return aggregation
# ---------------------------------------------------------------------------


def test_evaluate_runs_correct_episode_count():
    env = _MockEnv(steps_per_episode=3)
    metrics, frames = evaluate(_FixedAgent(), env, n_episodes=5)
    assert metrics.n_episodes == 5
    assert frames == []


def test_evaluate_return_accumulates_per_episode():
    steps = 4
    reward = 2.0
    env = _MockEnv(steps_per_episode=steps, reward_per_step=reward)
    metrics, _ = evaluate(_FixedAgent(), env, n_episodes=3)
    assert metrics.mean_return == pytest.approx(steps * reward)
    assert metrics.std_return == pytest.approx(0.0)


def test_evaluate_length_correct():
    env = _MockEnv(steps_per_episode=7)
    metrics, _ = evaluate(_FixedAgent(), env, n_episodes=2)
    assert metrics.mean_length == pytest.approx(7.0)


def test_evaluate_rejects_zero_episodes():
    env = _MockEnv()
    with pytest.raises(ValueError, match="n_episodes"):
        evaluate(_FixedAgent(), env, n_episodes=0)


# ---------------------------------------------------------------------------
# evaluate — death counting
# ---------------------------------------------------------------------------


def test_evaluate_counts_termination_as_death():
    env = _MockEnv(steps_per_episode=3, terminate_at_end=True)
    metrics, _ = evaluate(_FixedAgent(), env, n_episodes=4)
    assert metrics.mean_deaths == pytest.approx(1.0)


def test_evaluate_truncation_not_counted_as_death():
    env = _MockEnv(steps_per_episode=3, terminate_at_end=False)
    metrics, _ = evaluate(_FixedAgent(), env, n_episodes=4)
    assert metrics.mean_deaths == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate — stage-clear detection
# ---------------------------------------------------------------------------


def test_evaluate_detects_stage_clear():
    env = _MockEnv(steps_per_episode=5, stage_clear_at_step=2)
    metrics, _ = evaluate(
        _FixedAgent(), env, n_episodes=3, info_keys={"stage_clear": "stage_clear"}
    )
    assert metrics.stage_clear_rate == pytest.approx(1.0)


def test_evaluate_no_stage_clear_when_key_absent():
    env = _MockEnv(steps_per_episode=5, stage_clear_at_step=None)
    metrics, _ = evaluate(_FixedAgent(), env, n_episodes=3)
    assert metrics.stage_clear_rate == pytest.approx(0.0)


def test_evaluate_custom_info_key_for_stage_clear():
    """Simulates Airstriker where stage_clear maps to 'gameover'."""

    class _GameoverEnv(_MockEnv):
        def step(self, action):
            obs, r, term, trunc, _ = super().step(action)
            info = {"gameover": 1} if self._step_count == 3 else {}
            return obs, r, term, trunc, info

    env = _GameoverEnv(steps_per_episode=5)
    metrics, _ = evaluate(_FixedAgent(), env, n_episodes=2, info_keys={"stage_clear": "gameover"})
    assert metrics.stage_clear_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate — video recording
# ---------------------------------------------------------------------------


def test_evaluate_collects_frames_first_episode_only():
    frame = np.ones((224, 320, 3), dtype=np.uint8) * 128
    env = _MockEnv(steps_per_episode=4, render_frame=frame)
    _, frames = evaluate(_FixedAgent(), env, n_episodes=3, record_video=True)
    # 4 steps × 1 episode = 4 frames
    assert len(frames) == 4
    assert frames[0].shape == (224, 320, 3)


def test_evaluate_no_frames_when_record_video_false():
    frame = np.ones((224, 320, 3), dtype=np.uint8)
    env = _MockEnv(steps_per_episode=4, render_frame=frame)
    _, frames = evaluate(_FixedAgent(), env, n_episodes=2, record_video=False)
    assert frames == []


def test_evaluate_no_frames_when_render_returns_none():
    env = _MockEnv(steps_per_episode=4, render_frame=None)
    _, frames = evaluate(_FixedAgent(), env, n_episodes=2, record_video=True)
    assert frames == []


# ---------------------------------------------------------------------------
# write_mp4
# ---------------------------------------------------------------------------


def test_write_mp4_creates_file(tmp_path: Path):
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(10)]
    out = tmp_path / "test.mp4"
    write_mp4(frames, out, fps=10)
    assert out.exists()
    assert out.stat().st_size > 0


def test_write_mp4_no_tmp_file_remains(tmp_path: Path):
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]
    out = tmp_path / "out.mp4"
    write_mp4(frames, out, fps=5)
    leftovers = list(tmp_path.glob("*.tmp*"))
    assert leftovers == []


def test_write_mp4_empty_frames_raises():
    with pytest.raises(ValueError, match="non-empty"):
        write_mp4([], Path("/tmp/noop.mp4"))


def test_write_mp4_accepts_path_string(tmp_path: Path):
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
    out = str(tmp_path / "str_path.mp4")
    write_mp4(frames, out, fps=5)
    assert Path(out).exists()


def test_evaluate_threads_recurrent_state_resets_per_episode():
    """episode_start must be True only on the first step of each episode, so a
    recurrent policy zeroes its LSTM state at the boundary and accumulates it
    within the episode. Regression guard for the prior bug where the evaluator
    discarded the returned state and never passed episode_start, evaluating
    recurrent policies as if memoryless."""
    agent = _FixedAgent()
    env = _MockEnv(steps_per_episode=3)
    evaluate(agent, env, n_episodes=2)
    # 2 episodes × 3 steps; True at the start of each episode, False otherwise.
    assert agent.episode_starts == [True, False, False, True, False, False]
