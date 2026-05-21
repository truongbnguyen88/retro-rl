"""Play page — pick a checkpoint, start an episode, stream frames.

Streamlit's rerun model is the friction point. Two designs were considered:

1. **One-frame-per-rerun**: a button labelled "Step" advances exactly one
   frame. Simple and Streamlit-native but unusably slow for watching play.
2. **In-loop streaming** (chosen): on a "Play / Pause" button, run a tight
   ``while not done and st.session_state.playing`` loop inside one rerun,
   refreshing an ``st.empty()`` image placeholder at a configurable FPS. The
   loop yields between frames via ``time.sleep``; the user can pause via a
   Stop button or by triggering another rerun (e.g. changing a widget).

The Stop button works because Streamlit checks for new interactions between
loop iterations; toggling ``st.session_state.playing`` short-circuits the
next iteration. We DELETE the episode on stop or done to free the env.

Session state
-------------
``ep`` carries the active episode dict {episode_id, checkpoint_id, started_at}
or ``None`` between renders. ``playing`` is the run/pause flag.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_FRONTEND_DIR = Path(__file__).resolve().parents[1]
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))

import streamlit as st  # noqa: E402
from app import render_sidebar  # noqa: E402
from components.api_client import (  # noqa: E402
    BackendError,
    end_episode,
    get_episode_frame,
    get_episode_state,
    list_checkpoints,
    start_episode,
)

st.set_page_config(page_title="Play · retro-rl", layout="wide")
render_sidebar()
st.title("Play")
st.caption("Load a checkpoint, run a rollout, watch the agent act.")


# ---- session state init
if "ep" not in st.session_state:
    st.session_state.ep = None  # dict or None
if "playing" not in st.session_state:
    st.session_state.playing = False


# ---- catalog
try:
    ckpts = list_checkpoints()
except BackendError as e:
    st.error(f"Failed to load checkpoints: {e}")
    st.stop()

if not ckpts:
    st.info("No checkpoints available. Run training first.")
    st.stop()


def _checkpoint_label(c: dict) -> str:
    ret = f"{c['eval_return']:.1f}" if c.get("eval_return") is not None else "—"
    ln = f"{c['eval_length']:.0f}" if c.get("eval_length") is not None else "—"
    return f"{c['id']}  ·  step {c['step']:,}  ·  return {ret}  ·  len {ln}"


def _top_k_by_metric(all_ckpts: list[dict], metric_key: str, k: int = 5) -> list[dict]:
    """Per run, the ``k`` checkpoints with the highest ``metric_key``.

    ``metric_key`` is ``eval_return`` (mean reward) or ``eval_length`` (mean
    episode length). Checkpoints lacking that metric (eval never ran at that
    step) can't be ranked, so they're dropped. Runs are ordered by their best
    value so the strongest run — and the global-best checkpoint — sits at
    index 0.

    Note: this ranks only checkpoints still on disk. The trainer keeps
    ``keep_last_k`` step checkpoints + ``best``, so a high-scoring early
    checkpoint that was pruned can't reappear here.
    """
    by_run: dict[str, list[dict]] = {}
    for c in all_ckpts:
        if c.get(metric_key) is None:
            continue
        by_run.setdefault(c["run_name"], []).append(c)
    runs_ranked = sorted(
        by_run.values(),
        key=lambda lst: max(c[metric_key] for c in lst),
        reverse=True,
    )
    out: list[dict] = []
    for lst in runs_ranked:
        out.extend(sorted(lst, key=lambda c: c[metric_key], reverse=True)[:k])
    return out


# ---- controls
ctrl_col, opts_col = st.columns([2, 1])
with ctrl_col:
    sort_metric = st.radio(
        "Rank checkpoints by",
        ["mean_reward", "mean_length"],
        horizontal=True,
        disabled=st.session_state.ep is not None,
    )
    metric_key = "eval_return" if sort_metric == "mean_reward" else "eval_length"
    playable = _top_k_by_metric(ckpts, metric_key, k=5)
    if not playable:
        st.warning(f"No checkpoints carry an {sort_metric} value to rank by — showing all.")
        playable = ckpts
    selected_idx = st.selectbox(
        f"Checkpoint (top 5 per run by {sort_metric})",
        range(len(playable)),
        format_func=lambda i: _checkpoint_label(playable[i]),
        index=0,
        disabled=st.session_state.ep is not None,
    )
    selected = playable[selected_idx]
with opts_col:
    seed = st.number_input("Seed (optional)", min_value=0, value=0, step=1)
    use_seed = st.checkbox("Use seed", value=False)
    deterministic = st.checkbox("Deterministic policy", value=True)
    fps = st.slider("Playback FPS", 1, 30, 10)
    max_steps = st.number_input("Max steps (0 = use env default)", min_value=0, value=0)


# ---- action buttons
b1, b2, b3 = st.columns([1, 1, 1])
start_clicked = b1.button(
    "Start episode",
    use_container_width=True,
    disabled=st.session_state.ep is not None,
)
stop_clicked = b2.button(
    "Stop / End episode",
    use_container_width=True,
    disabled=st.session_state.ep is None,
)
pause_clicked = b3.button(
    "Pause" if st.session_state.playing else "Play",
    use_container_width=True,
    disabled=st.session_state.ep is None,
)


if start_clicked:
    try:
        st.session_state.ep = start_episode(
            checkpoint_id=selected["id"],
            seed=int(seed) if use_seed else None,
            deterministic=deterministic,
            max_steps=int(max_steps) if max_steps > 0 else None,
        )
        st.session_state.playing = True
        st.rerun()
    except BackendError as e:
        st.error(f"start_episode failed: {e}")

if stop_clicked and st.session_state.ep is not None:
    ep_id = st.session_state.ep["episode_id"]
    try:
        end_episode(ep_id)
    except BackendError as e:
        st.warning(f"end_episode failed (will drop locally): {e}")
    st.session_state.ep = None
    st.session_state.playing = False
    st.rerun()

if pause_clicked and st.session_state.ep is not None:
    st.session_state.playing = not st.session_state.playing
    st.rerun()


# ---- viewport
if st.session_state.ep is None:
    st.info("Pick a checkpoint and press **Start episode**.")
    st.stop()

ep_id = st.session_state.ep["episode_id"]
st.caption(f"episode `{ep_id}` on `{st.session_state.ep['checkpoint_id']}`")

# Constrain the viewport to ~70% of page width (14/20), centered with equal
# 15% gutters. `use_container_width=True` inside the middle column then
# scales with window.
_left, _mid, _right = st.columns([3, 14, 3])
with _mid:
    frame_slot = st.empty()
state_slot = st.empty()


def _render_state(s: dict) -> None:
    done_badge = ":red[done]" if s.get("done") else ":green[live]"
    state_slot.markdown(
        f"**step** `{s['step']:,}` · **reward** `{s['total_reward']:.2f}` · "
        f"**last_action** `{s.get('last_action')}` · {done_badge}"
    )


# Always render the current frame first (so a Pause holds the last frame visible).
try:
    initial_state = get_episode_state(ep_id)
    _render_state(initial_state)
    frame_slot.image(get_episode_frame(ep_id), use_container_width=True)
except BackendError as e:
    st.error(f"Failed to fetch initial state/frame: {e}")
    st.session_state.ep = None
    st.stop()


# ---- streaming loop (only while playing and not done)
period = 1.0 / max(fps, 1)
if st.session_state.playing and not initial_state["done"]:
    while st.session_state.playing:
        loop_start = time.monotonic()
        try:
            png = get_episode_frame(ep_id)  # advances one step + returns frame
            state = get_episode_state(ep_id)
        except BackendError as e:
            st.error(f"stream interrupted: {e}")
            st.session_state.playing = False
            break
        frame_slot.image(png, use_container_width=True)
        _render_state(state)
        if state["done"]:
            st.session_state.playing = False
            st.success(
                f"episode finished at step {state['step']:,}, total reward "
                f"{state['total_reward']:.2f}"
            )
            break
        elapsed = time.monotonic() - loop_start
        sleep_for = period - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)
