"""Training page — single-run TB scalar curves.

Pick a run, see eval/rollout/train metrics live. Uses ``get_run_metrics``
with a 15 s TTL — refreshes happen on Streamlit reruns (button or auto-poll).
"""

from __future__ import annotations

import sys
from pathlib import Path

_FRONTEND_DIR = Path(__file__).resolve().parents[1]
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))

import streamlit as st  # noqa: E402
from app import render_sidebar  # noqa: E402
from components.api_client import (  # noqa: E402
    BackendError,
    clear_catalog_cache,
    get_run_metrics,
    list_runs,
)
from components.plots import empty_placeholder, line_chart  # noqa: E402

st.set_page_config(page_title="Training · retro-rl", layout="wide")
render_sidebar()
st.title("Training")
st.caption("Live scalar curves from the active run's TensorBoard log.")


try:
    runs = list_runs()
except BackendError as e:
    st.error(f"Failed to load runs: {e}")
    st.stop()

if not runs:
    st.info(
        "No runs yet — kick off training with `python scripts/train.py --config configs/ppo.yaml`."
    )
    st.stop()

# Default selection: prefer a run that has eval metrics; else the alphabetical last.
run_names = [r["run_name"] for r in runs]
default_idx = len(run_names) - 1

col1, col2 = st.columns([3, 1])
selected = col1.selectbox("Run", run_names, index=default_idx)
if col2.button("Refresh", use_container_width=True):
    clear_catalog_cache()
    st.rerun()


# ---- per-run summary panel
run_info = next(r for r in runs if r["run_name"] == selected)
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "best return",
    f"{run_info['best_return']:.2f}" if run_info["best_return"] is not None else "—",
)
m2.metric(
    "latest step",
    f"{run_info['latest_step']:,}" if run_info["latest_step"] is not None else "—",
)
m3.metric("checkpoints", run_info["checkpoint_count"])
m4.metric("has best", "yes" if run_info["has_best"] else "no")


# ---- metrics
try:
    metrics = get_run_metrics(selected)
except BackendError as e:
    if e.status == 404:
        st.warning(
            f"No TensorBoard log directory found for `{selected}`. "
            "The run may not have flushed scalars yet — try again in a minute."
        )
    else:
        st.error(f"Failed to load metrics: {e}")
    st.stop()


def _plot_panel(title: str, series_keys: list[str]) -> None:
    st.subheader(title)
    cols = st.columns(2)
    for i, key in enumerate(series_keys):
        with cols[i % 2]:
            points = metrics.get(key)
            if points:
                st.plotly_chart(
                    line_chart(points, title=key, y_label=key.split("/")[-1]),
                    use_container_width=True,
                )
            else:
                st.plotly_chart(
                    empty_placeholder(f"no `{key}` data yet"),
                    use_container_width=True,
                )


_plot_panel("Eval", ["eval/mean_return", "eval/mean_length", "eval/std_return"])
_plot_panel("Rollout", ["rollout/ep_rew_mean", "rollout/ep_len_mean"])
_plot_panel(
    "Train",
    [
        "train/approx_kl",
        "train/entropy_loss",
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/clip_fraction",
        "train/ent_coef",
    ],
)


with st.expander("All available series", expanded=False):
    st.write([{"name": n, "points": len(p)} for n, p in metrics.items()])
