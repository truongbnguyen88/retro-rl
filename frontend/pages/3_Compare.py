"""Compare page — overlay return curves across multiple runs.

Designed for the v5/v6/v7/v8 ablation comparison; works for any set of runs
the backend can serve metrics for.
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
from components.plots import empty_placeholder, multi_line_chart  # noqa: E402

st.set_page_config(page_title="Compare · retro-rl", layout="wide")
render_sidebar()
st.title("Compare runs")
st.caption("Overlay scalar curves across runs to ablate hyperparameter changes.")


try:
    runs = list_runs()
except BackendError as e:
    st.error(f"Failed to load runs: {e}")
    st.stop()

if not runs:
    st.info("No runs to compare.")
    st.stop()

run_names = [r["run_name"] for r in runs]

# Default selection: latest 4 runs (most recent ablations are usually the interesting ones).
default = run_names[-4:] if len(run_names) > 4 else run_names

col1, col2 = st.columns([4, 1])
selected = col1.multiselect("Runs to overlay", run_names, default=default)
if col2.button("Refresh", use_container_width=True):
    clear_catalog_cache()
    st.rerun()

if not selected:
    st.info("Pick at least one run.")
    st.stop()

# Metric picker — drive everything from one shared series name so the overlay is meaningful.
metric_choices = [
    "eval/mean_return",
    "eval/mean_length",
    "eval/std_return",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/approx_kl",
    "train/ent_coef",
]
metric_name = st.selectbox("Metric", metric_choices, index=0)


# Fetch metrics for each selected run; tolerate missing TB dirs (e.g., very-new runs).
named_series: dict[str, list[dict[str, float]]] = {}
errors: list[tuple[str, str]] = []
for r in selected:
    try:
        m = get_run_metrics(r)
    except BackendError as e:
        errors.append((r, str(e)))
        continue
    named_series[r] = m.get(metric_name, [])

if errors:
    with st.expander("Some runs failed to load", expanded=False):
        for run, msg in errors:
            st.write(f"- `{run}`: {msg}")

if not any(named_series.values()):
    st.plotly_chart(
        empty_placeholder(f"No `{metric_name}` data in selected runs."),
        use_container_width=True,
    )
    st.stop()


st.plotly_chart(
    multi_line_chart(
        named_series,
        title=metric_name,
        x_label="env step",
        y_label=metric_name.split("/")[-1],
    ),
    use_container_width=True,
)


# ---- summary table: peak + final per run
st.subheader("Summary")
rows = []
for run, points in named_series.items():
    if not points:
        rows.append(
            {
                "run": run,
                "peak": "—",
                "peak_step": "—",
                "final": "—",
                "final_step": "—",
                "n_points": 0,
            }
        )
        continue
    peak = max(points, key=lambda p: p["value"])
    final = points[-1]
    rows.append(
        {
            "run": run,
            "peak": f"{peak['value']:.2f}",
            "peak_step": f"{peak['step']:,}",
            "final": f"{final['value']:.2f}",
            "final_step": f"{final['step']:,}",
            "n_points": len(points),
        }
    )
st.dataframe(rows, use_container_width=True, hide_index=True)
