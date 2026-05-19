"""retro-rl dashboard — landing page.

Streamlit auto-discovers ``frontend/pages/*.py`` and renders them in the
sidebar. This file is the entry point (``streamlit run frontend/app.py``)
and hosts:

* a top-level summary of available runs + checkpoints (cheap calls);
* a sidebar backend health probe shown on every page (via :func:`render_sidebar`).

Per CLAUDE.md, the frontend talks to the backend over HTTP only — every
backend call lives in :mod:`frontend.components.api_client`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit launches with cwd = repo root (``streamlit run frontend/app.py``),
# but doesn't put ``frontend/`` on sys.path; pages can't ``from components...``
# without help. Insert the frontend dir explicitly.
_FRONTEND_DIR = Path(__file__).resolve().parent
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))

import streamlit as st  # noqa: E402
from components.api_client import (  # noqa: E402
    BackendError,
    backend_url,
    clear_catalog_cache,
    get_health,
    list_checkpoints,
    list_runs,
)

st.set_page_config(
    page_title="retro-rl dashboard",
    page_icon=":video_game:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> None:
    """Health probe + backend URL display. Imported by every page."""
    with st.sidebar:
        st.markdown("### Backend")
        st.code(backend_url(), language="text")
        try:
            h = get_health()
            st.success(
                f"OK · v{h['version']} · up {h['uptime_seconds']:.0f}s",
                icon=":material/check_circle:",
            )
        except BackendError as e:
            st.error(f"backend unreachable: {e}", icon=":material/error:")
            st.caption("Start it with `python scripts/serve.py`.")
        if st.button("Refresh catalog cache", use_container_width=True):
            clear_catalog_cache()
            st.rerun()


def _runs_summary_table(runs: list[dict]) -> None:
    """Lightweight overview table — no plotting, fast to load."""
    if not runs:
        st.info("No training runs found in `outputs/checkpoints/`.")
        return
    rows = []
    for r in runs:
        rows.append(
            {
                "run": r["run_name"],
                "best_return": (f"{r['best_return']:.2f}" if r["best_return"] is not None else "—"),
                "latest_step": (f"{r['latest_step']:,}" if r["latest_step"] is not None else "—"),
                "checkpoints": r["checkpoint_count"],
                "has_best": "yes" if r["has_best"] else "no",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    render_sidebar()
    st.title("retro-rl dashboard")
    st.caption(
        "Inspect training runs, replay learned policies frame-by-frame, and "
        "compare returns across experiments. Use the sidebar to navigate."
    )

    try:
        runs = list_runs()
        ckpts = list_checkpoints()
    except BackendError as e:
        st.error(f"Failed to load catalog: {e}")
        st.stop()

    col1, col2 = st.columns(2)
    col1.metric("Runs", len(runs))
    col2.metric("Checkpoints", len(ckpts))

    st.subheader("Runs")
    _runs_summary_table(runs)

    with st.expander("All checkpoints", expanded=False):
        if not ckpts:
            st.write("No checkpoints discovered.")
        else:
            st.dataframe(
                [
                    {
                        "id": c["id"],
                        "step": f"{c['step']:,}",
                        "eval_return": (
                            f"{c['eval_return']:.2f}" if c["eval_return"] is not None else "—"
                        ),
                        "timestamp": c["timestamp"],
                    }
                    for c in ckpts
                ],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown(
        """
        ### Pages
        - **Training** — single-run TensorBoard scalar curves.
        - **Play** — load a checkpoint and watch the agent act, frame by frame.
        - **Compare** — overlay return curves across multiple runs.
        """
    )


if __name__ == "__main__":
    main()
