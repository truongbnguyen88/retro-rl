"""Plotly helpers shared by the Training and Compare pages.

All functions return a :class:`plotly.graph_objects.Figure` ready to be passed
to ``st.plotly_chart(fig, use_container_width=True)``. Layout is consistent
(dark theme, transparent background) so plots stitch into the Streamlit dark
theme without seams.
"""

from __future__ import annotations

from typing import Iterable

import plotly.graph_objects as go


_LAYOUT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=40),
    hovermode="x unified",
)


def line_chart(
    points: list[dict[str, float]],
    *,
    title: str,
    x_label: str = "step",
    y_label: str = "value",
    line_color: str | None = None,
) -> go.Figure:
    """Single-series line chart from a list of ``{"step": int, "value": float}``."""
    xs = [p["step"] for p in points]
    ys = [p["value"] for p in points]
    fig = go.Figure(
        data=go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(width=2, color=line_color) if line_color else dict(width=2),
            marker=dict(size=5),
            name=title,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        **_LAYOUT_BASE,
    )
    return fig


def multi_line_chart(
    named_series: dict[str, list[dict[str, float]]],
    *,
    title: str,
    x_label: str = "step",
    y_label: str = "value",
) -> go.Figure:
    """Overlay multiple named series on one chart.

    Used by the Compare page for v5/v6/v7/v8 side-by-side returns.
    """
    fig = go.Figure()
    for name, points in named_series.items():
        if not points:
            continue
        xs = [p["step"] for p in points]
        ys = [p["value"] for p in points]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers", name=name,
                line=dict(width=2), marker=dict(size=4),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_LAYOUT_BASE,
    )
    return fig


def empty_placeholder(message: str) -> go.Figure:
    """A blank figure with a centered message — for missing-data states."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        **_LAYOUT_BASE,
    )
    return fig


def filter_series(
    metrics: dict[str, list[dict[str, float]]],
    prefixes: Iterable[str],
) -> dict[str, list[dict[str, float]]]:
    """Return only series whose name starts with one of *prefixes*."""
    prefix_tuple = tuple(prefixes)
    return {name: pts for name, pts in metrics.items() if name.startswith(prefix_tuple)}


__all__ = [
    "line_chart",
    "multi_line_chart",
    "empty_placeholder",
    "filter_series",
]
