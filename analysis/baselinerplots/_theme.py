"""
Shared theme, colors, and layout defaults for all plots.
Edit this file to restyle the entire library at once.
"""

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

# Ordered sequence used for hue / multi-series
COLORS = [
    "#4C78A8",  # muted blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#E45756",  # red
    "#72B7B2",  # teal
    "#B279A2",  # purple
    "#FF9DA6",  # pink
    "#9D755D",  # brown
]

# Single-series default
DEFAULT_COLOR = COLORS[0]

# -----------------------------------
# ----------------------------------------
# Base layout applied to every figure
# ---------------------------------------------------------------------------

BASE_LAYOUT = dict(
    font=dict(family="IBM Plex Sans, sans-serif", size=13, color="#1a1a2e"),
    paper_bgcolor="white",
    plot_bgcolor="#f9f9fb",
    margin=dict(t=60, b=60, l=70, r=30),
    legend=dict(
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#e0e0e0",
        borderwidth=1,
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="#e8e8f0",
        linecolor="#cccccc",
        zerolinecolor="#cccccc",
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#e8e8f0",
        linecolor="#cccccc",
        zerolinecolor="#cccccc",
    ),
)


def apply_theme(fig, title: str | None = None) -> None:
    """Apply the base layout to a figure in-place."""
    fig.update_layout(**BASE_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15, color="#1a1a2e")))