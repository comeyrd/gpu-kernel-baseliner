import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ._theme import COLORS, apply_theme


def ecdf_plot(
    df: pd.DataFrame,
    x: str,
    hue: str | None = None,
    title: str | None = None,
    markers: bool = False,
    layout_override: dict | None = None,
) -> go.Figure:
    fig = px.ecdf(df, x=x, color=hue, markers=markers, color_discrete_sequence=COLORS)

    apply_theme(fig, title=title)

    if layout_override:
        fig.update_layout(**layout_override)

    return fig