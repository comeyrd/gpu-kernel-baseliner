import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ._theme import COLORS, apply_theme


def box_plot(
    df: pd.DataFrame,
    y: str,
    x: str | None = None,
    notch: bool = False,
    title: str | None = None,
    show_points: bool = False,
    layout_override: dict | None = None,
) -> go.Figure:
    fig = px.box(df, x=x, y=y, notched=notch, points="all" if show_points else False,
                 color_discrete_sequence=COLORS)

    apply_theme(fig, title=title)

    if layout_override:
        fig.update_layout(**layout_override)

    return fig