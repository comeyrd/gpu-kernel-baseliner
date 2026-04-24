import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ._theme import COLORS, apply_theme


def line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    y2: str | None = None,
    hue: str | None = None,
    title: str | None = None,
    markers: bool = False,
    layout_override: dict | None = None,
) -> go.Figure:
    mode = "lines+markers" if markers else "lines"
    colors = 0
    if y2 is not None:
        fig = go.Figure()
        if isinstance(y,list):
            for ins_y in y:
                fig.add_trace(go.Scatter(x=df[x], y=df[ins_y], mode=mode, name=ins_y , line=dict(color=COLORS[colors])))
                colors+=1
        else:
            fig.add_trace(go.Scatter(x=df[x], y=df[y], mode=mode, name=y, line=dict(color=COLORS[colors])))
            colors+=1
        if isinstance(y2,list):
            for ins_y2 in y2:
                fig.add_trace(go.Scatter(x=df[x], y=df[ins_y2], mode=mode, name=ins_y2, line=dict(color=COLORS[colors]), yaxis="y2"))
                colors+=1
        else:
            fig.add_trace(go.Scatter(x=df[x], y=df[y2], mode=mode, name=y2, line=dict(color=COLORS[colors]), yaxis="y2"))
            colors+=1
        
        fig.update_layout(yaxis2=dict(title=y2, overlaying="y", side="right", showgrid=False))
    else:
        fig = px.line(df, x=x, y=y, color=hue, markers=markers, color_discrete_sequence=COLORS)

    apply_theme(fig, title=title)

    if layout_override:
        fig.update_layout(**layout_override)

    return fig