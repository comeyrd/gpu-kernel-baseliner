"""
Plot builders for the Benchmark page.
Each function takes a dataframe and returns a go.Figure (or None if data missing).
Add new plot types here as new metrics appear.
"""

import pandas as pd
import plotly.graph_objects as go
from baselinerplots import line_plot, box_plot


# ---------------------------------------------------------------------------
# Plot A — line plot of scalar metrics vs sweep axis
# ---------------------------------------------------------------------------

BENCHMARK_LINE_METRICS = ["mean", "median"]  # extend this list to add more lines


def plot_benchmark_line(df: pd.DataFrame, axis_key: str) -> go.Figure | None:
    """
    Line plot of ONCE/ON_DEMAND scalar metrics vs a sweep axis.
    Returns None if none of the expected metrics are present.
    """
    available = [m for m in BENCHMARK_LINE_METRICS if m in df.columns]
    if not available or axis_key not in df.columns:
        return None

    return line_plot(df, x=axis_key, y=available, title="Performance vs sweep axis")


# ---------------------------------------------------------------------------
# Plot B — box plot of execution time vs sweep axis
# ---------------------------------------------------------------------------

EXECUTION_TIME_COL = "execution_time_vector"


def plot_benchmark_box(df: pd.DataFrame, axis_key: str) -> go.Figure | None:
    """
    Box plot of execution time distribution per sweep axis value.
    Returns None if execution_time_vector is not present.
    """
    if EXECUTION_TIME_COL not in df.columns or axis_key not in df.columns:
        return None

    return box_plot(df, y=EXECUTION_TIME_COL, x=axis_key, title="Execution time distribution")