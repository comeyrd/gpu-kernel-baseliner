"""
Plot builders for the Run page.
Each function takes a dataframe and returns a go.Figure (or None if data missing).
Add new plot types here as new metrics appear.
"""

import pandas as pd
import plotly.graph_objects as go
from baselinerplots import line_plot, box_plot, ecdf_plot


# ---------------------------------------------------------------------------
# Plot A — EVERY_ELEMENT metrics vs run_nb
# ---------------------------------------------------------------------------

EVERY_ELEMENT_METRICS = ["execution_time_vector", "sorted_execution_time_vector"]


def plot_run_element_lines(df: pd.DataFrame) -> go.Figure | None:
    """Line plot of EVERY_ELEMENT metrics vs run_nb."""
    available = [m for m in EVERY_ELEMENT_METRICS if m in df.columns]
    if not available or "run_nb" not in df.columns:
        return None

    melted = df[["run_nb"] + available].melt(
        id_vars="run_nb", value_vars=available, var_name="metric", value_name="value"
    )
    return line_plot(melted, x="run_nb", y="value", hue="metric", title="Execution time per element")


# ---------------------------------------------------------------------------
# Plot B — EVERY_BATCH metrics vs batch_nb
# ---------------------------------------------------------------------------

EVERY_BATCH_METRICS = ["batch_time_vector", "clock_frequency_vector", "temperature_vector", "power_utilization_vector"]


def plot_run_batch_lines(df: pd.DataFrame) -> go.Figure | None:
    """Line plot of EVERY_BATCH metrics vs batch_nb."""
    available = [m for m in EVERY_BATCH_METRICS if m in df.columns]
    if not available or "batch_nb" not in df.columns:
        return None

    melted = df[["batch_nb"] + available].melt(
        id_vars="batch_nb", value_vars=available, var_name="metric", value_name="value"
    )
    return line_plot(melted, x="batch_nb", y="value", hue="metric", title="Batch metrics")


# ---------------------------------------------------------------------------
# Plot C — Box plot of execution times
# ---------------------------------------------------------------------------

def plot_run_box(df: pd.DataFrame) -> go.Figure | None:
    """Box plot of execution time distribution for this run."""
    if "execution_time_vector" not in df.columns:
        return None
    return box_plot(df, y="execution_time_vector", title="Execution time distribution",show_points=True,notch=True)


# ---------------------------------------------------------------------------
# Plot D — ECDF of execution times (+ without_outliers if present)
# ---------------------------------------------------------------------------

ECDF_METRICS = ["execution_time_vector", "without_outliers"]


def plot_run_ecdf(df: pd.DataFrame) -> go.Figure | None:
    """ECDF of execution time, optionally overlaid with without_outliers."""
    available = [m for m in ECDF_METRICS if m in df.columns]
    if not available:
        return None

    melted = df[available].melt(var_name="metric", value_name="value").dropna()
    return ecdf_plot(melted, x="value", hue="metric", title="ECDF — execution time")