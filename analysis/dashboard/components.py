"""
Reusable Streamlit UI components.
Add new components here as the dashboard grows.
"""

import streamlit as st
import pandas as pd


def campaign_header(campaign_name: str, campaign_id: str):
    st.markdown(f"## {campaign_name}")
    st.caption(f"Campaign ID: {campaign_id}")


def benchmark_header(workload: str, backend: str, benchmark_id: str):
    st.markdown(f"**{workload} — {backend}**")
    st.caption(f"Benchmark ID: {benchmark_id}")


def card(content_fn, *args, **kwargs):
    """Wrap content in a visual card container."""
    with st.container(border=True):
        content_fn(*args, **kwargs)


def axis_value_selector(axis_key: str, available_values: list, key: str) -> str:
    """Inline selector for fixing a secondary sweep axis to a value."""
    return st.selectbox(
        label=axis_key,
        options=available_values,
        key=key,
    )


def scalar_summary_table(df: pd.DataFrame, sweep_axis_keys: list[str]):
    """
    Table comparing scalar metrics across runs.
    Rows = metrics, Columns = run sweep point labels.
    """
    if df.empty:
        st.info("No scalar data available.")
        return

    # Build a label per run from its sweep point values
    label_cols = [k for k in sweep_axis_keys if k in df.columns]
    if label_cols:
        df = df.copy()
        df["_label"] = df[label_cols].apply(
            lambda row: " | ".join(f"{k}={row[k]}" for k in label_cols), axis=1
        )
    else:
        df = df.copy()
        df["_label"] = df["run_id"]

    # Keep only numeric metric columns
    skip = {"run_id", "_label"} | set(sweep_axis_keys)
    metric_cols = [
        c for c in df.columns
        if c not in skip
        and not any(c.endswith(s) for s in (".unit", ".granularity"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not metric_cols:
        st.info("No numeric scalar metrics to display.")
        return

    pivot = df.set_index("_label")[metric_cols].T
    pivot.index.name = "Metric"
    st.dataframe(pivot, use_container_width=True)