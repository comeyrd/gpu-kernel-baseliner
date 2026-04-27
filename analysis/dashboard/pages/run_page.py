"""
Run Page
User selects (campaign, workload, backend) then sees one card per run_id.
"""

import streamlit as st
from baselinerdataframe import ReportDataframe

import dashboard.data as D
import dashboard.components as C
import dashboard.pages.plots_run as PR


def render(rdf: ReportDataframe):
    st.title("Run Inspector")

    # ------------------------------------------------------------------
    # Selectors
    # ------------------------------------------------------------------
    campaigns = D.get_campaigns(rdf)
    campaign = st.selectbox(
        "Campaign",
        campaigns,
        format_func=lambda c: c["name"],
        key="run_page_campaign",
    )
    cid = campaign["id"]

    benchmarks = D.get_benchmarks(rdf, cid)
    bench = st.selectbox(
        "Workload — Backend",
        benchmarks,
        format_func=lambda b: f"{b['workload']} — {b['backend']}",
        key="run_page_bench",
    )
    workload, backend, bid = bench["workload"], bench["backend"], bench["benchmark_id"]

    st.divider()

    # ------------------------------------------------------------------
    # Scalar summary table (once, above all run cards)
    # ------------------------------------------------------------------
    scalars_df = D.get_scalars(rdf, cid, workload, backend)
    sweep_axes = D.get_sweep_axes(rdf, cid)
    sweep_keys = [ax["full_key"] for ax in sweep_axes]

    st.markdown("### Scalar Summary")
    C.scalar_summary_table(scalars_df, sweep_keys)

    st.divider()

    # ------------------------------------------------------------------
    # One card per run
    # ------------------------------------------------------------------
    runs = D.get_run_ids(rdf, cid, workload, backend)
    first = True
    for run in runs:
        run_id = run["run_id"]
        label = run["label"]

        with st.expander(f"**Run: {label}** (ID: {run_id})", expanded=first):
            vectors_df = D.get_vectors_for_run(rdf, run_id)
            
            if not vectors_df.empty:
                _render_run_plots(vectors_df)
            else:
                st.warning("No vector data available for this run.")
        first = False

def _render_run_plots(df):
    """
    3-column layout per run card.
    Each plot is only shown if the required data is present.
    Add new plots here by calling new PR.plot_* functions.
    """
    col_a, col_b, col_c = st.columns(3)
    
    # Row 1
    with col_a:
        fig = PR.plot_run_element_lines(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = PR.plot_run_batch_lines(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col_c:
        fig = PR.plot_run_box(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Row 2 — ECDF spans first column, extend with new plots in col_b/col_c
    col_d, col_e, _ = st.columns(3)

    with col_d:
        fig = PR.plot_run_ecdf(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)