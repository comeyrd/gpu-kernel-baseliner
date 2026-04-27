"""
Benchmark Page
Shows all (campaign, workload, backend) benchmarks with sweep axis plots.
"""

import streamlit as st
from baselinerdataframe import ReportDataframe

import dashboard.data as D
import dashboard.components as C
import dashboard.pages.plots_benchmark as PB


def render(rdf: ReportDataframe):
    st.title("Benchmark Overview")

    for campaign in D.get_campaigns(rdf):
        cid = campaign["id"]
        C.campaign_header(campaign["name"], cid)

        sweep_axes = D.get_sweep_axes(rdf, cid)
        benchmarks = D.get_benchmarks(rdf, cid)

        for bench in benchmarks:
            workload = bench["workload"]
            backend = bench["backend"]
            bid = bench["benchmark_id"]

            with st.container(border=True):
                C.benchmark_header(workload, backend, bid)

                scalars_df = D.get_scalars(rdf, cid, workload, backend)
                vectors_df = D.get_vectors(rdf, cid, workload, backend)

                if not sweep_axes:
                    st.info("No sweep axes defined for this campaign.")
                    continue

                # One row per sweep axis
                for ax in sweep_axes:
                    axis_key = ax["full_key"]
                    other_axes = [a for a in sweep_axes if a["full_key"] != axis_key]

                    # Secondary axis selectors (fix other axes to a value)
                    fixed_filters = {}
                    if other_axes:
                        selector_cols = st.columns(len(other_axes))
                        for i, other in enumerate(other_axes):
                            other_key = other["full_key"]
                            values = D.get_axis_values(rdf, cid, workload, backend, other_key)
                            with selector_cols[i]:
                                chosen = C.axis_value_selector(
                                    other_key, values,
                                    key=f"{cid}_{bid}_{axis_key}_{other_key}"
                                )
                                fixed_filters[other_key] = chosen

                    # Apply fixed filters to both dfs
                    s_df = _apply_filters(scalars_df, fixed_filters)
                    v_df = _apply_filters(vectors_df, fixed_filters)

                    st.markdown(f"**Sweep axis: `{axis_key}`**")
                    col_a, col_b = st.columns(2)

                    with col_a:
                        fig = PB.plot_benchmark_line(s_df, axis_key)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.caption("No line metrics available.")

                    with col_b:
                        fig = PB.plot_benchmark_box(v_df, axis_key)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.caption("No execution time vector available.")

        st.divider()


def _apply_filters(df, filters: dict):
    """Filter a dataframe by fixed secondary axis values."""
    for col, val in filters.items():
        if col in df.columns:
            df = df[df[col] == val]
    return df