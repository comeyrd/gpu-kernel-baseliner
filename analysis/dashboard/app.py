"""
Baseliner Dashboard — entrypoint.
Run with: streamlit run app.py
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parent.parent
sys.path.append(str(parent))

import streamlit as st
from baselinerdataframe import ReportDataframe

# Pages
from dashboard.pages.benchmark_page import render as render_benchmark
from dashboard.pages.run_page import render as render_run


st.set_page_config(page_title="Baseliner Dashboard", layout="wide")

# ------------------------------------------------------------------
# File loader (cached so it only parses once per file)
# ------------------------------------------------------------------

@st.cache_resource
def load_report(path: str) -> ReportDataframe:
    return ReportDataframe(path)


# ------------------------------------------------------------------
# Sidebar — file picker + page navigation
# ------------------------------------------------------------------

with st.sidebar:
    st.title("Baseliner")
    report_path = st.text_input("Report JSON path", value="sample-data/small-result.json")
    page = st.radio("Page", ["Benchmark", "Run"])

rdf = load_report(report_path)

# ------------------------------------------------------------------
# Page routing
# ------------------------------------------------------------------

if page == "Benchmark":
    render_benchmark(rdf)
elif page == "Run":
    render_run(rdf)