"""
Data access helpers.
All dataframe logic lives here — pages never touch ReportDataframe internals directly.
"""

import pandas as pd
from baselinerdataframe import ReportDataframe 


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------

def get_campaigns(rdf: ReportDataframe) -> list[dict]:
    """Return list of {id, name} for all campaigns."""
    return [
        {"id": c.id, "name": c.name}
        for c in rdf.m_report.campaign_runs
    ]


def get_benchmarks(rdf: ReportDataframe, campaign_id: str) -> list[dict]:
    """Return list of {workload, backend, benchmark_id} for a campaign."""
    campaign = _get_campaign(rdf, campaign_id)
    result = []
    for backend, inner in campaign.benchmark_runs.items():
        for workload, bench_exec in inner.items():
            result.append({
                "workload": workload,
                "backend": backend,
                "benchmark_id": bench_exec.benchmark_report.id,
            })
    return result


def get_sweep_axes(rdf: ReportDataframe, campaign_id: str) -> list[dict]:
    """Return sweep axes [{interface, option, full_key}] for a campaign."""
    campaign = _get_campaign(rdf, campaign_id)
    if campaign.recipe.sweep is None:
        return []
    return [
        {
            "interface": ax.interface,
            "option": ax.option,
            "full_key": f"{ax.interface}.{ax.option}",
        }
        for ax in campaign.recipe.sweep.axes
    ]


def get_axis_values(rdf: ReportDataframe, campaign_id: str, workload: str, backend: str, axis_key: str) -> list:
    """Return sorted unique values for a sweep axis in a given benchmark."""
    df = _metadata(rdf, campaign_id, workload, backend)
    if axis_key not in df.columns:
        return []
    return sorted(df[axis_key].unique().tolist())


def get_run_ids(rdf: ReportDataframe, campaign_id: str, workload: str, backend: str) -> list[dict]:
    """Return list of {run_id, label} where label is the sweep point values."""
    df = _metadata(rdf, campaign_id, workload, backend)
    axes = get_sweep_axes(rdf, campaign_id)
    axis_keys = [ax["full_key"] for ax in axes if ax["full_key"] in df.columns]

    result = []
    for _, row in df.iterrows():
        label = " | ".join(f"{k}={row[k]}" for k in axis_keys) if axis_keys else row["run_id"]
        result.append({"run_id": row["run_id"], "label": label})
    return result


# ---------------------------------------------------------------------------
# Data fetchers (merge lazily, only when needed)
# ---------------------------------------------------------------------------

def get_scalars(rdf: ReportDataframe, campaign_id: str, workload: str, backend: str) -> pd.DataFrame:
    filtered = rdf.filter(campaign_id=campaign_id, workload=workload, backend=backend)
    return filtered.merge_scalars()


def get_vectors(rdf: ReportDataframe, campaign_id: str, workload: str, backend: str) -> pd.DataFrame:
    filtered = rdf.filter(campaign_id=campaign_id, workload=workload, backend=backend)
    return filtered.merge_vectors()


def get_scalars_for_run(rdf: ReportDataframe, run_id: str) -> pd.DataFrame:
    filtered = rdf.filter(run_id=run_id)
    return filtered.merge_scalars()


def get_vectors_for_run(rdf: ReportDataframe, run_id: str) -> pd.DataFrame:
    filtered = rdf.filter(run_id=run_id)
    return filtered.merge_vectors()


# ---------------------------------------------------------------------------
# Metric availability checks
# ---------------------------------------------------------------------------

def has_column(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def get_scalar_metrics(df: pd.DataFrame, exclude_suffixes=(".unit", ".granularity", "run_id")) -> list[str]:
    """Return scalar metric column names, excluding metadata/unit columns."""
    return [c for c in df.columns if not any(c.endswith(s) for s in exclude_suffixes)]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_campaign(rdf: ReportDataframe, campaign_id: str):
    for c in rdf.m_report.campaign_runs:
        if c.id == campaign_id:
            return c
    raise ValueError(f"Campaign '{campaign_id}' not found.")


def _metadata(rdf: ReportDataframe, campaign_id: str, workload: str, backend: str) -> pd.DataFrame:
    return rdf.filter(campaign_id=campaign_id, workload=workload, backend=backend).m_metadata_df