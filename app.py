"""Streamlit Web Application: Cardiovascular Disease Risk Prediction Platform.

Academic & Clinical Machine-Learning Interface for Cardiovascular Disease Stratification.
Evaluated on the 70,000-patient CardioTrain cohort with 4 classical architectures:
Random Forest, Logistic Regression, Decision Tree, and K-Nearest Neighbours.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import PROJECT_ROOT, load_config
from src.data_loader import get_upload_template
from src.predict import (
    load_artifacts,
    predict_from_features,
    predict_patient,
    validate_input,
)

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Disease Risk Prediction Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CLINICAL SLATE DESIGN SYSTEM & CSS
# ==============================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Root Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .font-mono {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Backgrounds */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #080d1a 100%);
        color: #f1f5f9;
    }

    /* Streamlit Sidebar Dark Slate Override */
    [data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* Header Banner */
    .app-header-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 3px solid #0ea5e9;
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 0 15px rgba(14, 165, 233, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .app-title-group h1 {
        margin: 0;
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .app-title-group p {
        margin: 0.35rem 0 0 0;
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 400;
    }

    .badge-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.4rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    .badge-chip-cyan {
        background: rgba(14, 165, 233, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.35);
    }

    .badge-chip-emerald {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #38bdf8;
        box-shadow: 0 0 8px #38bdf8;
        display: inline-block;
    }

    /* Soft Dismissible Notice */
    .notice-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 3px solid #38bdf8;
        border-radius: 10px;
        padding: 0.75rem 1.25rem;
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Risk Outcome Cards */
    .risk-banner {
        padding: 1.6rem;
        border-radius: 16px;
        text-align: left;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }

    .risk-banner-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(136, 19, 55, 0.12) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.15);
    }

    .risk-banner-moderate {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(146, 64, 14, 0.12) 100%);
        border: 1px solid rgba(245, 158, 11, 0.45);
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.15);
    }

    .risk-banner-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(6, 78, 59, 0.12) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.15);
    }

    .risk-banner-tag {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }

    .tag-high { background: rgba(239, 68, 68, 0.25); color: #fecdd3; border: 1px solid rgba(239, 68, 68, 0.4); }
    .tag-moderate { background: rgba(245, 158, 11, 0.25); color: #fef3c7; border: 1px solid rgba(245, 158, 11, 0.4); }
    .tag-low { background: rgba(16, 185, 129, 0.25); color: #a7f3d0; border: 1px solid rgba(16, 185, 129, 0.4); }

    .risk-headline {
        font-size: 1.65rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        color: #ffffff;
    }

    .risk-body-desc {
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.5;
        margin: 0;
    }

    /* Clinical Patient Demographics Card */
    .patient-profile-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .profile-item {
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
    }

    .profile-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .profile-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
    }

    /* Custom KPI Metric Cards */
    .vital-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.3);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }

    .vital-card:hover {
        transform: translateY(-2px);
        border-color: rgba(14, 165, 233, 0.3);
    }

    .vital-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }

    .vital-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .vital-value-row {
        display: flex;
        align-items: baseline;
        gap: 0.35rem;
        margin: 0.2rem 0;
    }

    .vital-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #f8fafc;
    }

    .vital-unit {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 500;
    }

    .vital-delta-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.74rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        width: fit-content;
        margin-bottom: 0.45rem;
    }

    .delta-red { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .delta-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .delta-neutral { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .delta-slate { background: rgba(100, 116, 139, 0.15); color: #cbd5e1; border: 1px solid rgba(100, 116, 139, 0.3); }

    .vital-ref-tag {
        font-size: 0.74rem;
        color: #94a3b8;
        border-top: 1px dashed rgba(255, 255, 255, 0.08);
        padding-top: 0.4rem;
        margin-top: 0.2rem;
    }

    /* Consultation Summary & Care Plan */
    .summary-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
    }

    .plan-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.15rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.25);
    }

    .plan-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 700;
        font-size: 0.98rem;
        color: #f8fafc;
        margin-bottom: 0.75rem;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }

    .plan-item {
        font-size: 0.86rem;
        color: #cbd5e1;
        line-height: 1.55;
        margin-bottom: 0.55rem;
    }

    .driver-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }
    .driver-chip-red { background: rgba(239, 68, 68, 0.18); color: #fecdd3; border: 1px solid rgba(239, 68, 68, 0.4); }
    .driver-chip-amber { background: rgba(245, 158, 11, 0.18); color: #fef3c7; border: 1px solid rgba(245, 158, 11, 0.4); }
    .driver-chip-green { background: rgba(16, 185, 129, 0.18); color: #a7f3d0; border: 1px solid rgba(16, 185, 129, 0.4); }

    /* Model Champion Spotlight Banner */
    .champion-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 25px rgba(14, 165, 233, 0.08);
    }

    /* Styled Tab overrides */
    div[data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding-bottom: 4px;
        margin-bottom: 1.2rem;
    }

    div[data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 18px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        color: #94a3b8 !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.2s ease;
    }

    div[data-baseweb="tab"]:hover {
        color: #38bdf8 !important;
        background: rgba(255, 255, 255, 0.03) !important;
    }

    div[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
        background: rgba(14, 165, 233, 0.06) !important;
    }

    /* Streamlit Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%) !important;
        border: 1px solid #38bdf8 !important;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.3) !important;
        color: #ffffff !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.45) !important;
    }

    /* Streamlit Metric Container Override */
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    /* Viva Defense Accordions */
    .viva-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# RISK GAUGE & VISUALIZATION BUILDERS
# ==============================================================================
def create_risk_gauge(prob: float, risk_level: str) -> go.Figure:
    """Create a semi-circular clinical cardiovascular risk gauge."""
    pct = round(prob * 100.0, 1)
    bar_color = (
        "#f43f5e"
        if risk_level == "High"
        else ("#f59e0b" if risk_level == "Moderate" else "#10b981")
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={
                "suffix": "%",
                "font": {
                    "size": 42,
                    "color": "#f8fafc",
                    "family": "Plus Jakarta Sans",
                },
            },
            title={
                "text": "<b>CVD Probability</b><br><span style='font-size:12px;color:#94a3b8;'>Model Risk Estimate</span>",
                "font": {
                    "size": 13,
                    "color": "#cbd5e1",
                    "family": "Plus Jakarta Sans",
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#475569",
                    "tickvals": [0, 20, 35, 60, 80, 100],
                    "ticktext": ["0%", "20%", "35% (Mod)", "60% (High)", "80%", "100%"],
                    "tickfont": {"size": 10, "color": "#94a3b8"},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(255, 255, 255, 0.03)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.20)"},
                    {"range": [35, 60], "color": "rgba(245, 158, 11, 0.20)"},
                    {"range": [60, 100], "color": "rgba(244, 63, 94, 0.20)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.8,
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=25, r=25, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans, sans-serif"},
    )
    return fig


def create_model_comparison_bar_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Create interactive multi-metric grouped bar chart comparing all models."""
    records = []
    for model_key, data in metrics.items():
        name = data["display_name"]
        test_m = data.get("test_metrics", {})
        for metric_name, val in test_m.items():
            if val is not None:
                records.append({
                    "Model": name,
                    "Metric": metric_name.upper(),
                    "Score": round(val * 100, 2),
                })
    df = pd.DataFrame(records)

    colors = {
        "Random Forest": "#38bdf8",
        "Logistic Regression": "#818cf8",
        "Decision Tree": "#fbbf24",
        "K-Nearest Neighbours": "#34d399",
    }

    fig = px.bar(
        df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text="Score",
        color_discrete_map=colors,
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        textfont=dict(size=10, color="#f8fafc", family="Plus Jakarta Sans"),
        cliponaxis=False,
    )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=25, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.45)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#cbd5e1", size=11),
        ),
        xaxis=dict(
            title="",
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(color="#f1f5f9", size=12, family="Plus Jakarta Sans"),
        ),
        yaxis=dict(
            title="Score (%)",
            range=[0, 105],
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#94a3b8", size=11),
        ),
    )
    return fig


@st.cache_resource
def get_cached_artifacts():
    """Cache trained model pipeline and metadata."""
    try:
        cfg = load_config()
        model, metadata, metrics = load_artifacts()
        return cfg, model, metadata, metrics, None
    except Exception as e:
        return None, None, None, None, str(e)


# ==============================================================================
# MAIN APPLICATION CONTROLLER
# ==============================================================================
def main():
    cfg, model, metadata, metrics, err = get_cached_artifacts()

    if err:
        st.error(f"⚠️ **Artifact Loading Error:** {err}")
        st.info("Execute training to regenerate model artifacts:\n`python -m src.train`")
        return

    # App Header with Medical Accent
    st.markdown(
        """
        <div class="app-header-container">
            <div class="app-title-group">
                <h1>🩺 Cardiovascular Disease Risk Prediction Platform</h1>
                <p>Supervised Machine Learning Diagnostic Screening • CardioTrain Real Cohort (N=70,000)</p>
            </div>
            <div style="display: flex; gap: 0.6rem; flex-wrap: wrap;">
                <span class="badge-chip badge-chip-cyan">
                    <span class="pulse-dot"></span> Production Engine Active
                </span>
                <span class="badge-chip badge-chip-emerald">
                    🔬 Verified Cohort (N=70,000)
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Academic Project Notice
    st.markdown(
        """
        <div class="notice-card">
            <div>
                <strong>Clinical ML Diagnostic Demonstration:</strong> Supervised classification for cardiovascular disease screening based on physiological, hemodynamic, and lifestyle indicators.
                Outputs are calibrated statistical probabilities and do not substitute for licensed clinical consultation.
            </div>
            <span style="font-size: 0.76rem; color: #64748b; font-weight: 600;">CARDIOTRAIN BENCHMARK</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Navigation & Settings
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <span style="font-size: 1.3rem;">🎛️</span>
            <span style="font-weight: 700; font-size: 1.1rem; color: #f8fafc;">System Controls</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_name = metadata.get("best_model_name", "Random Forest")
    best_key = metadata.get("best_model_key", "random_forest")
    best_m = metrics.get(best_key, {}).get("test_metrics", {})
    best_f1 = best_m.get("f1", 0.7235)
    best_acc = best_m.get("accuracy", 0.7328)
    best_roc = best_m.get("roc_auc", 0.7972)

    st.sidebar.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.85); padding: 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); font-size: 0.84rem; margin-bottom: 15px;">
            <div style="color: #94a3b8; font-size: 0.72rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 4px;">Champion Estimator</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #38bdf8; margin-bottom: 8px;">{best_name} ⭐</div>
            <div style="display: flex; justify-content: space-between; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 6px; margin-top: 4px;">
                <span style="color: #94a3b8;">Holdout F1:</span>
                <span style="font-weight: 600; color: #34d399;">{best_f1:.4f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                <span style="color: #94a3b8;">Accuracy:</span>
                <span style="font-weight: 600; color: #cbd5e1;">{best_acc * 100:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                <span style="color: #94a3b8;">ROC-AUC:</span>
                <span style="font-weight: 600; color: #818cf8;">{best_roc:.4f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                <span style="color: #94a3b8;">Cohort Patients:</span>
                <span style="font-weight: 600; color: #cbd5e1;">70,000</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    assessment_mode = st.sidebar.radio(
        "Assessment Mode",
        (
            "🩺 Interactive Patient Form",
            "👥 Cohort Patient Explorer",
            "📁 Batch CSV Upload",
        ),
        index=0,
    )

    # Main Workspace Tabs
    tab_pred, tab_perf, tab_about = st.tabs([
        "🩺 Patient Risk Assessment",
        "📊 Model Benchmarking & Performance",
        "ℹ️ Clinical Methodology & System Architecture",
    ])

    with tab_pred:
        render_prediction_tab(cfg, model, metadata, assessment_mode)

    with tab_perf:
        render_performance_tab(cfg, metrics, metadata)

    with tab_about:
        render_about_tab(cfg, metadata)


# ==============================================================================
# TAB 1: PATIENT RISK ASSESSMENT
# ==============================================================================
def render_prediction_tab(
    cfg: Dict[str, Any], model: Any, metadata: Dict[str, Any], assessment_mode: str
):
    """Patient risk assessment tab with interactive form, demo patients, and batch CSV upload."""

    # --------------------------------------------------------------------------
    # MODE 1: INTERACTIVE PATIENT FORM
    # --------------------------------------------------------------------------
    if assessment_mode == "🩺 Interactive Patient Form":
        st.subheader("1. Enter Patient Physiological & Lifestyle Biomarkers")
        st.caption("Provide current patient clinical measurements to compute real-time cardiovascular risk probability.")

        with st.form("interactive_patient_form"):
            col_demo, col_hemo = st.columns(2)

            with col_demo:
                st.markdown("##### 👤 Demographics & Anthropometrics")
                c1, c2 = st.columns(2)
                with c1:
                    age_years = st.number_input("Age (Years)", min_value=20.0, max_value=95.0, value=52.0, step=1.0)
                with c2:
                    gender_choice = st.selectbox("Biological Sex", ["Female", "Male"], index=0)
                    gender_code = 1 if gender_choice == "Female" else 2

                c3, c4 = st.columns(2)
                with c3:
                    height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=165.0, step=1.0)
                with c4:
                    weight_kg = st.number_input("Weight (kg)", min_value=35.0, max_value=200.0, value=72.0, step=0.5)

                # Real-time BMI indicator preview
                calc_bmi = weight_kg / ((height_cm / 100.0) ** 2)
                bmi_class = (
                    "Underweight (<18.5)" if calc_bmi < 18.5
                    else ("Normal (18.5-24.9)" if calc_bmi < 25.0
                          else ("Overweight (25-29.9)" if calc_bmi < 30.0 else "Obese (≥30)"))
                )
                bmi_color = "#34d399" if 18.5 <= calc_bmi < 25.0 else ("#fbbf24" if calc_bmi < 30 else "#f87171")
                st.markdown(
                    f"""
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 8px 12px; margin-top: 8px;">
                        <span style="font-size: 0.8rem; color: #94a3b8;">Calculated BMI: </span>
                        <strong style="color: #f8fafc; font-size: 0.95rem;">{calc_bmi:.1f} kg/m²</strong>
                        <span style="font-size: 0.78rem; color: {bmi_color}; margin-left: 8px; font-weight: 600;">[{bmi_class}]</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_hemo:
                st.markdown("##### 🩺 Hemodynamics & Blood Pressure")
                c5, c6 = st.columns(2)
                with c5:
                    ap_hi = st.number_input("Systolic Blood Pressure (ap_hi, mmHg)", min_value=70.0, max_value=240.0, value=125.0, step=1.0)
                with c6:
                    ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo, mmHg)", min_value=40.0, max_value=160.0, value=80.0, step=1.0)

                # Real-time Hemodynamic preview
                pulse_press = ap_hi - ap_lo
                map_val = ap_lo + pulse_press / 3.0
                bp_stage = (
                    "Crisis (≥180)" if ap_hi >= 180 or ap_lo >= 120
                    else ("Stage 2 HTN (≥140/90)" if ap_hi >= 140 or ap_lo >= 90
                          else ("Stage 1 HTN (130-139)" if ap_hi >= 130 or ap_lo >= 80
                                else ("Elevated (120-129)" if ap_hi >= 120 and ap_lo < 80 else "Normal (<120/80)")))
                )
                bp_color = "#34d399" if ap_hi < 120 and ap_lo < 80 else ("#fbbf24" if ap_hi < 140 and ap_lo < 90 else "#f87171")

                st.markdown(
                    f"""
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 8px 12px; margin-top: 8px;">
                        <span style="font-size: 0.8rem; color: #94a3b8;">Pulse Pressure: </span>
                        <strong style="color: #f8fafc; font-size: 0.95rem;">{pulse_press:.0f} mmHg</strong> • 
                        <span style="font-size: 0.8rem; color: #94a3b8;">MAP: </span>
                        <strong style="color: #f8fafc; font-size: 0.95rem;">{map_val:.1f} mmHg</strong>
                        <div style="font-size: 0.78rem; color: {bp_color}; margin-top: 4px; font-weight: 600;">AHA Classification: {bp_stage}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            col_meta, col_life = st.columns(2)

            with col_meta:
                st.markdown("##### 🧪 Metabolic Status")
                chol_options = {
                    "1: Normal (< 200 mg/dL)": 1,
                    "2: Above Normal (200 - 239 mg/dL)": 2,
                    "3: Well Above Normal (≥ 240 mg/dL)": 3,
                }
                gluc_options = {
                    "1: Normal (< 100 mg/dL)": 1,
                    "2: Above Normal (100 - 125 mg/dL)": 2,
                    "3: Well Above Normal (≥ 126 mg/dL)": 3,
                }
                c7, c8 = st.columns(2)
                with c7:
                    chol_sel = st.selectbox("Serum Cholesterol Level", list(chol_options.keys()), index=0)
                    chol_code = chol_options[chol_sel]
                with c8:
                    gluc_sel = st.selectbox("Fasting Glucose Level", list(gluc_options.keys()), index=0)
                    gluc_code = gluc_options[gluc_sel]

            with col_life:
                st.markdown("##### 🏃 Lifestyle & Behavioral Factors")
                c9, c10, c11 = st.columns(3)
                with c9:
                    smoke_choice = st.selectbox("Smoking Status", ["Non-smoker (0)", "Smoker (1)"], index=0)
                    smoke_code = 1 if "1" in smoke_choice else 0
                with c10:
                    alco_choice = st.selectbox("Alcohol Intake", ["Non-drinker (0)", "Drinker (1)"], index=0)
                    alco_code = 1 if "1" in alco_choice else 0
                with c11:
                    act_choice = st.selectbox("Physical Activity", ["Active (1)", "Sedentary (0)"], index=0)
                    act_code = 1 if "1" in act_choice else 0

            st.write("")
            submitted = st.form_submit_button("🚀 Compute Cardiovascular Disease Risk", type="primary", use_container_width=True)

        if submitted:
            patient_record = {
                "age": age_years,
                "gender": gender_code,
                "height": height_cm,
                "weight": weight_kg,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": chol_code,
                "gluc": gluc_code,
                "smoke": smoke_code,
                "alco": alco_code,
                "active": act_code,
            }
            patient_df = pd.DataFrame([patient_record])
            with st.spinner("Processing biomarkers and executing inference pipeline..."):
                try:
                    res = predict_patient(patient_df, cfg, model)
                    display_patient_results(res, patient_df, cfg)
                except Exception as e:
                    st.error(f"Inference failed: {e}")

    # --------------------------------------------------------------------------
    # MODE 2: COHORT PATIENT EXPLORER
    # --------------------------------------------------------------------------
    elif assessment_mode == "👥 Cohort Patient Explorer":
        st.subheader("1. Select Real Holdout Test Patient")
        st.caption("Inspect authentic patient profiles from the 14,000-patient holdout validation set and compare predictions to clinical ground truth.")

        demo_path = PROJECT_ROOT / "data" / "processed" / "demo_patients.csv"
        if not demo_path.exists():
            st.warning("Validation cohort file `data/processed/demo_patients.csv` not found. Please train models to generate it.")
            return

        all_demo = pd.read_csv(demo_path)
        patient_ids = all_demo["id"].tolist()

        c_sel, c_info = st.columns([2, 3])
        with c_sel:
            selected_id = st.selectbox("Select Patient Record (ID)", patient_ids)
        with c_info:
            selected_row = all_demo[all_demo["id"] == selected_id].iloc[0]
            truth = int(selected_row["cardio"])
            truth_label = "Presence of Cardiovascular Disease (1)" if truth == 1 else "Absence of CVD / Healthy (0)"
            truth_color = "#f87171" if truth == 1 else "#34d399"
            st.write("")
            st.markdown(
                f"""
                <div style="padding-top: 4px;">
                    <span style="font-size: 0.8rem; color: #94a3b8;">Recorded Clinical Outcome: </span>
                    <strong style="color: {truth_color}; font-size: 0.95rem;">{truth_label}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Patient Profile Strip
        age_in_years = round(selected_row["age"] / 365.25, 1) if selected_row["age"] > 120 else round(selected_row["age"], 1)
        gender_str = "Female" if str(selected_row["gender"]) in ("1", "1.0", "F") else "Male"
        bp_str = f"{int(selected_row['ap_hi'])} / {int(selected_row['ap_lo'])} mmHg"
        bmi_calc = selected_row["weight"] / ((selected_row["height"] / 100.0) ** 2)

        st.markdown(
            f"""
            <div class="patient-profile-card">
                <div class="profile-item">
                    <span class="profile-label">Patient ID</span>
                    <span class="profile-value" style="color: #38bdf8;">#{selected_id}</span>
                </div>
                <div class="profile-item">
                    <span class="profile-label">Age & Sex</span>
                    <span class="profile-value">{age_in_years} yrs • {gender_str}</span>
                </div>
                <div class="profile-item">
                    <span class="profile-label">Height & Weight</span>
                    <span class="profile-value">{selected_row['height']:.0f} cm • {selected_row['weight']:.1f} kg</span>
                </div>
                <div class="profile-item">
                    <span class="profile-label">Blood Pressure</span>
                    <span class="profile-value">{bp_str}</span>
                </div>
                <div class="profile-item">
                    <span class="profile-label">BMI</span>
                    <span class="profile-value">{bmi_calc:.1f} kg/m²</span>
                </div>
                <div class="profile-item">
                    <span class="profile-label">True Outcome</span>
                    <span class="profile-value" style="color: {truth_color};">
                        {'CVD Positive (1)' if truth == 1 else 'CVD Negative (0)'}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        patient_record_df = all_demo[all_demo["id"] == selected_id].copy()

        with st.expander("📋 View Raw Patient Attributes from Cohort Dataset", expanded=False):
            st.dataframe(patient_record_df, use_container_width=True)

        if st.button("🚀 Evaluate Model Prediction Against Clinical Ground Truth", type="primary", use_container_width=True):
            with st.spinner("Executing model pipeline..."):
                try:
                    res = predict_patient(patient_record_df, cfg, model)
                    display_patient_results(res, patient_record_df, cfg, ground_truth=truth)
                except Exception as e:
                    st.error(f"Inference error: {e}")

    # --------------------------------------------------------------------------
    # MODE 3: BATCH CSV SCREENING
    # --------------------------------------------------------------------------
    else:
        st.subheader("1. Batch Patient Screening via CSV Upload")
        st.caption("Upload a CSV file containing multiple patient records to perform batch cardiovascular disease stratification.")

        c_up, c_dl = st.columns([3, 1])

        with c_dl:
            st.write("")
            st.write("")
            template_df = get_upload_template(cfg)
            csv_bytes = template_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV Template",
                data=csv_bytes,
                file_name="cardio_patient_template.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with c_up:
            uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

        if uploaded_file is not None:
            try:
                # Detect separator
                try:
                    batch_df = pd.read_csv(uploaded_file, sep=";")
                    if len(batch_df.columns) < 5:
                        uploaded_file.seek(0)
                        batch_df = pd.read_csv(uploaded_file, sep=",")
                except Exception:
                    uploaded_file.seek(0)
                    batch_df = pd.read_csv(uploaded_file, sep=None, engine="python")

                errs = validate_input(batch_df, cfg)
                if errs:
                    st.error("Uploaded CSV failed schema validation:")
                    for err in errs:
                        st.write(f"- {err}")
                else:
                    st.success(f"Successfully loaded {len(batch_df)} patient records with valid schema!")
                    st.dataframe(batch_df.head(10), use_container_width=True)

                    if st.button("🚀 Run Batch Prediction on Cohort", type="primary", use_container_width=True):
                        with st.spinner("Scoring batch records through champion model pipeline..."):
                            scored_df = predict_from_features(batch_df, model)

                            st.markdown("---")
                            st.subheader("2. Batch Screening Results")

                            # Summary KPI cards
                            n_total = len(scored_df)
                            n_high = (scored_df["prediction"] == 1).sum()
                            pct_high = (n_high / n_total) * 100.0

                            k1, k2, k3 = st.columns(3)
                            k1.metric("Total Patients Screened", f"{n_total:,}")
                            k2.metric("Predicted CVD Positive", f"{n_high:,}", f"{pct_high:.1f}% Prevalence")
                            k3.metric("Predicted Low Risk / Safe", f"{n_total - n_high:,}", f"{100.0 - pct_high:.1f}%")

                            # Download scored results
                            result_csv = scored_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Export Full Scored Dataset (CSV)",
                                data=result_csv,
                                file_name="batch_cardio_predictions.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

                            st.dataframe(scored_df, use_container_width=True)

            except Exception as e:
                st.error(f"Failed to process CSV: {e}")


def display_patient_results(
    res: Dict[str, Any], patient_df: pd.DataFrame, cfg: Dict[str, Any], ground_truth: Optional[int] = None
):
    """Render computed prediction, semi-circular risk gauge, and clinical KPI diagnostic cards."""
    st.markdown("---")
    st.subheader("2. Diagnostic Risk Assessment")

    prob = res["probability"]
    label_text = res["label_text"]
    risk_level = res["risk_level"]
    pred = res["prediction"]

    # Recommendations based on risk level
    if risk_level == "High":
        banner_class = "risk-banner-high"
        tag_class = "tag-high"
        headline = "🚨 Elevated Cardiovascular Disease Risk Detected"
        desc = (
            "The model identifies a high likelihood of cardiovascular pathology based on the multivariable physiological profile. "
            "<strong>Recommended Clinical Actions:</strong> 12-lead ECG, comprehensive lipid panel (LDL, HDL, Triglycerides), "
            "blood pressure monitoring protocol, and therapeutic lifestyle changes or antihypertensive therapy."
        )
    elif risk_level == "Moderate":
        banner_class = "risk-banner-moderate"
        tag_class = "tag-moderate"
        headline = "⚠️ Intermediate Cardiovascular Risk Profile"
        desc = (
            "Patient exhibits borderline cardiovascular risk factors requiring clinical vigilance. "
            "<strong>Recommended Actions:</strong> Sodium restriction, aerobic physical activity (≥150 min/week), dietary consultation, "
            "and repeat blood pressure and lipid screening in 60–90 days."
        )
    else:
        banner_class = "risk-banner-low"
        tag_class = "tag-low"
        headline = "✅ Normal / Low Cardiovascular Risk Profile"
        desc = (
            "Patient biomarkers align with desirable physiological ranges without acute hemodynamic or metabolic risk indicators. "
            "<strong>Recommended Actions:</strong> Maintain current heart-healthy habits, balanced nutrition, and standard annual preventative check-ups."
        )

    # Two-Column Hero: Gauge Meter + Clinical Decision Card
    col_gauge, col_decision = st.columns([1, 1.4])

    with col_gauge:
        gauge_fig = create_risk_gauge(prob, risk_level)
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_decision:
        st.write("")
        truth_badge = ""
        if ground_truth is not None:
            is_correct = (pred == ground_truth)
            verdict_text = "MATCHES CLINICAL TRUTH" if is_correct else "DISCREPANCY WITH GROUND TRUTH"
            verdict_color = "#34d399" if is_correct else "#f87171"
            truth_badge = (
                f'<div style="margin-bottom: 8px;">'
                f'<span style="font-size: 0.76rem; font-weight: 700; background: rgba(255,255,255,0.06); padding: 4px 8px; border-radius: 4px; color: {verdict_color}; border: 1px solid {verdict_color}40;">'
                f'✓ {verdict_text} (True: {ground_truth} | Pred: {pred})'
                f'</span></div>'
            )

        banner_html = (
            f'<div class="risk-banner {banner_class}">'
            f'{truth_badge}'
            f'<span class="risk-banner-tag {tag_class}">{risk_level} Risk Category</span>'
            f'<div class="risk-headline">{headline}</div>'
            f'<p class="risk-body-desc">{desc}</p>'
            f'<div style="margin-top: 12px; display: flex; gap: 14px; font-size: 0.84rem; color: #94a3b8; flex-wrap: wrap;">'
            f'<span>Calculated Probability: <strong style="color: #f8fafc;">{prob * 100:.1f}%</strong></span>'
            f'<span>•</span>'
            f'<span>Classification: <strong style="color: #f8fafc;">{label_text}</strong></span>'
            f'</div></div>'
        )
        st.markdown(banner_html, unsafe_allow_html=True)

    # 3. Clinical Vital KPI Cards
    st.write("")
    st.subheader("3. Clinical Biomarker Breakdown & Categorical Stratification")
    summary = res.get("patient_summary", {})
    demo = summary.get("demographics", {})
    hemo = summary.get("hemodynamics", {})
    meta = summary.get("metabolic", {})
    life = summary.get("lifestyle", {})

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    # Card 1: Hemodynamics
    bp_cat = hemo.get("bp_category", "Normal")
    bp_pill_class = (
        "delta-red" if "Stage 2" in bp_cat or "Crisis" in bp_cat
        else ("delta-neutral" if "Stage 1" in bp_cat or "Elevated" in bp_cat else "delta-green")
    )
    with kpi1:
        st.markdown(
            f"""
            <div class="vital-card">
                <div>
                    <div class="vital-header">
                        <span class="vital-title">🩺 Blood Pressure</span>
                    </div>
                    <div class="vital-value-row">
                        <span class="vital-value">{hemo.get('ap_hi', 120):.0f} / {hemo.get('ap_lo', 80):.0f}</span>
                        <span class="vital-unit">mmHg</span>
                    </div>
                    <div class="vital-delta-pill {bp_pill_class}">
                        <span>AHA Stage: {bp_cat}</span>
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.78rem; color: #cbd5e1; font-weight: 600; margin-bottom: 2px;">
                        Pulse Pressure: {hemo.get('pulse_pressure', 40):.0f} mmHg
                    </div>
                    <div class="vital-ref-tag">
                        Mean Arterial Pressure: {hemo.get('mean_arterial_pressure', 93.3):.1f} mmHg
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Card 2: Anthropometrics & BMI
    bmi_val = demo.get("bmi", 24.0)
    bmi_cat = demo.get("bmi_category", "Normal")
    bmi_pill_class = (
        "delta-green" if bmi_cat == "Normal"
        else ("delta-neutral" if bmi_cat == "Overweight" else "delta-red")
    )
    with kpi2:
        st.markdown(
            f"""
            <div class="vital-card">
                <div>
                    <div class="vital-header">
                        <span class="vital-title">⚖️ Body Mass Index</span>
                    </div>
                    <div class="vital-value-row">
                        <span class="vital-value">{bmi_val:.1f}</span>
                        <span class="vital-unit">kg/m²</span>
                    </div>
                    <div class="vital-delta-pill {bmi_pill_class}">
                        <span>WHO: {bmi_cat}</span>
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.78rem; color: #cbd5e1; font-weight: 600; margin-bottom: 2px;">
                        Height: {demo.get('height', 165):.0f} cm • Weight: {demo.get('weight', 70):.1f} kg
                    </div>
                    <div class="vital-ref-tag">
                        Healthy Target: 18.5 – 24.9 kg/m²
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Card 3: Metabolic Status
    chol_label = meta.get("cholesterol_label", "Normal")
    gluc_label = meta.get("glucose_label", "Normal")
    meta_pill_class = (
        "delta-red" if chol_label == "Well Above Normal" or gluc_label == "Well Above Normal"
        else ("delta-neutral" if chol_label == "Above Normal" or gluc_label == "Above Normal" else "delta-green")
    )
    with kpi3:
        st.markdown(
            f"""
            <div class="vital-card">
                <div>
                    <div class="vital-header">
                        <span class="vital-title">🧪 Metabolic Profile</span>
                    </div>
                    <div class="vital-value-row">
                        <span class="vital-value" style="font-size: 1.15rem;">Chol: {chol_label}</span>
                    </div>
                    <div class="vital-delta-pill {meta_pill_class}">
                        <span>Glucose: {gluc_label}</span>
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.78rem; color: #cbd5e1; font-weight: 600; margin-bottom: 2px;">
                        Atherosclerosis Indicator: {'Elevated' if chol_label != 'Normal' else 'Optimal'}
                    </div>
                    <div class="vital-ref-tag">
                        Clinical Target: Fasting & Total Lipids Level 1
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Card 4: Lifestyle Factors
    smoke_str = life.get("smoking", "No")
    act_str = life.get("physical_activity", "Active")
    is_smoker = "Yes" in smoke_str
    is_inactive = "Sedentary" in act_str
    life_pill_class = "delta-red" if (is_smoker or is_inactive) else "delta-green"
    with kpi4:
        st.markdown(
            f"""
            <div class="vital-card">
                <div>
                    <div class="vital-header">
                        <span class="vital-title">🏃 Behavioral Factors</span>
                    </div>
                    <div class="vital-value-row">
                        <span class="vital-value" style="font-size: 1.15rem;">{act_str}</span>
                    </div>
                    <div class="vital-delta-pill {life_pill_class}">
                        <span>Smoking: {smoke_str}</span>
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.78rem; color: #cbd5e1; font-weight: 600; margin-bottom: 2px;">
                        Alcohol Consumption: {life.get('alcohol', 'No')}
                    </div>
                    <div class="vital-ref-tag">
                        Risk Factor: Smoking + Physical Inactivity
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Comprehensive Clinical Consultation Summary & Patient Care Plan
    render_patient_consultation_summary(res, patient_df)

    # 5. Feature Vector Inspector
    with st.expander("🔍 Inspect Internal Engineered Feature Vector (Sent to Pipeline)"):
        feat_rows = []
        for k, v in res.get("features_used", {}).items():
            if isinstance(v, (float, np.floating)):
                val_str = f"{v:.4f}"
            else:
                val_str = str(v)
            feat_rows.append({"Feature Name": k, "Engineered Value": val_str})
        feat_df = pd.DataFrame(feat_rows)
        st.dataframe(feat_df, use_container_width=True)


def render_patient_consultation_summary(res: Dict[str, Any], patient_df: pd.DataFrame):
    """Render comprehensive clinical consultation summary, primary risk drivers, actionable care plan, and downloadable report."""
    st.write("")
    st.subheader("4. 📋 Patient Clinical Consultation Summary & Care Plan")
    st.caption("Personalized risk analysis, primary physiological drivers, and evidence-based patient management recommendations.")

    summary = res.get("patient_summary", {})
    demo = summary.get("demographics", {})
    hemo = summary.get("hemodynamics", {})
    meta = summary.get("metabolic", {})
    life = summary.get("lifestyle", {})

    prob = res["probability"]
    risk_level = res["risk_level"]
    pred = res["prediction"]

    age = demo.get("age_years", 50.0)
    gender = demo.get("gender", "Female")
    height = demo.get("height", 165.0)
    weight = demo.get("weight", 70.0)
    bmi = demo.get("bmi", 25.0)
    bmi_cat = demo.get("bmi_category", "Normal")

    ap_hi = hemo.get("ap_hi", 120.0)
    ap_lo = hemo.get("ap_lo", 80.0)
    pulse_pressure = hemo.get("pulse_pressure", 40.0)
    mean_ap = hemo.get("mean_arterial_pressure", 93.3)
    bp_cat = hemo.get("bp_category", "Normal")

    chol_lvl = meta.get("cholesterol_level", 1)
    chol_label = meta.get("cholesterol_label", "Normal")
    gluc_lvl = meta.get("glucose_level", 1)
    gluc_label = meta.get("glucose_label", "Normal")

    smoking_str = life.get("smoking", "No")
    alcohol_str = life.get("alcohol", "No")
    active_str = life.get("physical_activity", "Active")

    is_smoker = "Yes" in smoking_str or "Smoker" in smoking_str
    is_drinker = "Yes" in alcohol_str or "Drinks" in alcohol_str
    is_active = "Active" in active_str

    # 1. Identify Clinical Risk Drivers & Protective Factors
    risk_drivers = []
    protective_factors = []

    # Blood pressure
    if ap_hi >= 140 or ap_lo >= 90:
        risk_drivers.append((
            "red",
            f"Stage 2 Hypertension ({ap_hi:.0f}/{ap_lo:.0f} mmHg)",
            "High systemic vascular resistance directly increases myocardial workload, vascular shear stress, and stroke risk."
        ))
    elif ap_hi >= 130 or ap_lo >= 80:
        risk_drivers.append((
            "amber",
            f"Stage 1 Hypertension ({ap_hi:.0f}/{ap_lo:.0f} mmHg)",
            "Elevated blood pressure exceeds optimal vascular target (<120/80 mmHg), accelerating arterial wall remodeling."
        ))
    elif ap_hi >= 120:
        risk_drivers.append((
            "amber",
            f"Elevated Blood Pressure ({ap_hi:.0f}/{ap_lo:.0f} mmHg)",
            "Systolic pressure is above desirable threshold and requires lifestyle modification to avoid hypertension progression."
        ))
    else:
        protective_factors.append(f"Optimal Blood Pressure ({ap_hi:.0f}/{ap_lo:.0f} mmHg)")

    # Arterial stiffness (Pulse pressure)
    if pulse_pressure >= 60:
        risk_drivers.append((
            "red",
            f"Wide Pulse Pressure ({pulse_pressure:.0f} mmHg)",
            "Elevated differential indicates loss of large-artery elastic compliance (arterial stiffness)."
        ))

    # Cholesterol
    if chol_lvl >= 3:
        risk_drivers.append((
            "red",
            "Severe Hypercholesterolemia (Level 3: Well Above Normal)",
            "Critical excess of circulating atherogenic lipoproteins promotes coronary plaque formation and vascular stenosis."
        ))
    elif chol_lvl == 2:
        risk_drivers.append((
            "amber",
            "Elevated Serum Cholesterol (Level 2: Above Normal)",
            "Above-target lipid level increases 10-year coronary atherosclerosis likelihood."
        ))
    else:
        protective_factors.append("Optimal Serum Cholesterol (Level 1: < 200 mg/dL)")

    # Glucose
    if gluc_lvl >= 3:
        risk_drivers.append((
            "red",
            "High Fasting Blood Glucose (Level 3: Well Above Normal)",
            "Hyperglycemia triggers endothelial inflammation, oxidative damage, and heightened cardiovascular mortality."
        ))
    elif gluc_lvl == 2:
        risk_drivers.append((
            "amber",
            "Impaired Fasting Glucose (Level 2: Above Normal)",
            "Prediabetic range indicates developing insulin resistance, compounding cardiovascular risk."
        ))
    else:
        protective_factors.append("Optimal Fasting Blood Glucose (Level 1: < 100 mg/dL)")

    # BMI
    if bmi >= 30.0:
        risk_drivers.append((
            "red",
            f"Clinical Obesity (BMI: {bmi:.1f} kg/m²)",
            "Significant excess adiposity raises cardiac preload, systemic inflammation, and metabolic syndrome risk."
        ))
    elif bmi >= 25.0:
        risk_drivers.append((
            "amber",
            f"Overweight Classification (BMI: {bmi:.1f} kg/m²)",
            "Weight exceeds normal reference window (18.5–24.9 kg/m²), placing additional hemodynamic load on the heart."
        ))
    else:
        protective_factors.append(f"Healthy Body Mass Index ({bmi:.1f} kg/m²)")

    # Smoking
    if is_smoker:
        risk_drivers.append((
            "red",
            "Active Tobacco Smoker",
            "Tobacco exposure impairs nitric oxide release, provokes coronary vasospasm, and elevates platelet thrombogenicity."
        ))
    else:
        protective_factors.append("Non-Smoker (Absence of Tobacco Toxins)")

    # Physical Activity
    if not is_active:
        risk_drivers.append((
            "amber",
            "Physical Inactivity / Sedentary Lifestyle",
            "Sedentary habits correlate with reduced endothelial responsiveness and sub-optimal cardiorespiratory fitness."
        ))
    else:
        protective_factors.append("Physically Active Lifestyle (Cardioprotective)")

    # Alcohol
    if is_drinker:
        risk_drivers.append((
            "amber",
            "Regular Alcohol Intake",
            "Can stimulate sympathetic activity, elevate nocturnal blood pressure, and contribute to excess caloric intake."
        ))
    else:
        protective_factors.append("Non-Drinker")

    # Render Executive Narrative
    summary_bg = "rgba(239, 68, 68, 0.08)" if risk_level == "High" else ("rgba(245, 158, 11, 0.08)" if risk_level == "Moderate" else "rgba(16, 185, 129, 0.08)")
    summary_border = "#ef4444" if risk_level == "High" else ("#f59e0b" if risk_level == "Moderate" else "#10b981")

    top_drivers_text = ", ".join([d[1] for d in risk_drivers[:3]]) if risk_drivers else "No acute risk factors detected"

    narrative_html = (
        f'<div style="background: {summary_bg}; border: 1px solid {summary_border}50; border-left: 4px solid {summary_border}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem;">'
        f'<div style="font-weight: 800; font-size: 1.15rem; color: #ffffff; margin-bottom: 0.4rem;">'
        f'Executive Clinical Synthesis for {gender}, Age {age:.0f}'
        f'</div>'
        f'<p style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.6; margin: 0 0 0.8rem 0;">'
        f'Comprehensive machine learning analysis evaluates this patient at <strong style="color: #ffffff;">{risk_level} Cardiovascular Risk ({prob * 100:.1f}% estimated probability)</strong>. '
        f'The outcome is primarily governed by: <strong style="color: #f8fafc;">{top_drivers_text}</strong>.'
        f'</p>'
        f'<div style="display: flex; gap: 10px; flex-wrap: wrap;">'
    )
    for chip_type, label, _ in risk_drivers:
        chip_cls = "driver-chip-red" if chip_type == "red" else "driver-chip-amber"
        narrative_html += f'<span class="driver-chip {chip_cls}">⚠️ {label}</span>'
    for prot in protective_factors:
        narrative_html += f'<span class="driver-chip driver-chip-green">✓ {prot}</span>'
    narrative_html += '</div></div>'

    st.markdown(narrative_html, unsafe_allow_html=True)

    # Risk Drivers and Protective Factors Columns
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("##### 🚨 Primary Risk Factors (Clinical Drivers)")
        if risk_drivers:
            for chip_type, title, detail in risk_drivers:
                icon = "🔴" if chip_type == "red" else "🟠"
                card_html = (
                    f'<div style="background: rgba(15, 23, 42, 0.55); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">'
                    f'<div style="font-weight: 700; color: #f8fafc; font-size: 0.88rem; margin-bottom: 3px;">'
                    f'{icon} {title}'
                    f'</div>'
                    f'<div style="color: #94a3b8; font-size: 0.81rem; line-height: 1.45;">'
                    f'{detail}'
                    f'</div></div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.success("No elevated risk drivers identified. Patient exhibits healthy baseline parameters.")

    with col_d2:
        st.markdown("##### 🛡️ Protective & Positive Biomarkers")
        if protective_factors:
            for prot in protective_factors:
                prot_html = (
                    f'<div style="background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">'
                    f'<div style="font-weight: 700; color: #34d399; font-size: 0.88rem; margin-bottom: 2px;">'
                    f'✓ {prot}'
                    f'</div>'
                    f'<div style="color: #94a3b8; font-size: 0.81rem; line-height: 1.45;">'
                    f'Within recommended guideline thresholds; supports long-term vascular health.'
                    f'</div></div>'
                )
                st.markdown(prot_html, unsafe_allow_html=True)
        else:
            st.warning("Multiple elevated biomarkers observed across hemodynamic and metabolic categories.")

    # 4 Actionable Patient Care Plan Columns
    st.write("")
    st.markdown("##### 🎯 Evidence-Based Patient Care Plan & Actionable Suggestions")

    plan1, plan2, plan3, plan4 = st.columns(4)

    # 1. Clinical Follow-up
    with plan1:
        urgency_text = (
            "Urgent clinical evaluation within <strong>7–14 days</strong>"
            if risk_level == "High"
            else ("Consultation recommended within <strong>30–45 days</strong>" if risk_level == "Moderate" else "Routine annual cardiovascular wellness visit")
        )
        plan1_html = (
            f'<div class="plan-card">'
            f'<div>'
            f'<div class="plan-header">🩺 1. Medical Management</div>'
            f'<div class="plan-item">• <strong>Timeline:</strong> {urgency_text}.</div>'
            f'<div class="plan-item">• <strong>Electrocardiogram (ECG):</strong> 12-lead ECG to evaluate left ventricular strain, conduction anomalies, or ischemia.</div>'
            f'<div class="plan-item">• <strong>Comprehensive Lipid Panel:</strong> Quantify LDL-C, ApoB, and non-HDL-C to assess atherogenic burden.</div>'
            f'<div class="plan-item">• <strong>Ambulatory BP Protocol:</strong> Log morning and evening BP twice daily for 14 days before therapy adjustment.</div>'
            f'</div></div>'
        )
        st.markdown(plan1_html, unsafe_allow_html=True)

    # 2. Nutrition
    with plan2:
        plan2_html = (
            f'<div class="plan-card">'
            f'<div>'
            f'<div class="plan-header">🥗 2. Nutrition & Diet</div>'
            f'<div class="plan-item">• <strong>DASH / Mediterranean Diet:</strong> High in vegetables, whole grains, extra virgin olive oil, and legumes.</div>'
            f'<div class="plan-item">• <strong>Sodium Restriction:</strong> Cap dietary sodium at <strong>&lt; 2,000 mg/day</strong> (&lt; 1 level tsp salt). Avoid ultra-processed foods.</div>'
            f'<div class="plan-item">• <strong>Lipid Optimization:</strong> Restrict saturated fats to &lt; 7% daily calories; eliminate hydrogenated trans fats.</div>'
            f'<div class="plan-item">• <strong>Viscous Soluble Fiber:</strong> Consume 25–35g/day (oats, barley, psyllium husk) to promote hepatic LDL clearance.</div>'
            f'</div></div>'
        )
        st.markdown(plan2_html, unsafe_allow_html=True)

    # 3. Exercise
    with plan3:
        plan3_html = (
            f'<div class="plan-card">'
            f'<div>'
            f'<div class="plan-header">🏃 3. Exercise Prescription</div>'
            f'<div class="plan-item">• <strong>Moderate Aerobic Target:</strong> Minimum <strong>150–300 minutes/week</strong> (e.g. brisk walking 30–45 min, 5 days/week).</div>'
            f'<div class="plan-item">• <strong>Resistance Training:</strong> 2 non-consecutive sessions/week targeting major muscle groups to boost insulin sensitivity.</div>'
            f'<div class="plan-item">• <strong>Anti-Sedentary Habit:</strong> Break prolonged sitting every 45–60 minutes with 3 minutes of light movement.</div>'
            f'<div class="plan-item">• <strong>Cardiorespiratory Goal:</strong> Gradual progression with target heart rate at 50–70% heart rate reserve.</div>'
            f'</div></div>'
        )
        st.markdown(plan3_html, unsafe_allow_html=True)

    # 4. Lifestyle & Habit
    with plan4:
        smoke_advice = (
            "Enroll in structured smoking cessation; consider nicotine replacement or physician-guided pharmacotherapy (varenicline)."
            if is_smoker else "Maintain tobacco-free status (prevents vascular endothelial injury)."
        )
        plan4_html = (
            f'<div class="plan-card">'
            f'<div>'
            f'<div class="plan-header">🚭 4. Behavioral & Wellness</div>'
            f'<div class="plan-item">• <strong>Tobacco Protocol:</strong> {smoke_advice}</div>'
            f'<div class="plan-item">• <strong>Alcohol Moderation:</strong> Restrict to &le; 1 drink/day for women, &le; 2 for men; abstain if hypertensive.</div>'
            f'<div class="plan-item">• <strong>Restorative Sleep:</strong> Ensure 7–8 hours consistent sleep; screen for obstructive sleep apnea if snoring.</div>'
            f'<div class="plan-item">• <strong>Stress Hygiene:</strong> 10–15 min daily diaphragmatic breathing, mindfulness, or progressive muscle relaxation.</div>'
            f'</div></div>'
        )
        st.markdown(plan4_html, unsafe_allow_html=True)




# ==============================================================================
# TAB 2: MODEL BENCHMARKING & PERFORMANCE
# ==============================================================================
def render_performance_tab(
    cfg: Dict[str, Any], metrics: Dict[str, Any], metadata: Dict[str, Any]
):
    """Render comprehensive model comparison benchmark tab."""
    best_key = metadata.get("best_model_key", "random_forest")
    best_data = metrics.get(best_key, {})
    best_name = metadata.get("best_model_name", "Random Forest")
    test_m = best_data.get("test_metrics", {})

    # Champion Model Spotlight Card
    st.markdown(
        f"""
        <div class="champion-banner">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.6rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.4rem;">🏆</span>
                    <span style="font-size: 1.25rem; font-weight: 800; color: #f8fafc;">Selected Winning Estimator: {best_name}</span>
                </div>
                <span class="badge-chip badge-chip-cyan">Active Production Pipeline</span>
            </div>
            <p style="margin: 0 0 0.8rem 0; color: #cbd5e1; font-size: 0.88rem;">
                Trained and evaluated on 70,000 patient records with 5-fold stratified cross-validation (56,000 training samples) and held-out evaluation on 14,000 independent test patients.
            </p>
            <div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
                <div><span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Holdout F1 Score</span><br><strong style="font-size: 1.25rem; color: #38bdf8;">{test_m.get('f1', 0):.4f}</strong></div>
                <div><span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">ROC-AUC</span><br><strong style="font-size: 1.25rem; color: #818cf8;">{test_m.get('roc_auc', 0):.4f}</strong></div>
                <div><span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Test Accuracy</span><br><strong style="font-size: 1.25rem; color: #34d399;">{test_m.get('accuracy', 0) * 100:.2f}%</strong></div>
                <div><span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Precision</span><br><strong style="font-size: 1.25rem; color: #fbbf24;">{test_m.get('precision', 0):.4f}</strong></div>
                <div><span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Sensitivity / Recall</span><br><strong style="font-size: 1.25rem; color: #f43f5e;">{test_m.get('recall', 0):.4f}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Interactive Bar Chart
    st.subheader("1. Holdout Test Set Comparative Evaluation")
    st.caption("Standardized test performance across 4 classical supervised learning architectures.")
    fig_bar = create_model_comparison_bar_chart(metrics)
    st.plotly_chart(fig_bar, use_container_width=True)

    # 2. Benchmark Metrics Table
    summary_rows = []
    for k, v in metrics.items():
        tm = v.get("test_metrics", {})
        summary_rows.append({
            "Model Architecture": v["display_name"],
            "5-Fold CV Score": v.get("best_cv_score", 0.0),
            "Test Accuracy": f"{tm.get('accuracy', 0) * 100:.2f}%",
            "Precision": f"{tm.get('precision', 0):.4f}",
            "Sensitivity / Recall": f"{tm.get('recall', 0):.4f}",
            "F1-Score (Primary)": f"{tm.get('f1', 0):.4f}",
            "ROC-AUC": f"{tm.get('roc_auc', 0):.4f}",
        })
    df_sum = pd.DataFrame(summary_rows)
    st.dataframe(df_sum, use_container_width=True)

    # 3. Model Diagnostic Plots
    st.write("")
    st.subheader("2. Model Diagnostics: ROC Curves & Feature Importance")
    c_roc, c_imp = st.columns(2)

    roc_path = PROJECT_ROOT / "reports" / "figures" / "roc_curves.png"
    imp_path = PROJECT_ROOT / "reports" / "figures" / "feature_importance.png"

    with c_roc:
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Curves Benchmark on 14,000 Holdout Test Patients", use_container_width=True)
        else:
            st.info("ROC Curve plot not found.")

    with c_imp:
        if imp_path.exists():
            st.image(str(imp_path), caption=f"Gini Feature Importance: {best_name}", use_container_width=True)
        else:
            st.info("Feature importance plot not found.")

    # 4. Confusion Matrices
    st.write("")
    st.subheader("3. Confusion Matrices on Holdout Test Split (N=14,000)")
    cm_cols = st.columns(4)
    model_keys = ["random_forest", "logistic_regression", "decision_tree", "knn"]

    for i, m_key in enumerate(model_keys):
        cm_file = PROJECT_ROOT / "reports" / "figures" / f"cm_{m_key}.png"
        with cm_cols[i]:
            if cm_file.exists():
                st.image(str(cm_file), caption=f"{metrics.get(m_key, {}).get('display_name', m_key)}", use_container_width=True)
            else:
                st.info(f"CM for {m_key} not available.")

    # 5. Hyperparameter Tuning Inspector
    st.write("")
    st.subheader("4. Best Hyperparameter Configurations from Stratified 5-Fold Cross Validation")
    param_cols = st.columns(4)
    for i, m_key in enumerate(model_keys):
        data = metrics.get(m_key, {})
        is_winner = (m_key == best_key)
        with param_cols[i]:
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid {'#38bdf8' if is_winner else 'rgba(255,255,255,0.08)'}; border-radius: 10px; padding: 12px; height: 100%;">
                    <div style="font-weight: 700; color: {'#38bdf8' if is_winner else '#f8fafc'}; font-size: 0.95rem; margin-bottom: 6px;">
                        {data.get('display_name', m_key)} {'⭐' if is_winner else ''}
                    </div>
                    <div style="font-size: 0.76rem; color: #94a3b8; margin-bottom: 6px;">
                        5-Fold CV Score: <strong style="color: #cbd5e1;">{data.get('best_cv_score', 'N/A')}</strong>
                    </div>
                    <div style="font-size: 0.74rem; color: #cbd5e1; background: rgba(0,0,0,0.3); padding: 8px; border-radius: 6px; font-family: monospace;">
                        {str(data.get('best_params', {}))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# TAB 3: METHODOLOGY & SYSTEM ARCHITECTURE
# ==============================================================================
def render_about_tab(cfg: Dict[str, Any], metadata: Dict[str, Any]):
    """Documentation, Clinical Architecture, and System Methodology."""
    st.subheader("System Architecture & Clinical Rationale")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="viva-card">
                <h4 style="margin: 0 0 8px 0; color: #38bdf8;">📌 Clinical Problem & Motivation</h4>
                <ul style="margin: 0; padding-left: 20px; color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                    <li><strong>Global Burden:</strong> Cardiovascular disease (CVD) is the leading cause of global mortality. Early stratification enables preventative lifestyle and pharmacological interventions before acute myocardial infarction or stroke.</li>
                    <li><strong>Non-Linear Synergies:</strong> A single elevated biomarker (e.g. Systolic BP = 135 mmHg) may appear moderate in isolation, but combined with elevated cholesterol, smoking, and age > 50, it confers multiplicative risk captured by machine learning.</li>
                </ul>
            </div>

            <div class="viva-card">
                <h4 style="margin: 0 0 8px 0; color: #38bdf8;">🛡️ Zero Data Leakage Architecture</h4>
                <ul style="margin: 0; padding-left: 20px; color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                    <li><strong>Stratified Holdout Splitting:</strong> The 70,000 patient dataset is strictly split into 80% training (56,000) and 20% testing (14,000) with preserved class stratification.</li>
                    <li><strong>Atomic Scikit-Learn Pipelines:</strong> Imputers (<code>SimpleImputer(strategy='median')</code>) and scalers (<code>StandardScaler</code>) are fitted exclusively on training folds, eliminating data snooping into the holdout distribution.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        best_name = metadata.get("best_model_name", "Random Forest")
        st.markdown(
            f"""
            <div class="viva-card">
                <h4 style="margin: 0 0 8px 0; color: #34d399;">⚙️ Pipeline Specifications</h4>
                <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.7;">
                    <div>• <strong>Dataset Source:</strong> <code>Cardiovascular Disease (cardio_train)</code></div>
                    <div>• <strong>Cohort Size:</strong> <strong>70,000 real patient examinations</strong></div>
                    <div>• <strong>Target Variable:</strong> <code>cardio</code> (0 = Healthy, 1 = Presence of CVD)</div>
                    <div>• <strong>Primary Selection Metric:</strong> <code>F1-Score</code> & <code>ROC-AUC</code></div>
                    <div>• <strong>Champion Estimator:</strong> <strong style="color: #38bdf8;">{best_name}</strong></div>
                    <div>• <strong>Engineered Feature Count:</strong> <strong>18 features</strong> (hemodynamics, BMI, categories, interactions)</div>
                    <div>• <strong>Missing Data Strategy:</strong> Median numeric & most-frequent categorical imputation</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            r"""
            <div class="viva-card">
                <h4 style="margin: 0 0 8px 0; color: #34d399;">📐 Cardiovascular Domain Engineering</h4>
                <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                    • <strong>Pulse Pressure:</strong> &Delta;P = SBP - DBP (independent predictor of arterial stiffness)
                    <br>• <strong>Mean Arterial Pressure (MAP):</strong> MAP = DBP + ⅓(SBP - DBP)
                    <br>• <strong>Body Mass Index:</strong> BMI = Weight<sub>kg</sub> / (Height<sub>m</sub>)<sup>2</sup>
                    <br>• <strong>AHA Blood Pressure Stages:</strong> Normal, Elevated, Stage 1 HTN, Stage 2 HTN, Crisis
                    <br>• <strong>Interaction:</strong> Age &times; (SBP / 100) capturing age-accelerated vascular pressure
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
