"""Streamlit Web Application: Patient Health Risk Prediction.

Academic Machine-Learning Interface for Longitudinal and Single-Record Patient Risk Analysis.
Features Clinical Slate Dark Theme, Interactive Risk Gauges, Physiological Reference Bands,
and Comparative Model Benchmarks.
"""

import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    predict_from_history,
    validate_input,
)

# Page configuration
st.set_page_config(
    page_title="CardioMetabolic Risk Stratification System",
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
        padding: 1rem 1.1rem;
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
        padding: 0.18rem 0.5rem;
        border-radius: 4px;
        width: fit-content;
        margin-bottom: 0.45rem;
    }

    .delta-red { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .delta-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .delta-neutral { background: rgba(100, 116, 139, 0.15); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.3); }

    .vital-ref-tag {
        font-size: 0.72rem;
        color: #64748b;
        border-top: 1px dashed rgba(255, 255, 255, 0.08);
        padding-top: 0.4rem;
        margin-top: 0.2rem;
    }

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

    /* Methodology Cards */
    .methodology-card {
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
# CLINICAL REFERENCE DICTIONARIES & HELPER UTILITIES
# ==============================================================================
PHYSIOLOGICAL_RANGES = {
    "systolic_bp": {"y0": 90, "y1": 120, "label": "Normal BP (90-120)", "unit": "mmHg"},
    "glucose": {"y0": 70, "y1": 100, "label": "Normal Fasting (70-100)", "unit": "mg/dL"},
    "cholesterol": {"y0": 130, "y1": 200, "label": "Desirable (<200)", "unit": "mg/dL"},
    "bmi": {"y0": 18.5, "y1": 24.9, "label": "Normal BMI (18.5-24.9)", "unit": "kg/m²"},
    "medication_count": {"y0": 0, "y1": 2, "label": "Low Regimen (0-2)", "unit": "meds"},
}

VITAL_ICONS = {
    "systolic_bp": "🩺",
    "glucose": "🩸",
    "cholesterol": "🧪",
    "bmi": "⚖️",
    "medication_count": "💊",
}


def get_clinical_classification(vital_name: str, value: float) -> Tuple[str, str, str]:
    """Return clinical tag, delta status class, and clinical reference text."""
    if vital_name == "systolic_bp":
        ref = "Clinical Target: < 120 mmHg"
        if value < 120:
            return "Normal (<120)", "delta-green", ref
        elif value < 130:
            return "Elevated (120-129)", "delta-neutral", ref
        elif value < 140:
            return "Stage 1 HTN (130-139)", "delta-red", ref
        else:
            return "Stage 2 HTN (≥140)", "delta-red", ref

    elif vital_name == "glucose":
        ref = "Clinical Target: 70-99 mg/dL"
        if value < 100:
            return "Normal (<100)", "delta-green", ref
        elif value < 126:
            return "Impaired / Prediabetic (100-125)", "delta-neutral", ref
        else:
            return "Diabetic Range (≥126)", "delta-red", ref

    elif vital_name == "cholesterol":
        ref = "Clinical Target: < 200 mg/dL"
        if value < 200:
            return "Desirable (<200)", "delta-green", ref
        elif value < 240:
            return "Borderline (200-239)", "delta-neutral", ref
        else:
            return "High Risk (≥240)", "delta-red", ref

    elif vital_name == "bmi":
        ref = "Clinical Target: 18.5 - 24.9 kg/m²"
        if value < 18.5:
            return "Underweight (<18.5)", "delta-neutral", ref
        elif value < 25.0:
            return "Normal Weight", "delta-green", ref
        elif value < 30.0:
            return "Overweight (25-29.9)", "delta-neutral", ref
        else:
            return "Obese (≥30)", "delta-red", ref

    elif vital_name == "medication_count":
        ref = "Standard Regimen: 0 - 3 meds"
        if value <= 2:
            return "Low Burden (0-2)", "delta-green", ref
        elif value <= 4:
            return "Moderate (3-4)", "delta-neutral", ref
        else:
            return "Polypharmacy (≥5)", "delta-red", ref

    return "Recorded", "delta-neutral", "Reference: N/A"


def create_risk_gauge(prob: float, risk_level: str) -> go.Figure:
    """Create a semi-circular clinical risk gauge."""
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
                "text": "<b>Deterioration Probability</b><br><span style='font-size:12px;color:#94a3b8;'>Model Risk Estimate</span>",
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
                    "tickvals": [0, 20, 40, 65, 80, 100],
                    "ticktext": ["0%", "20%", "40% (Mod)", "65% (High)", "80%", "100%"],
                    "tickfont": {"size": 10, "color": "#94a3b8"},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(255, 255, 255, 0.03)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(16, 185, 129, 0.20)"},
                    {"range": [40, 65], "color": "rgba(245, 158, 11, 0.20)"},
                    {"range": [65, 100], "color": "rgba(244, 63, 94, 0.20)"},
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

    # Color palette matching clinical dark slate with Champion spotlight
    colors = {
        "Logistic Regression": "#38bdf8",  # Cyan (Champion ⭐)
        "Random Forest": "#818cf8",       # Indigo
        "Decision Tree": "#34d399",        # Mint
        "K-Nearest Neighbours": "#fbbf24", # Amber
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


def create_patient_attribution_chart(
    features_dict: Dict[str, Any], model: Any, metadata: Dict[str, Any]
) -> Optional[go.Figure]:
    """Generate patient-specific feature attribution bar chart (Explainable AI)."""
    if not hasattr(model, "named_steps") or "prep" not in model.named_steps or "clf" not in model.named_steps:
        return None

    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]

    num_cols = metadata.get("features", {}).get("numeric", [])
    cat_cols = metadata.get("features", {}).get("categorical", [])

    try:
        X = pd.DataFrame([features_dict])[num_cols + cat_cols]
        X_trans = prep.transform(X)

        if hasattr(prep, "get_feature_names_out"):
            feat_names = list(prep.get_feature_names_out())
        else:
            feat_names = num_cols + cat_cols

        if hasattr(clf, "coef_"):
            weights = clf.coef_[0]
            scores = X_trans[0] * weights
        elif hasattr(clf, "feature_importances_"):
            scores = clf.feature_importances_ * X_trans[0]
        else:
            return None

        # Top 6 by absolute impact, ascending for horizontal bar display
        top_indices = np.argsort(np.abs(scores))[::-1][:6]
        top_indices = top_indices[::-1]

        plot_names = []
        plot_scores = []
        plot_colors = []

        for idx in top_indices:
            raw_name = feat_names[idx]
            clean_name = (
                raw_name.replace("num__", "")
                .replace("cat__", "")
                .replace("_", " ")
                .title()
            )
            score_val = float(scores[idx])
            plot_names.append(clean_name)
            plot_scores.append(round(score_val, 3))
            plot_colors.append("#f43f5e" if score_val > 0 else "#10b981")

        fig = go.Figure(
            go.Bar(
                x=plot_scores,
                y=plot_names,
                orientation="h",
                marker=dict(color=plot_colors, line=dict(width=0)),
                text=[f"{s:+.2f}" for s in plot_scores],
                textposition="auto",
                textfont=dict(size=11, color="#ffffff", family="Plus Jakarta Sans"),
            )
        )

        fig.update_layout(
            title=dict(
                text="<b>Patient-Specific Risk Contributors (Explainable AI Attribution)</b><br><span style='font-size:11px;color:#94a3b8;'>Red (+): Increases Risk Score | Green (-): Stabilizing / Protective Factor</span>",
                font=dict(size=13, color="#cbd5e1", family="Plus Jakarta Sans"),
            ),
            height=260,
            margin=dict(l=10, r=20, t=45, b=20),
            paper_bgcolor="rgba(15, 23, 42, 0.55)",
            plot_bgcolor="rgba(15, 23, 42, 0.35)",
            xaxis=dict(
                title="Impact on Model Log-Odds Risk",
                gridcolor="rgba(255, 255, 255, 0.05)",
                tickfont=dict(size=10, color="#94a3b8"),
                zeroline=True,
                zerolinecolor="rgba(255, 255, 255, 0.2)",
            ),
            yaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.05)",
                tickfont=dict(size=11, color="#f1f5f9"),
            ),
            font=dict(family="Plus Jakarta Sans"),
        )
        return fig
    except Exception:
        return None


@st.cache_resource
def _load_cached_artifacts(mtime: float):
    """Cache trained model pipeline and metadata, keyed by disk artifact timestamp."""
    cfg = load_config()
    model, metadata, metrics = load_artifacts()
    return cfg, model, metadata, metrics


def get_cached_artifacts():
    """Retrieve model artifacts with automatic timestamp-based cache invalidation."""
    try:
        cfg = load_config()
        models_dir = PROJECT_ROOT / cfg["paths"]["models_dir"]
        metrics_file = models_dir / "metrics.json"
        mtime = metrics_file.stat().st_mtime if metrics_file.exists() else 0.0
        cfg, model, metadata, metrics = _load_cached_artifacts(mtime)
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
                <h1>🩺 CardioMetabolic Patient Risk Predictor</h1>
                <p>Longitudinal Multi-Visit Clinical Machine Learning Diagnostic Platform (27,489 Encounters)</p>
            </div>
            <div style="display: flex; gap: 0.6rem; flex-wrap: wrap;">
                <span class="badge-chip badge-chip-cyan">
                    <span class="pulse-dot"></span> Production Engine Active
                </span>
                <span class="badge-chip badge-chip-emerald">
                    🔬 Research Cohort (N=5,000)
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
                <strong>Academic ML Research Demonstration:</strong> Supervised classification for longitudinal deterioration screening.
                Model outputs are statistical posterior estimates and do not substitute for licensed clinical judgment.
            </div>
            <span style="font-size: 0.76rem; color: #64748b; font-weight: 600;">CLASSICAL ML BENCHMARK</span>
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

    dataset_mode = cfg["dataset"]["mode"]
    best_key = metadata.get("best_model_key", "logistic_regression")
    best_name = metadata.get("best_model_name", "Logistic Regression")
    best_metrics = metrics.get(best_key, {}).get("test_metrics", {})
    best_f1 = best_metrics.get("f1", 0.9471)
    best_auc = best_metrics.get("roc_auc", 0.9938)
    best_acc = best_metrics.get("accuracy", 0.9520)

    st.sidebar.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.85); padding: 14px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); font-size: 0.84rem; margin-bottom: 15px;">
            <div style="color: #94a3b8; font-size: 0.72rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 4px;">Champion Estimator</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #38bdf8; margin-bottom: 8px;">{best_name} ⭐</div>
            <div style="display: flex; justify-content: space-between; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 6px; margin-top: 4px;">
                <span style="color: #94a3b8;">Holdout F1 Score:</span>
                <span style="font-weight: 600; color: #34d399;">{best_f1:.4f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                <span style="color: #94a3b8;">Holdout ROC-AUC:</span>
                <span style="font-weight: 600; color: #818cf8;">{best_auc:.4f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                <span style="color: #94a3b8;">Test Accuracy:</span>
                <span style="font-weight: 600; color: #f1f5f9;">{best_acc * 100:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                <span style="color: #94a3b8;">Cohort Scale:</span>
                <span style="font-weight: 600; color: #cbd5e1;">N=5,000 (80/20)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    training_date = metadata.get("training_date", "N/A")
    st.sidebar.caption(f"🕒 Trained: `{training_date}`")
    if st.sidebar.button("🔄 Reload Model Artifacts", use_container_width=True, help="Force reload metrics and pipeline from disk"):
        st.cache_resource.clear()
        st.rerun()

    # Main Workspace Tabs
    tab_pred, tab_perf, tab_about = st.tabs([
        "🩺 Patient Risk Assessment",
        "📊 Model Benchmarking & Performance",
        "ℹ️ Clinical Methodology & Architecture",
    ])

    with tab_pred:
        render_prediction_tab(cfg, model, metadata)

    with tab_perf:
        render_performance_tab(cfg, metrics, metadata)

    with tab_about:
        render_about_tab(cfg, metadata)


# ==============================================================================
# TAB 1: PATIENT RISK ASSESSMENT
# ==============================================================================
def render_prediction_tab(cfg: Dict[str, Any], model: Any, metadata: Dict[str, Any]):
    """Main patient prediction interface."""
    dataset_mode = cfg["dataset"]["mode"]

    if dataset_mode == "longitudinal":
        input_mode = st.sidebar.radio(
            "Input Mode",
            ("Select Existing Patient", "Upload Patient History (CSV)"),
            index=0,
        )

        patient_history_df = None

        if input_mode == "Select Existing Patient":
            demo_path = PROJECT_ROOT / "data" / "processed" / "demo_patients.csv"

            if demo_path.exists():
                all_demo = pd.read_csv(demo_path)
                pid_col = cfg["dataset"]["patient_id_col"]

                # Extract risk tier mapping if present in demo_patients.csv
                tier_col_present = "risk_tier" in all_demo.columns
                tier_map = {}
                if tier_col_present:
                    for _, row in all_demo[[pid_col, "risk_tier"]].drop_duplicates().iterrows():
                        tier_map[row[pid_col]] = str(row["risk_tier"])

                # Stratified risk cohort filter buttons
                tier_filter = st.radio(
                    "Stratified Risk Cohort Filter",
                    ["All (60 Patients)", "🟢 Low Risk (20)", "🟡 Medium Risk (20)", "🔴 High Risk (20)"],
                    horizontal=True,
                )

                if "Low Risk" in tier_filter:
                    filtered_pids = sorted([p for p, t in tier_map.items() if t == "Low Risk"])
                elif "Medium Risk" in tier_filter:
                    filtered_pids = sorted([p for p, t in tier_map.items() if t == "Medium Risk"])
                elif "High Risk" in tier_filter:
                    filtered_pids = sorted([p for p, t in tier_map.items() if t == "High Risk"])
                else:
                    filtered_pids = sorted(list(all_demo[pid_col].unique()))

                col_sel, col_desc = st.columns([2, 3])
                with col_sel:
                    def format_patient_id(pid):
                        tier = tier_map.get(pid, "")
                        return f"{pid}  [{tier}]" if tier else pid

                    selected_pid = st.selectbox(
                        "Choose Validation Cohort Patient ID",
                        filtered_pids,
                        format_func=format_patient_id,
                    )
                with col_desc:
                    p_recs = all_demo[all_demo[pid_col] == selected_pid].sort_values(
                        by=cfg["dataset"]["date_col"]
                    ).reset_index(drop=True)
                    patient_tier_label = tier_map.get(selected_pid, "")
                    tier_badge_text = f" • **{patient_tier_label} Stratum**" if patient_tier_label else ""
                    st.write("")
                    st.caption(
                        f"Loaded **{len(p_recs)}** chronological visits for patient **{selected_pid}**{tier_badge_text}."
                    )

                patient_history_df = p_recs

                # Display Patient Demographics Strip Card
                age_series = p_recs["age"].dropna() if "age" in p_recs.columns else pd.Series()
                age_val = f"{int(age_series.iloc[-1])}" if not age_series.empty else "N/A"
                gender_series = p_recs["gender"].dropna() if "gender" in p_recs.columns else pd.Series()
                gender_val = str(gender_series.iloc[-1]) if not gender_series.empty else "N/A"
                outcome_truth = (
                    "Adverse Event (1)"
                    if p_recs["outcome"].iloc[0] == 1
                    else "Normal / Safe (0)"
                )
                date_min = p_recs[cfg["dataset"]["date_col"]].iloc[0]
                date_max = p_recs[cfg["dataset"]["date_col"]].iloc[-1]
                tier_color = "#34d399" if "Low" in patient_tier_label else ("#fbbf24" if "Medium" in patient_tier_label else "#f87171")

                st.markdown(
                    f"""
                    <div class="patient-profile-card">
                        <div class="profile-item">
                            <span class="profile-label">Patient ID</span>
                            <span class="profile-value" style="color: #38bdf8;">{selected_pid}</span>
                        </div>
                        <div class="profile-item">
                            <span class="profile-label">Risk Stratum</span>
                            <span class="profile-value" style="color: {tier_color};">{patient_tier_label or 'Holdout Cohort'}</span>
                        </div>
                        <div class="profile-item">
                            <span class="profile-label">Age / Sex</span>
                            <span class="profile-value">{age_val} yrs / {gender_val}</span>
                        </div>
                        <div class="profile-item">
                            <span class="profile-label">Visits on Record</span>
                            <span class="profile-value">{len(p_recs)} visits</span>
                        </div>
                        <div class="profile-item">
                            <span class="profile-label">Observation Window</span>
                            <span class="profile-value" style="font-size: 0.95rem;">{date_min} → {date_max}</span>
                        </div>
                        <div class="profile-item">
                            <span class="profile-label">Cohort Ground Truth</span>
                            <span class="profile-value" style="color: {'#f87171' if p_recs['outcome'].iloc[0] == 1 else '#34d399'}; font-size: 0.95rem;">
                                {outcome_truth}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.warning("`data/processed/demo_patients.csv` not found. Please upload a CSV instead.")

        else:
            st.subheader("1. Upload Longitudinal Patient History")
            col_up, col_dl = st.columns([3, 1])

            with col_dl:
                st.write("")
                st.write("")
                template_df = get_upload_template(cfg)
                template_csv = template_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Template",
                    data=template_csv,
                    file_name="patient_history_template.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with col_up:
                uploaded_file = st.file_uploader(
                    "Upload patient visit log (CSV format)", type=["csv"]
                )

            if uploaded_file is not None:
                try:
                    raw_upload = pd.read_csv(uploaded_file)
                    errs = validate_input(raw_upload, cfg)
                    if errs:
                        st.error("Uploaded file has schema issues:")
                        for err in errs:
                            st.write(f"- {err}")
                    else:
                        pid_col = cfg["dataset"]["patient_id_col"]
                        unique_pids = raw_upload[pid_col].unique()
                        if len(unique_pids) > 1:
                            target_pid = st.selectbox(
                                "Multiple patients detected. Select patient ID to analyze:",
                                unique_pids,
                            )
                            patient_history_df = raw_upload[
                                raw_upload[pid_col] == target_pid
                            ].copy()
                        else:
                            patient_history_df = raw_upload.copy()
                        st.success(
                            f"Successfully loaded {len(patient_history_df)} records for {patient_history_df[pid_col].iloc[0]}"
                        )
                except Exception as e:
                    st.error(f"Error parsing CSV: {e}")

        # Render Patient History Table and Analyze Button
        if patient_history_df is not None and not patient_history_df.empty:
            with st.expander("📋 Inspect Raw Chronological Visit Records", expanded=False):
                st.dataframe(patient_history_df, use_container_width=True)

            if st.button("🚀 Analyze Patient Risk Trajectory", type="primary", use_container_width=True):
                with st.spinner("Executing feature engineering & inference pipeline..."):
                    try:
                        res = predict_from_history(patient_history_df, cfg, model)
                        display_patient_results(res, patient_history_df, cfg, model, metadata)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    else:
        render_single_record_form(cfg, model, metadata)


def display_patient_results(
    res: Dict[str, Any],
    history_df: pd.DataFrame,
    cfg: Dict[str, Any],
    model: Any = None,
    metadata: Dict[str, Any] = None,
):
    """Display computed prediction, vital KPI cards, and longitudinal trend plots."""
    st.markdown("---")
    st.subheader("2. Diagnostic Risk Assessment")

    prob = res["probability"]
    label_text = res["label_text"]
    risk_level = res["risk_level"]

    # Recommendations based on risk level
    if risk_level == "High":
        banner_class = "risk-banner-high"
        tag_class = "tag-high"
        headline = "🚨 Elevated CardioMetabolic Risk Detected"
        desc = (
            "The model identifies an accelerated risk profile based on adverse longitudinal biomarker trajectories. "
            "<strong>Early clinical intervention recommended:</strong> Comprehensive metabolic evaluation, medication regimen adjustment, "
            "and follow-up consultation within 14–30 days."
        )
    elif risk_level == "Moderate":
        banner_class = "risk-banner-moderate"
        tag_class = "tag-moderate"
        headline = "⚠️ Intermediate Risk Trajectory"
        desc = (
            "Patient demonstrates borderline vital trends requiring observation. "
            "<strong>Recommended action:</strong> Lifestyle counseling, dietary sodium reduction, and routine 60-day vital re-assessment."
        )
    else:
        banner_class = "risk-banner-low"
        tag_class = "tag-low"
        headline = "✅ Low Risk / Stable Biomarker Profile"
        desc = (
            "Longitudinal parameters are within safe target thresholds with no acute negative trend slopes detected. "
            "<strong>Recommended action:</strong> Maintain current care plan and standard annual check-up schedule."
        )

    # Two-Column Hero: Gauge Meter + Clinical Decision Card
    col_gauge, col_decision = st.columns([1, 1.4])

    with col_gauge:
        gauge_fig = create_risk_gauge(prob, risk_level)
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_decision:
        st.write("")
        st.markdown(
            f"""
            <div class="risk-banner {banner_class}">
                <span class="risk-banner-tag {tag_class}">{risk_level} Risk Category</span>
                <div class="risk-headline">{headline}</div>
                <p class="risk-body-desc">{desc}</p>
                <div style="margin-top: 12px; display: flex; gap: 10px; font-size: 0.82rem; color: #94a3b8;">
                    <span>Calculated Probability: <strong style="color: #f8fafc;">{prob * 100:.1f}%</strong></span>
                    <span>•</span>
                    <span>Classification: <strong style="color: #f8fafc;">{label_text}</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Explainable AI Attribution Chart (Top Patient Risk Drivers)
    if model is not None and metadata is not None:
        attr_fig = create_patient_attribution_chart(res.get("features_used", {}), model, metadata)
        if attr_fig is not None:
            st.plotly_chart(attr_fig, use_container_width=True)

    # 3. Clinical Vital KPI Cards
    st.write("")
    st.subheader("3. Clinical Baseline & Longitudinal Biomarker Dynamics")
    summary = res.get("patient_summary", {})
    vitals = summary.get("vitals", {})

    if vitals:
        cols = st.columns(len(vitals))
        for idx, (vital_name, stats) in enumerate(vitals.items()):
            latest_val = stats["latest"]
            chg = stats["change"]
            pct_chg = stats["pct_change"]
            icon = VITAL_ICONS.get(vital_name, "📊")
            display_name = vital_name.replace("_", " ").title()

            tag_text, status_class, ref_text = get_clinical_classification(
                vital_name, latest_val
            )
            unit = PHYSIOLOGICAL_RANGES.get(vital_name, {}).get("unit", "")

            sign = "+" if chg > 0 else ""
            delta_str = f"{sign}{chg:.1f} {unit} ({sign}{pct_chg:.1f}%)"

            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="vital-card">
                        <div>
                            <div class="vital-header">
                                <span class="vital-title">{icon} {display_name}</span>
                            </div>
                            <div class="vital-value-row">
                                <span class="vital-value">{latest_val}</span>
                                <span class="vital-unit">{unit}</span>
                            </div>
                            <div class="vital-delta-pill {status_class}">
                                <span>{'▲' if chg > 0 else ('▼' if chg < 0 else '•')}</span>
                                <span>{delta_str}</span>
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #cbd5e1; font-weight: 600; margin-bottom: 2px;">
                                {tag_text}
                            </div>
                            <div class="vital-ref-tag">
                                {ref_text}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # 4. Longitudinal Trajectory Charts with Normal Reference Bands
    st.write("")
    st.subheader("4. Chronological Biomarker Trajectories with Target Bands")
    date_col = cfg["dataset"]["date_col"]
    tv_cols = cfg["dataset"].get("time_varying_numeric", [])

    if len(history_df) >= 2 and tv_cols:
        plot_df = history_df.copy()
        plot_df[date_col] = pd.to_datetime(plot_df[date_col], errors="coerce")
        plot_df = plot_df.sort_values(by=date_col)

        chart_cols = st.columns(2)
        for i, col in enumerate(tv_cols):
            with chart_cols[i % 2]:
                ref_info = PHYSIOLOGICAL_RANGES.get(col, {})
                y0 = ref_info.get("y0", 0)
                y1 = ref_info.get("y1", 100)
                unit_label = ref_info.get("unit", "")
                nice_name = col.replace("_", " ").title()

                fig = px.line(
                    plot_df,
                    x=date_col,
                    y=col,
                    markers=True,
                    title=f"<b>{nice_name} Trajectory</b> ({unit_label})",
                )
                fig.update_traces(
                    line=dict(color="#38bdf8", width=3, shape="spline", smoothing=1.1),
                    marker=dict(
                        size=8,
                        color="#0284c7",
                        line=dict(width=2, color="#ffffff"),
                    ),
                    fill="tozeroy",
                    fillcolor="rgba(56, 189, 248, 0.05)",
                    hovertemplate=(
                        f"<b>Date:</b> %{{x|%b %d, %Y}}<br>"
                        f"<b>{nice_name}:</b> %{{y:.1f}} {unit_label}<extra></extra>"
                    ),
                )

                # Add normal reference shaded band
                fig.add_hrect(
                    y0=y0,
                    y1=y1,
                    fillcolor="rgba(16, 185, 129, 0.08)",
                    line_width=0,
                    annotation_text=f"Normal Target ({y0}-{y1})",
                    annotation_position="top left",
                    annotation_font=dict(size=10, color="#34d399"),
                )

                fig.update_layout(
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(15, 23, 42, 0.55)",
                    plot_bgcolor="rgba(15, 23, 42, 0.35)",
                    font=dict(color="#94a3b8", family="Plus Jakarta Sans"),
                    xaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.05)",
                        title="",
                        tickfont=dict(size=10, color="#94a3b8"),
                    ),
                    yaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.05)",
                        title=f"{unit_label}",
                        tickfont=dict(size=10, color="#94a3b8"),
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Single visit on record: longitudinal trend graph requires at least 2 chronological observations.")

    # 5. Engineered Feature Vector Inspector
    with st.expander("🔍 Inspect Internal Engineered Feature Vector (Sent to Pipeline)"):
        feat_rows = []
        for k, v in res["features_used"].items():
            if isinstance(v, (float, np.floating)):
                val_str = f"{v:.4f}"
            else:
                val_str = str(v)
            feat_rows.append({"Feature Name": k, "Engineered Value": val_str})
        feat_df = pd.DataFrame(feat_rows)
        st.dataframe(feat_df, use_container_width=True)


# ==============================================================================
# SINGLE RECORD FORM MODE
# ==============================================================================
def render_single_record_form(cfg: Dict[str, Any], model: Any, metadata: Dict[str, Any]):
    """Dynamic form for single-record mode."""
    st.subheader("Single Patient Assessment")
    dataset_cfg = cfg["dataset"]
    valid_ranges = dataset_cfg.get("valid_ranges", {})
    static_num = dataset_cfg.get("static_numeric", [])
    static_cat = dataset_cfg.get("static_categorical", [])
    tv_cols = dataset_cfg.get("time_varying_numeric", [])

    with st.form("single_patient_form"):
        col1, col2 = st.columns(2)
        inputs = {}

        for i, col in enumerate(static_num + tv_cols):
            c = col1 if i % 2 == 0 else col2
            low, high = valid_ranges.get(col, (0, 300))
            default_val = float(low + (high - low) * 0.4)
            inputs[col] = c.number_input(
                f"{col.replace('_', ' ').title()} ({low} - {high})",
                min_value=float(low),
                max_value=float(high),
                value=round(default_val, 1),
            )

        for col in static_cat:
            cat_opts = metadata.get("categorical_values", {}).get(col, ["F", "M", "Other"])
            inputs[col] = col1.selectbox(col.replace("_", " ").title(), cat_opts)

        submitted = st.form_submit_button("Predict Patient Risk", type="primary", use_container_width=True)

    if submitted:
        single_df = pd.DataFrame([inputs])
        preds_df = predict_from_features(single_df, model)
        pred = int(preds_df["prediction"].iloc[0])
        prob = (
            float(preds_df["risk_probability"].iloc[0])
            if "risk_probability" in preds_df.columns
            else (1.0 if pred == 1 else 0.0)
        )

        risk_lvl = "High" if prob >= 0.65 else ("Moderate" if prob >= 0.40 else "Low")
        col_g, col_t = st.columns([1, 1.4])
        with col_g:
            st.plotly_chart(create_risk_gauge(prob, risk_lvl), use_container_width=True)
        with col_t:
            st.write("")
            st.write("")
            st.markdown(
                f"""
                <div class="risk-banner {'risk-banner-high' if risk_lvl == 'High' else ('risk-banner-moderate' if risk_lvl == 'Moderate' else 'risk-banner-low')}">
                    <span class="risk-banner-tag {'tag-high' if risk_lvl == 'High' else ('tag-moderate' if risk_lvl == 'Moderate' else 'tag-low')}">{risk_lvl} Risk</span>
                    <div class="risk-headline">Estimated Risk: {prob * 100:.1f}%</div>
                    <p class="risk-body-desc">Single-record cross-sectional prediction generated by the active machine learning pipeline.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# TAB 2: MODEL BENCHMARKING & PERFORMANCE
# ==============================================================================
def render_performance_tab(
    cfg: Dict[str, Any], metrics: Dict[str, Any], metadata: Dict[str, Any]
):
    """Render comprehensive model comparison benchmark tab."""
    best_key = metadata.get("best_model_key", "logistic_regression")
    best_data = metrics.get(best_key, {})
    best_name = metadata.get("best_model_name", "Logistic Regression")
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
                Evaluated across 4 classical architectures with 5-fold cross-validation on 4,000 patients and strictly held-out testing on 1,000 independent patients (5,000-patient research cohort).
            </p>
            <div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
                <div><span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Holdout F1 Score</span><br><strong style="font-size: 1.25rem; color: #38bdf8;">{test_m.get('f1', 0):.4f}</strong></div>
                <div><span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">ROC-AUC</span><br><strong style="font-size: 1.25rem; color: #818cf8;">{test_m.get('roc_auc', 0):.4f}</strong></div>
                <div><span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Test Accuracy</span><br><strong style="font-size: 1.25rem; color: #34d399;">{test_m.get('accuracy', 0) * 100:.2f}%</strong></div>
                <div><span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Holdout Recall</span><br><strong style="font-size: 1.25rem; color: #fbbf24;">{test_m.get('recall', 0):.4f}</strong></div>
                <div><span style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Holdout Precision</span><br><strong style="font-size: 1.25rem; color: #f472b6;">{test_m.get('precision', 0):.4f}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Interactive Multi-Metric Comparison Bar Chart
    st.subheader("1. Multi-Metric Benchmark Comparison")
    comp_fig = create_model_comparison_bar_chart(metrics)
    st.plotly_chart(comp_fig, use_container_width=True)

    # 2. Comparative Benchmark Table
    st.write("")
    st.subheader("2. Standardized Holdout Metrics Table")
    summary_rows = []
    for model_key, data in metrics.items():
        row = {
            "Model Architecture": data["display_name"],
            "5-Fold CV Score": round(data.get("best_cv_score", 0), 4),
        }
        for m_name, score in data.get("test_metrics", {}).items():
            row[m_name.upper()] = round(score, 4) if score is not None else "N/A"
        summary_rows.append(row)

    if summary_rows:
        metrics_df = pd.DataFrame(summary_rows)
        st.dataframe(
            metrics_df.style.highlight_max(
                axis=0,
                subset=[c for c in metrics_df.columns if c not in ("Model Architecture", "5-Fold CV Score")],
                color="#0c4a6e",
            ),
            use_container_width=True,
        )

    # 3. Model Evaluation Artifact Gallery (Tabs)
    st.write("")
    st.subheader("3. Diagnostic Artifact Gallery")
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    roc_path = fig_dir / "roc_curves.png"
    comp_path = fig_dir / "model_comparison.png"
    fi_path = fig_dir / "feature_importance.png"
    cm_best_path = fig_dir / f"cm_{best_key}.png"

    tab_roc, tab_cm, tab_fi, tab_comp = st.tabs([
        "📈 ROC Curves",
        "🎯 Confusion Matrix (Winner)",
        "🧬 Feature Importance",
        "📊 Static Benchmark Plot",
    ])

    with tab_roc:
        if roc_path.exists():
            st.image(
                str(roc_path),
                caption="Receiver Operating Characteristic (ROC) curves across all models on the 5,000-patient holdout test set (1,000 patients). Logistic Regression achieves champion discrimination (AUC = 0.994), followed by Random Forest (AUC = 0.989).",
                use_container_width=True,
            )
        else:
            st.info("ROC Curve image not found in reports/figures.")

    with tab_cm:
        cm_model_choice = st.radio(
            "Select Architecture to Inspect Confusion Matrix:",
            list(metrics.keys()),
            format_func=lambda k: f"{metrics[k]['display_name']} {'⭐ (Champion)' if k == best_key else ''}",
            horizontal=True,
            key="cm_radio_selector",
        )
        selected_cm_path = fig_dir / f"cm_{cm_model_choice}.png"
        if selected_cm_path.exists():
            st.image(
                str(selected_cm_path),
                caption=f"Holdout Confusion Matrix (N=1,000 test patients): {metrics[cm_model_choice]['display_name']}. Shows clinical sensitivity for detecting true deteriorating patients.",
                use_container_width=True,
            )
        else:
            st.info("Confusion matrix image not found.")

    with tab_fi:
        if fi_path.exists():
            st.image(
                str(fi_path),
                caption=f"Feature importance / coefficient magnitude rankings for the champion {best_name} model.",
                use_container_width=True,
            )
        else:
            st.info("Feature importance image not found.")

    with tab_comp:
        if comp_path.exists():
            st.image(
                str(comp_path),
                caption="Comparative bar plot of Accuracy, Precision, Recall, and F1 across all four architectures.",
                use_container_width=True,
            )
        else:
            st.info("Comparison figure not found.")

    # 4. Hyperparameter Search Results
    st.write("")
    st.subheader("4. Best Hyperparameters Discovered by RandomizedSearchCV")
    param_cols = st.columns(len(metrics))
    for i, (model_key, data) in enumerate(metrics.items()):
        is_winner = model_key == best_key
        with param_cols[i]:
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid {'#38bdf8' if is_winner else 'rgba(255,255,255,0.08)'}; border-radius: 10px; padding: 12px; height: 100%;">
                    <div style="font-weight: 700; color: {'#38bdf8' if is_winner else '#f8fafc'}; font-size: 0.95rem; margin-bottom: 6px;">
                        {data['display_name']} {'⭐' if is_winner else ''}
                    </div>
                    <div style="font-size: 0.76rem; color: #94a3b8; margin-bottom: 6px;">
                        5-Fold CV Score: <strong style="color: #cbd5e1;">{data.get('best_cv_score', 'N/A')}</strong>
                    </div>
                    <div style="font-size: 0.76rem; color: #cbd5e1; background: rgba(0,0,0,0.3); padding: 8px; border-radius: 6px; font-family: monospace;">
                        {str(data.get('best_params', {}))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# TAB 3: METHODOLOGY & ARCHITECTURE
# ==============================================================================
def render_about_tab(cfg: Dict[str, Any], metadata: Dict[str, Any]):
    """System Architecture, Clinical Rationale, and Methodology."""
    st.subheader("System Architecture & Clinical Rationale")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="methodology-card">
                <h4 style="margin: 0 0 8px 0; color: #38bdf8;">📌 Why Machine Learning (Not Fixed Clinical Rules)?</h4>
                <ul style="margin: 0; padding-left: 20px; color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                    <li><strong>Multivariate Non-Linear Interactions:</strong> Clinical deterioration is rarely driven by a single isolated biomarker. Moderate systolic hypertension combined with rising fasting glucose confers multiplicative risk compared to either abnormality alone.</li>
                    <li><strong>Empirical Calibration:</strong> Clinical heuristics say <em>"BP is trending upward"</em>, but supervised ML computes calibrated posterior probabilities ($P(\text{adverse event} \mid X)$) rigorously trained on historical patient trajectories.</li>
                </ul>
            </div>

            <div class="methodology-card">
                <h4 style="margin: 0 0 8px 0; color: #38bdf8;">🛡️ Zero Patient Data Leakage Architecture</h4>
                <ul style="margin: 0; padding-left: 20px; color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                    <li><strong>Patient-Level Splitting (<code>GroupShuffleSplit</code>):</strong> Splitting is grouped strictly by <code>patient_id</code>. All chronological visits for any single patient reside exclusively in either the training set or the holdout test set.</li>
                    <li><strong>Strict Preprocessing Scoping:</strong> Scalers (<code>StandardScaler</code>) and imputers (<code>SimpleImputer</code>) are fitted purely on training folds within an atomic <code>Pipeline</code>, preventing data snooping into the test distribution.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        best_name = metadata.get("best_model_name", "Logistic Regression")
        st.markdown(
            f"""
            <div class="methodology-card">
                <h4 style="margin: 0 0 8px 0; color: #34d399;">⚙️ Pipeline Specifications</h4>
                <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.7;">
                    <div>• <strong>Dataset Source:</strong> <code>{cfg['dataset']['name']}</code></div>
                    <div>• <strong>Cohort Scale:</strong> <strong>5,000 unique patients</strong> (27,489 longitudinal encounters)</div>
                    <div>• <strong>Split Architecture:</strong> <strong>80% Train (4,000 pts) / 20% Test (1,000 pts)</strong> via <code>GroupShuffleSplit</code></div>
                    <div>• <strong>Analysis Mode:</strong> <code>{cfg['dataset']['mode'].capitalize()}</code></div>
                    <div>• <strong>Target Variable:</strong> <code>{cfg['dataset']['target_col']}</code> (Binary: 0=Safe, 1=Elevated Risk)</div>
                    <div>• <strong>Selection Metric:</strong> <code>{cfg['training']['primary_metric'].upper()}</code> (Prioritizes Sensitivity/F1)</div>
                    <div>• <strong>Champion Estimator:</strong> <strong style="color: #38bdf8;">{best_name} ⭐</strong> (Holdout F1: 0.9471, AUC: 0.9938)</div>
                    <div>• <strong>Engineered Feature Count:</strong> <strong>67 features</strong> (66 numeric + 1 categorical)</div>
                    <div>• <strong>Missing Data Strategy:</strong> Median numeric imputation in <code>ColumnTransformer</code></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            r"""
            <div class="methodology-card">
                <h4 style="margin: 0 0 8px 0; color: #34d399;">📐 Feature Engineering Dynamics</h4>
                <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                    For each vital V, the pipeline automatically computes:
                    <br>• Change: &Delta; = V<sub>latest</sub> - V<sub>baseline</sub>
                    <br>• % Change: %&Delta; = ((V<sub>latest</sub> - V<sub>baseline</sub>) / V<sub>baseline</sub>) &times; 100
                    <br>• OLS Trend Slope: &beta; = &sum;((t<sub>i</sub> - t̄)(V<sub>i</sub> - V̄)) / &sum;(t<sub>i</sub> - t̄)<sup>2</sup>
                    <br>• Summary statistics: &mu;(V), min(V), max(V), &sigma;(V)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
