import os
import re
import warnings
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
from rich.console import Console
from rich.panel import Panel
from agent.agent_engine import run_retention_agent

# ── Rich console for terminal logs ─────────────────────────────────────────────
console = Console()

# ── Session state init ─────────────────────────────────────────────────────────
for key, default in {
    "customer_data": None,
    "churn_prob": 0.0,
    "agent_result": None,
    "active_provider": None,
    "show_dashboard": False,
    "active_tab": "Overview",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dropout — Telco Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #070d1a;
    color: #e2e8f0;
}

/* Fade-in animation */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.main { animation: fadeUp 0.55s ease-out; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #1e2d45;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.45);
    border-color: rgba(56, 189, 248, 0.45);
}
div[data-testid="metric-container"] label { color: #94a3b8 !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f8fafc !important; font-size: 1.6rem !important; font-weight: 700;
}

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    transition: all 0.25s ease;
    box-shadow: 0 4px 14px rgba(29,78,216,0.35);
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 20px rgba(29,78,216,0.5);
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
}

/* ── Section cards ── */
.section-card {
    background: #0f1e36;
    border: 1px solid #1e2d45;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 6px 24px rgba(0,0,0,0.25);
}

/* ── Hero shell ── */
.hero-shell {
    background: linear-gradient(135deg, #070d1a 0%, #0c1830 60%, #091526 100%);
    border: 1px solid #1e2d45;
    border-radius: 22px;
    padding: 1.2rem 1.6rem 2rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
}
.hero-shell::before {
    content: '';
    position: absolute; top: -40%; left: -20%;
    width: 60%; height: 200%;
    background: radial-gradient(ellipse, rgba(56,189,248,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.hero-nav {
    display: flex; justify-content: space-between; align-items: center;
    padding-bottom: 1.2rem; border-bottom: 1px solid #1e2d45; margin-bottom: 2rem;
}
.hero-brand {
    font-size: 1.3rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;
}
.hero-brand span { color: #38bdf8; }
.hero-nav-links { display: flex; gap: 1.4rem; color: #64748b; font-size: 0.9rem; font-weight: 500; }
.hero-title {
    font-size: clamp(2.2rem, 5vw, 4.2rem); font-weight: 800; line-height: 1.07;
    letter-spacing: -0.03em; color: #f8fafc; text-align: center; margin-bottom: 1rem;
}
.hero-title .accent { color: #38bdf8; }
.hero-sub {
    color: #94a3b8; font-size: clamp(1rem, 1.5vw, 1.2rem); max-width: 820px;
    margin: 0 auto 1.8rem; text-align: center; line-height: 1.65;
}
.pill-row { display: flex; justify-content: center; gap: 0.6rem; flex-wrap: wrap; }
.pill {
    background: #111827; border: 1px solid #1e2d45; color: #93c5fd;
    border-radius: 999px; padding: 0.32rem 0.85rem; font-size: 0.78rem; font-weight: 500;
}
.info-card {
    background: #0a1628; border: 1px solid #1e2d45; border-radius: 14px;
    padding: 1.1rem 1.3rem; min-height: 160px; height: 100%;
}
.info-card h4 { color: #e2e8f0; margin-bottom: 0.5rem; font-size: 0.98rem; }
.info-card p { color: #94a3b8; font-size: 0.9rem; line-height: 1.6; margin: 0; }
.flow-box {
    background: #0a1628; border: 1px solid #1e2d45; border-radius: 14px;
    padding: 1.2rem 1.5rem; margin-top: 1.2rem;
}
.flow-box h3 { color: #e2e8f0; margin-bottom: 0.7rem; }
.flow-step { color: #94a3b8; font-size: 0.93rem; margin-bottom: 0.35rem; }
.flow-step strong { color: #38bdf8; }

/* ── App header ── */
.app-header {
    background: linear-gradient(90deg, #0f1e36 0%, #0a1628 100%);
    border: 1px solid #1e2d45; border-radius: 16px; padding: 1.1rem 1.4rem;
    margin-bottom: 1rem; display: flex; align-items: center; gap: 1rem;
}
.app-header-icon { font-size: 2rem; }
.app-title { margin: 0; color: #f8fafc; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.02em; }
.app-subtitle { margin: 0.2rem 0 0; color: #64748b; font-size: 0.92rem; }
.badge {
    display: inline-block; margin-right: 0.4rem; margin-bottom: 0.3rem;
    background: #111827; color: #93c5fd; border: 1px solid #1e3a5f;
    border-radius: 999px; padding: 0.2rem 0.65rem; font-size: 0.76rem; font-weight: 500;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Tabs ── */
button[role="tab"] {
    font-weight: 600; font-size: 0.93rem; padding: 0.45rem 0.85rem; color: #64748b;
    border-radius: 8px 8px 0 0; transition: all 0.2s ease;
}
button[role="tab"]:hover { color: #e2e8f0; background: #0f1e36; }
button[role="tab"][aria-selected="true"] { color: #38bdf8 !important; background: #0f1e36; }

/* ── Download buttons ── */
div.stDownloadButton > button {
    background: #111827 !important;
    border: 1px solid #1e2d45 !important;
    color: #93c5fd !important;
    border-radius: 10px; font-weight: 600;
    transition: all 0.22s ease;
}
div.stDownloadButton > button:hover {
    border-color: #38bdf8 !important;
    background: #0f1e36 !important;
}

/* ── Status / Info / Error / Warning overrides ── */
div[data-testid="stAlert"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly chart theme ─────────────────────────────────────────────────────────
# Base layout — only keys that are NEVER overridden per-chart
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0a1628",
    font=dict(family="Inter", color="#e2e8f0", size=12),
    colorway=["#38bdf8", "#f97316", "#a78bfa", "#34d399"],
    hoverlabel=dict(bgcolor="#0f1e36", bordercolor="#1e2d45", font_size=13),
)
# Default axis / margin styles (used by apply_layout helper below)
_AXIS_STYLE = dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45")
_DEFAULT_MARGIN = dict(l=16, r=16, t=32, b=16)

def apply_layout(fig, height=None, margin=None, xaxis=None, yaxis=None, **extra):
    """Apply PLOTLY_LAYOUT + per-chart overrides without duplicate-key errors."""
    overrides = dict(
        height=height,
        margin=margin or _DEFAULT_MARGIN,
        xaxis={**_AXIS_STYLE, **(xaxis or {})},
        yaxis={**_AXIS_STYLE, **(yaxis or {})},
        **extra,
    )
    if height is None:
        overrides.pop("height")
    fig.update_layout(**PLOTLY_LAYOUT, **overrides)
    return fig

BLUE  = "#38bdf8"
ORG   = "#f97316"
PURP  = "#a78bfa"
GREEN = "#34d399"

# ── Artifacts / data ───────────────────────────────────────────────────────────
import joblib

ARTIFACTS = ["model_pipeline.pkl", "feature_columns.pkl"]
if not all(os.path.exists(f) for f in ARTIFACTS):
    st.error("Model artifacts not found. Run `churn.ipynb` first.")
    st.info("The dashboard requires trained model artifacts to function.")
    st.stop()

@st.cache_resource
def load_artifacts():
    pipeline       = joblib.load("model_pipeline.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return pipeline, feature_columns

@st.cache_data
def load_data():
    df = pd.read_csv("telco_customer_churn.csv").drop(columns=["customerID"])
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
    return df

def is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email.strip()))

pipeline, feature_columns = load_artifacts()
df = load_data()
console.print(Panel("[bold green]Dropout Dashboard loaded[/bold green]", title="[cyan]startup[/cyan]"))

# ══════════════════════════════════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_home_page():
    st.markdown("""
    <div class="hero-shell">
        <div class="hero-nav">
            <div class="hero-brand">Drop<span>out</span></div>
            <div class="hero-nav-links">
                <span>Overview</span><span>Predict</span><span>AI Strategist</span><span>Docs</span>
            </div>
        </div>
        <div style="padding:1rem 0 0.5rem;">
            <div class="hero-title">
                Predict churn early.<br/>
                <span class="accent">Retain every customer.</span>
            </div>
            <p class="hero-sub">
                A production-grade churn intelligence platform combining ML prediction,
                RAG-powered knowledge retrieval, and agentic AI reasoning — all in one workspace.
            </p>
            <div class="pill-row">
                <span class="pill">🤖 LangGraph Agent</span>
                <span class="pill">📊 ML Pipeline</span>
                <span class="pill">🔍 RAG Knowledge Base</span>
                <span class="pill">📥 Downloadable Reports</span>
                <span class="pill">🎨 Interactive Charts</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    vector_index_ready = any(os.path.exists(p) for p in [
        "vectorstore/db_faiss/index.faiss",
        "vectorstore/db_faiss/index 2.faiss",
    ])
    kb_ready = os.path.exists("knowledge_base/retention_strategies.md")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset Rows", f"{len(df):,}")
    m2.metric("Model Features", f"{len(feature_columns)}")
    m3.metric("Knowledge Base", "✅ Ready" if kb_ready else "❌ Missing")
    m4.metric("Vector Index",   "✅ Ready" if vector_index_ready else "❌ Missing")
    style_metric_cards(background_color="#0f1e36", border_left_color="#38bdf8", border_color="#1e2d45")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="info-card">
            <h4>🎯 What it does</h4>
            <p>Estimates churn probability from customer profile data and converts ML outputs into
               business-focused retention guidance via an AI agent.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="info-card">
            <h4>⚙️ How it works</h4>
            <p>Input customer attributes → preprocessing pipeline → risk scoring →
               LangGraph + RAG agent → personalised intervention plan.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="info-card">
            <h4>📂 Data supported</h4>
            <p>Structured CSV records, model artifacts (.pkl), markdown strategy docs,
               and FAISS vector indexes for retrieval-augmented planning.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="flow-box">
        <h3>Product Flow</h3>
        <p class="flow-step"><strong>1.</strong> Explore churn trends in the analytics dashboard.</p>
        <p class="flow-step"><strong>2.</strong> Predict churn probability for any customer profile.</p>
        <p class="flow-step"><strong>3.</strong> Generate an expert retention strategy and download the report.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀  Enter Dashboard", width='stretch'):
        st.session_state.show_dashboard = True
        st.rerun()


if not st.session_state.show_dashboard:
    render_home_page()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  (option-menu nav)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0.8rem 0 1.2rem; border-bottom:1px solid #1e2d45; margin-bottom:1rem;">
        <div style="font-size:1.25rem;font-weight:800;color:#f8fafc;letter-spacing:-0.02em;">
            Drop<span style="color:#38bdf8;">out</span>
        </div>
        <div style="color:#475569;font-size:0.8rem;margin-top:0.2rem;">Telco Churn Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Overview", "Churn Prediction", "AI Strategist", "Model Performance"],
        icons=["bar-chart-fill", "cpu-fill", "robot", "activity"],
        default_index=["Overview", "Churn Prediction", "AI Strategist", "Model Performance"].index(
            st.session_state.active_tab
        ),
        styles={
            "container":      {"background-color": "transparent", "padding": "0"},
            "nav-link":       {"font-size": "0.9rem", "font-weight": "500",
                               "color": "#94a3b8", "border-radius": "10px",
                               "margin": "2px 0", "--hover-color": "#111827"},
            "nav-link-selected": {"background": "linear-gradient(135deg,#1d4ed8,#1e40af)",
                                   "color": "white", "font-weight": "600"},
            "icon":           {"color": "#38bdf8"},
        },
    )
    st.session_state.active_tab = selected

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick dataset stats
    churn_rate = (df["Churn"] == "Yes").sum() / len(df) * 100
    st.markdown(f"""
    <div style="background:#0a1628;border:1px solid #1e2d45;border-radius:12px;padding:0.9rem 1rem;">
        <div style="color:#475569;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.6rem;">Dataset Snapshot</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:0.35rem;">
            <span style="color:#94a3b8;font-size:0.83rem;">Total customers</span>
            <span style="color:#e2e8f0;font-weight:600;font-size:0.83rem;">{len(df):,}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:0.35rem;">
            <span style="color:#94a3b8;font-size:0.83rem;">Churn rate</span>
            <span style="color:#f97316;font-weight:600;font-size:0.83rem;">{churn_rate:.1f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:#94a3b8;font-size:0.83rem;">Features</span>
            <span style="color:#e2e8f0;font-weight:600;font-size:0.83rem;">{len(feature_columns)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Home", width='stretch'):
        st.session_state.show_dashboard = False
        st.rerun()

# ── App header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-header-icon">📡</div>
    <div>
        <h1 class="app-title">Dropout — Churn Command Center</h1>
        <p class="app-subtitle">Monitor churn signals · Predict risk · Generate AI-powered retention actions</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if selected == "Overview":
    total      = len(df)
    churned    = (df["Churn"] == "Yes").sum()
    churn_rate = churned / total * 100
    avg_monthly = df["MonthlyCharges"].mean()
    avg_tenure  = df["tenure"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers",     f"{total:,}")
    m2.metric("Churned",             f"{churned:,}",    delta=f"-{churn_rate:.1f}% of base", delta_color="inverse")
    m3.metric("Churn Rate",          f"{churn_rate:.1f}%")
    m4.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
    style_metric_cards(background_color="#0f1e36", border_left_color="#38bdf8", border_color="#1e2d45")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: donut + contract bar ──
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Churn Distribution")
        counts = df["Churn"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        counts["Status"] = counts["Status"].map({"No": "Retained", "Yes": "Churned"})
        fig = go.Figure(go.Pie(
            labels=counts["Status"], values=counts["Count"],
            hole=0.58, marker_colors=[BLUE, ORG],
            textinfo="percent+label",
            textfont=dict(size=13, color="#e2e8f0"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        ))
        apply_layout(fig, height=300, showlegend=False)
        fig.add_annotation(text=f"<b>{churn_rate:.0f}%</b><br><span style='font-size:11px'>churn</span>",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=18, color="#f97316"))
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Contract Type vs Churn")
        contract_churn = df.groupby(["Contract", "Churn"]).size().reset_index(name="count")
        contract_churn["Churn"] = contract_churn["Churn"].map({"No": "Retained", "Yes": "Churned"})
        fig = px.bar(contract_churn, x="Contract", y="count", color="Churn",
                     barmode="group", color_discrete_map={"Retained": BLUE, "Churned": ORG},
                     labels={"count": "Customers", "Contract": "Contract Type"},
                     hover_data={"count": ":,"},
                     template="plotly_dark")
        apply_layout(fig, height=300, bargap=0.28)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: tenure histogram + monthly charges violin ──
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Tenure Distribution by Churn")
        fig = px.histogram(
            df, x="tenure", color="Churn", nbins=35,
            color_discrete_map={"No": BLUE, "Yes": ORG},
            labels={"tenure": "Tenure (months)", "count": "Customers"},
            opacity=0.8, barmode="overlay", template="plotly_dark",
        )
        apply_layout(fig, height=300, legend=dict(title="", bgcolor="rgba(0,0,0,0)"))
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Monthly Charges by Churn")
        fig = px.violin(
            df, x="Churn", y="MonthlyCharges", color="Churn", box=True,
            color_discrete_map={"No": BLUE, "Yes": ORG},
            labels={"MonthlyCharges": "Monthly Charges ($)", "Churn": ""},
            template="plotly_dark", points="outliers",
        )
        apply_layout(fig, height=300, showlegend=False)
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: internet service + heatmap ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Internet Service vs Churn")
    internet_churn = df.groupby(["InternetService", "Churn"]).size().reset_index(name="count")
    internet_churn["Churn"] = internet_churn["Churn"].map({"No": "Retained", "Yes": "Churned"})
    fig = px.bar(internet_churn, x="InternetService", y="count", color="Churn",
                 barmode="group", color_discrete_map={"Retained": BLUE, "Churned": ORG},
                 labels={"count": "Customers", "InternetService": "Internet Service"},
                 template="plotly_dark")
    apply_layout(fig, height=300, bargap=0.3)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Correlation heatmap  ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Feature Correlation Heatmap")
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    corr_df  = df[num_cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr_df.values, x=corr_df.columns, y=corr_df.columns,
        colorscale=[[0,"#0a1628"],[0.5,"#1d4ed8"],[1,"#38bdf8"]],
        text=corr_df.values.round(2), texttemplate="%{text}",
        hovertemplate="%{x} × %{y}: %{z}<extra></extra>",
    ))
    apply_layout(fig, height=450, margin=dict(l=50, r=50, t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Sample Data (first 100 rows)")
    st.dataframe(df.head(100), width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — CHURN PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Churn Prediction":
    st.subheader("Predict Customer Churn")
    st.caption("Fill in the customer profile and run inference to estimate churn probability.")

    with st.form("prediction_form"):
        st.markdown("##### Contact Details")
        i1, i2, i3 = st.columns(3)
        with i1:
            with st.container(border=True):
                customer_name = st.text_input("Customer Name", value="")
        with i2:
            with st.container(border=True):
                customer_email = st.text_input("Customer Email", value="", placeholder="name@example.com")
        with i3:
            with st.container(border=True):
                company_name = st.text_input("Company Name", value="Telco")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### Account Info")
            tenure          = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
            contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment_method  = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            monthly_charges   = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 0.5)
            total_charges     = st.number_input("Total Charges ($)",   0.0, 10000.0, 780.0, 1.0)

        with col2:
            st.markdown("##### Demographics")
            gender         = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner        = st.selectbox("Partner", ["Yes", "No"])
            dependents     = st.selectbox("Dependents", ["Yes", "No"])
            st.markdown("##### Phone")
            phone_service  = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        with col3:
            st.markdown("##### Internet Services")
            internet_service  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security   = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
            online_backup     = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support      = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
            streaming_tv      = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
            streaming_movies  = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("🔍  Predict Churn", width='stretch')

    if submitted and not is_valid_email(customer_email):
        st.error("Please enter a valid email address before running prediction.")

    if submitted and is_valid_email(customer_email):
        with st.status("Analysing customer profile...", expanded=True) as status:
            st.write("Extracting demographic and service data...")
            time.sleep(0.4)
            st.write("Running preprocessing pipeline...")
            time.sleep(0.4)
            st.write("Executing predictive model...")

            input_data = {
                "CustomerName": customer_name.strip() or "Customer",
                "CustomerEmail": customer_email.strip(),
                "CompanyName": company_name.strip() or "Telco",
                "gender": gender,
                "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
                "Partner": partner, "Dependents": dependents,
                "tenure": tenure, "PhoneService": phone_service,
                "MultipleLines": multiple_lines, "InternetService": internet_service,
                "OnlineSecurity": online_security, "OnlineBackup": online_backup,
                "DeviceProtection": device_protection, "TechSupport": tech_support,
                "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
                "Contract": contract, "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
            }

            input_df  = pd.DataFrame([input_data]).reindex(columns=feature_columns, fill_value=0)
            prediction = pipeline.predict(input_df)[0]
            proba      = pipeline.predict_proba(input_df)[0]
            churn_prob = proba[1] * 100
            stay_prob  = proba[0] * 100

            st.session_state.customer_data = input_data
            st.session_state.churn_prob    = churn_prob
            st.session_state.agent_result  = None
            time.sleep(0.4)
            status.update(label="Analysis complete ✓", state="complete", expanded=False)

        console.print(Panel(
            f"Customer: [bold]{customer_name}[/bold] | Churn prob: [bold red]{churn_prob:.1f}%[/bold red]",
            title="[cyan]prediction[/cyan]"
        ))
        st.toast("Prediction generated ✅")

        st.divider()
        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns([1, 1.8, 1])

        with r1:
            if prediction == 1:
                st.error(f"⚠️  High Churn Risk\n\n**{churn_prob:.1f}%** probability")
            else:
                st.success(f"✅  Likely to Stay\n\n**{stay_prob:.1f}%** probability")

        with r2:
            # Gauge chart
            gauge_color = ORG if churn_prob > 50 else BLUE
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob,
                number={"suffix": "%", "font": {"size": 28, "color": gauge_color}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"size": 11}},
                    "bar":  {"color": gauge_color, "thickness": 0.25},
                    "bgcolor": "#0a1628",
                    "bordercolor": "#1e2d45",
                    "steps": [
                        {"range": [0,   40], "color": "#0a2a1a"},
                        {"range": [40,  65], "color": "#2a1f0a"},
                        {"range": [65, 100], "color": "#2a0e0e"},
                    ],
                    "threshold": {"line": {"color": gauge_color, "width": 3}, "value": churn_prob},
                },
                title={"text": "Churn Probability", "font": {"size": 14, "color": "#94a3b8"}},
            ))
            apply_layout(fig, height=240, margin=dict(l=16, r=16, t=40, b=0))
            st.plotly_chart(fig, width='stretch')

        with r3:
            st.markdown("<br><br>", unsafe_allow_html=True)
            # Horizontal prob bar
            fig2 = go.Figure(go.Bar(
                x=[stay_prob, churn_prob], y=["Stays", "Churns"],
                orientation="h",
                marker_color=[BLUE, ORG],
                text=[f"{stay_prob:.1f}%", f"{churn_prob:.1f}%"],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=13),
            ))
            apply_layout(fig2, height=200, xaxis=dict(range=[0, 120]))
            st.plotly_chart(fig2, width='stretch')

        st.divider()
        st.subheader("Customer Insights & Recommendations")
        ci1, ci2 = st.columns(2)

        with ci1:
            st.markdown("##### Key Risk Factors")
            factors = []
            if contract        == "Month-to-month": factors.append("**Month-to-month contract** — highest churn segment.")
            if internet_service == "Fiber optic":   factors.append("**Fiber optic** — above-average churn rate.")
            if tech_support    == "No":             factors.append("**No Tech Support** — strong churn predictor.")
            if tenure          < 12:                factors.append("**Low tenure (<1 yr)** — critical risk window.")
            if not factors:
                st.success("No major churn risk factors detected in this profile.")
            else:
                for f in factors:
                    st.markdown(f"- {f}")

        with ci2:
            st.markdown("##### Recommended Actions")
            if prediction == 1:
                st.markdown("- 💰 **Offer 10–20% discount** to switch to a 1-year contract.")
                if tech_support    == "No":           st.markdown("- 🛠  **3 months free Tech Support** — add significant perceived value.")
                if internet_service == "Fiber optic": st.markdown("- 📞 **Proactive service call** to address fiber network satisfaction.")
            else:
                st.markdown("- 🎁 **Upsell opportunity** — offer hardware upgrades or add-on streaming.")
                st.markdown("- 🙏 **Referral bonus** — send loyalty reward email.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Customer vs Average Metrics")
        metrics       = ["Tenure (months)", "Monthly Charges ($)", "Total Charges ($)"]
        customer_vals = [tenure, monthly_charges, total_charges]
        avg_vals      = [df["tenure"].mean(), df["MonthlyCharges"].mean(), df["TotalCharges"].mean()]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="This Customer", x=metrics, y=customer_vals,
                              marker_color=ORG if prediction == 1 else BLUE,
                              text=[f"{v:,.1f}" for v in customer_vals], textposition="outside"))
        fig3.add_trace(go.Bar(name="Dataset Average", x=metrics, y=avg_vals,
                              marker_color="#1e293b",
                              text=[f"{v:,.1f}" for v in avg_vals], textposition="outside"))
        fig3.update_traces(textfont=dict(color="#e2e8f0", size=12))
        apply_layout(fig3, height=320, barmode="group", bargap=0.3)
        st.plotly_chart(fig3, width='stretch')

        st.divider()
        st.subheader("Export Results")
        export_data = input_data.copy()
        export_data["Prediction_Churn"]    = "Yes" if prediction == 1 else "No"
        export_data["Churn_Probability"]   = f"{churn_prob:.1f}%"
        export_data["Stay_Probability"]    = f"{stay_prob:.1f}%"
        csv_data = pd.DataFrame([export_data]).to_csv(index=False).encode("utf-8")
        st.download_button("📥  Download Prediction as CSV", csv_data,
                           "churn_prediction_result.csv", "text/csv", width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — AI STRATEGIST
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "AI Strategist":
    st.subheader("AI-Driven Retention Strategy")

    if st.session_state.customer_data is None:
        st.info("🔍  Run a prediction in **Churn Prediction** first to enable the AI Strategist.")
    else:
        st.markdown("""
        This agent uses **LangGraph** to process customer data, query a **RAG knowledge base**
        of retention strategies, and generate a personalised intervention plan via **Gemini Flash**.
        """)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### Target Customer Profile")
            st.json(st.session_state.customer_data)
        with c2:
            st.markdown("### Risk Context")
            st.metric("Churn Probability", f"{st.session_state.churn_prob:.1f}%")
            style_metric_cards(background_color="#0f1e36", border_left_color="#f97316", border_color="#1e2d45")
            if st.session_state.churn_prob > 50:
                st.error("Priority: **CRITICAL** — High Churn Risk")
            else:
                st.warning("Priority: **MEDIUM** — Maintenance Mode")

        st.divider()

        if st.button("🤖  Start Expert AI Analysis", width='stretch'):
            with st.status("Senior Agent thinking...", expanded=True) as status:
                try:
                    progress_bar        = st.progress(0)
                    eta_text            = st.empty()
                    estimated_total_sec = 14
                    start_time          = time.time()

                    progress_bar.progress(8)
                    eta_text.caption(f"Estimated time remaining: ~{estimated_total_sec}s")

                    result = run_retention_agent(
                        st.session_state.customer_data,
                        st.session_state.churn_prob,
                    )

                    progress_bar.progress(65)
                    elapsed   = time.time() - start_time
                    remaining = max(0, int(estimated_total_sec - elapsed))
                    eta_text.caption(f"Estimated time remaining: ~{remaining}s")

                    thought_log = result.get("thought_log", [])
                    total_logs  = max(1, len(thought_log))
                    for i, log_entry in enumerate(thought_log, start=1):
                        st.write(log_entry)
                        time.sleep(0.3)
                        progress = min(95, 65 + int((i / total_logs) * 30))
                        progress_bar.progress(progress)
                        elapsed   = time.time() - start_time
                        remaining = max(0, int(estimated_total_sec - elapsed))
                        eta_text.caption(f"Estimated time remaining: ~{remaining}s")

                    st.session_state.agent_result    = result["final_report"]
                    st.session_state.active_provider = result.get("active_provider", "Unknown")
                    progress_bar.progress(100)
                    total_elapsed = time.time() - start_time
                    eta_text.caption(f"Completed in {total_elapsed:.1f}s")

                    console.print(Panel(
                        f"Provider: [bold cyan]{st.session_state.active_provider}[/bold cyan] | "
                        f"Time: [bold]{total_elapsed:.1f}s[/bold]",
                        title="[green]agent result[/green]"
                    ))

                    if st.session_state.active_provider == "Heuristic Mode":
                        status.update(label="Strategy Generated (Safe Mode Fallback)", state="complete", expanded=False)
                        st.toast("AI quota reached — using expert heuristics fallback.")
                    else:
                        status.update(label=f"Strategy Ready via {st.session_state.active_provider}!", state="complete", expanded=False)

                except Exception as e:
                    st.error(f"Agent Workflow Error: {str(e)}")
                    st.info("Tip: Verify your API keys or check network access.")
                    status.update(label="Operation Failed", state="error")

        if st.session_state.agent_result:
            if st.session_state.active_provider != "Heuristic Mode":
                st.success(f"✅  Analysis Complete! (Generated via **{st.session_state.active_provider}**)")
            else:
                st.info("Expert Strategy Ready (Heuristic Fallback Mode)")

            st.markdown(st.session_state.agent_result)

            report_text = f"DROPOUT — TELCO CHURN STRATEGY REPORT\n\n{st.session_state.agent_result}"
            st.download_button(
                "📥  Download Full Retention Report",
                report_text,
                "retention_strategy_report.md",
                "text/markdown",
                width='stretch',
            )
# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Model Performance":
    st.subheader("Model Performance & Diagnostics")
    st.caption("Detailed view of the trained Machine Learning pipeline and its evaluation metrics.")

    # ── Performance Metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Accuracy", "77.2%", help="Overall percentage of correct predictions")
    m2.metric("Precision (Churn)", "54.0%", help="How many selected as 'Churn' were actually 'Churn'")
    m3.metric("Recall (Churn)", "58.0%", help="How many actual 'Churn' cases were correctly identified")
    m4.metric("F1 Score (Churn)", "56.0%", help="Harmonic mean of Precision and Recall")
    style_metric_cards(background_color="#0f1e36", border_left_color="#a78bfa", border_color="#1e2d45")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model Benchmarking Leaderboard ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Model Benchmarking Leaderboard")
    st.caption("Comparison of the top 3 models evaluated during the model selection phase.")
    
    col_bench1, col_bench2 = st.columns([1.5, 1])
    
    with col_bench1:
        # Leaderboard Chart
        bench_data = pd.DataFrame({
            "Model": ["Random Forest", "XGBoost", "Decision Tree"],
            "Accuracy": [0.842, 0.838, 0.787],
            "Rank": ["Gold", "Silver", "Bronze"]
        })
        
        fig_bench = px.bar(bench_data, x="Accuracy", y="Model", orientation="h",
                          text=bench_data["Accuracy"].apply(lambda x: f"{x*100:.1f}%"),
                          color="Model",
                          color_discrete_map={
                              "Random Forest": "#fbbf24", # Gold
                              "XGBoost": "#94a3b8",      # Silver
                              "Decision Tree": "#92400e" # Bronze
                          },
                          template="plotly_dark")
        
        apply_layout(fig_bench, height=300, showlegend=False)
        fig_bench.update_traces(textposition='inside', marker_line_width=0)
        st.plotly_chart(fig_bench, use_container_width=True)
        
    with col_bench2:
        # Winner Card
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(10, 22, 40, 0.1) 100%); 
                    border: 1px solid rgba(251, 191, 36, 0.3); padding: 25px; border-radius: 12px; height: 100%;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            <h3 style="color: #fbbf24; margin-top: 0; display: flex; align-items: center; gap: 10px;">
                <span>🏆</span> The Winner
            </h3>
            <h4 style="margin: 15px 0 10px 0; color: #fff;">Random Forest Classifier</h4>
            <p style="font-size: 0.95rem; color: #94a3b8; line-height: 1.5;">
                Chosen as the best model with a <b>84.2%</b> cross-validation accuracy. 
                It demonstrates the best balance between precision and recall for churn detection.
            </p>
            <div style="background: rgba(251, 191, 36, 0.2); color: #fbbf24; padding: 6px 12px; 
                        border-radius: 20px; display: inline-block; font-size: 0.8rem; font-weight: bold; margin-top: 10px;">
                BEST PERFORMANCE
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.2])

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        # Fixed data from churn.ipynb evaluation on test set
        z = [[882, 171], [149, 203]]
        x = ["Predicted: No", "Predicted: Yes"]
        y = ["Actual: No", "Actual: Yes"]
        
        fig = ff_fig = go.Figure(data=go.Heatmap(
            z=z, x=x, y=y,
            colorscale=[[0, "#0a1628"], [0.5, "#1d4ed8"], [1, "#38bdf8"]],
            text=[[str(v) for v in row] for row in z],
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>%{x}<br>Count: %{z}<extra></extra>",
            showscale=False
        ))
        apply_layout(fig, height=350, margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Feature Importance")
        
        # Extract features from the pipeline
        try:
            # The model is part of an imblearn/sklearn pipeline
            preprocessor = pipeline.named_steps.get('preprocessor')
            classifier = pipeline.named_steps.get('classifier', pipeline)
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # Get names from preprocessor if available, otherwise fallback
                if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                    names = preprocessor.get_feature_names_out()
                else:
                    names = feature_columns
                
                if len(names) == len(importances):
                    feat_df = pd.DataFrame({
                        "Feature": names,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=True).tail(12)
                    
                    # Clean up feature names for better readability
                    feat_df["Feature"] = feat_df["Feature"].str.replace(r'^(cat|num)__', '', regex=True)
                    
                    fig = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                                 color="Importance", color_continuous_scale="Blues",
                                 template="plotly_dark")
                    apply_layout(fig, height=350, showlegend=False, coloraxis_showscale=False)
                    fig.update_traces(marker_line_width=0)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.error(f"Mismatch: {len(names)} features vs {len(importances)} importances.")
            else:
                st.info("Feature importance not available for this model type.")
        except Exception as e:
            st.error(f"Error extracting feature importance: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Model Config Card ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Model Configuration")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### Pipeline Architecture")
        st.code("""
Pipeline(steps=[
    ('classifier', RandomForestClassifier(
        random_state=42, 
        n_estimators=100
    ))
])
        """, language="python")

    with col_b:
        st.markdown("##### Training Summary")
        st.markdown(f"""
        - **Model Type:** Random Forest Classifier
        - **Total Training Samples:** ~5,600 (with SMOTE)
        - **Test Samples:** 1,405
        - **Input Features:** {len(feature_columns)}
        - **Data Prep:** Label Encoding + SMOTE Oversampling
        """)
    st.markdown('</div>', unsafe_allow_html=True)
