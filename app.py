import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import time
from agent.agent_engine import run_retention_agent

# Initialize session state for agent reasoning
if "customer_data" not in st.session_state:
    st.session_state.customer_data = None
if "churn_prob" not in st.session_state:
    st.session_state.churn_prob = 0.0
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None
if "active_provider" not in st.session_state:
    st.session_state.active_provider = None
if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Telco Churn Dashboard",
    layout="wide",
)

st.markdown("""
<style>
    /* Global Font and Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Fade in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main {
        animation: fadeIn 0.6s ease-out;
    }

    /* Metric Cards Glassmorphism */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(0, 210, 255, 0.3);
    }
    div[data-testid="metric-container"] > div {
        color: #e2e8f0;
    }

    /* Buttons */
    div.stButton > button {
        background: #1d4ed8;
        color: white;
        border: 1px solid #1e40af;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(29, 78, 216, 0.35);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(29, 78, 216, 0.45);
        background: #1e40af;
        color: white;
        border-color: #1e3a8a;
    }

    /* Tabs formatting */
    button[role="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.45rem 0.8rem;
        border-radius: 8px 8px 0 0;
        color: #cbd5e1;
        transition: all 0.2s ease;
    }
    button[role="tab"]:hover {
        color: #e2e8f0;
        background: #111827;
    }
    button[role="tab"][aria-selected="true"] {
        color: #e2e8f0;
        background: #0f172a;
    }
    
    /* Custom divider line */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* Landing page - premium hero */
    .home-shell {
        border: 1px solid #1f2937;
        border-radius: 22px;
        padding: 1.1rem 1.2rem 1.8rem 1.2rem;
        background: #0b1220;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
    }

    .home-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        padding: 0.2rem 0.2rem 1.2rem 0.2rem;
        border-bottom: 1px solid #1f2937;
        margin-bottom: 1.7rem;
    }

    .home-brand {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: 0.2px;
    }

    .home-nav-links {
        color: #94a3b8;
        font-size: 0.92rem;
        font-weight: 500;
        display: flex;
        gap: 1.2rem;
        flex-wrap: wrap;
        justify-content: flex-end;
    }

    .home-hero {
        padding: 1.2rem 1rem 0.6rem 1rem;
        text-align: center;
    }

    .home-title {
        color: #f8fafc;
        font-size: clamp(2rem, 4.6vw, 4.1rem);
        line-height: 1.08;
        letter-spacing: -0.02em;
        margin-bottom: 0.9rem;
        font-weight: 700;
    }

    .home-subtitle {
        color: #aab8cc;
        max-width: 860px;
        margin: 0 auto 1.4rem auto;
        font-size: clamp(1rem, 1.4vw, 1.25rem);
    }

    .home-pill-row {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.7rem;
    }

    .home-chip {
        display: inline-block;
        border: 1px solid #334155;
        color: #dbeafe;
        background: #111827;
        border-radius: 999px;
        padding: 0.34rem 0.8rem;
        font-size: 0.78rem;
    }

    .home-card {
        margin-top: 1.2rem;
        background: #0f172a;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 170px;
    }

    .home-subtle {
        color: #cbd5e1;
    }

    .home-metric-label {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 0.3rem;
    }

    .home-flow {
        margin-top: 1rem;
        border: 1px solid #1f2937;
        background: #0f172a;
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }

    /* Dashboard professional header */
    .app-header {
        border: 1px solid #1f2937;
        background: #0f172a;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }

    .app-title {
        margin: 0;
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        margin: 0.3rem 0 0 0;
        color: #94a3b8;
        font-size: 0.98rem;
    }

    .badge-row {
        margin-top: 0.7rem;
    }

    .badge {
        display: inline-block;
        margin-right: 0.45rem;
        margin-bottom: 0.35rem;
        background: #111827;
        color: #cbd5e1;
        border: 1px solid #334155;
        border-radius: 999px;
        padding: 0.2rem 0.6rem;
        font-size: 0.78rem;
    }
</style>
""", unsafe_allow_html=True)

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
import os
import joblib

ARTIFACTS = ["model_pipeline.pkl", "feature_columns.pkl"]

if not all(os.path.exists(f) for f in ARTIFACTS):
    st.error("Model artifacts not found. Please run the notebook first to generate `model_pipeline.pkl` and `feature_columns.pkl`.", icon="⚠️")
    st.info("The dashboard requires a trained model to function. Please execute `churn.ipynb` to train the Random Forest model.")
    st.stop()

@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("model_pipeline.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return pipeline, feature_columns

@st.cache_data
def load_data():
    df = pd.read_csv("telco_customer_churn.csv").drop(columns=["customerID"])
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
    return df


pipeline, feature_columns = load_artifacts()
df = load_data()


def render_home_page():
    st.markdown(
        """
        <div class="home-shell">
            <div class="home-nav">
                <div class="home-brand">TelcoAI</div>
                <div class="home-nav-links">
                    <span>Products</span>
                    <span>Docs</span>
                    <span>Ecosystem</span>
                    <span>Resources</span>
                    <span>Pricing</span>
                </div>
            </div>
            <div class="home-hero">
                <div class="home-title">
                    Predict telecom churn early,<br/>retain customers with AI strategies
                </div>
                <p class="home-subtitle">
                    This platform combines a production-ready machine learning pipeline with agentic reasoning
                    to transform churn probability into practical, customer-specific retention actions.
                </p>
                <div class="home-pill-row">
                    <span class="home-chip">ML Prediction Engine</span>
                    <span class="home-chip">RAG Knowledge Base</span>
                    <span class="home-chip">Executive Dashboard</span>
                    <span class="home-chip">Downloadable Reports</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="home-card">
                <h4>What this project does</h4>
                <p class="home-subtle">
                    The app estimates churn probability from customer profile data and
                    converts model outputs into business-focused retention guidance.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="home-card">
                <h4>How it works</h4>
                <p class="home-subtle">
                    Input customer attributes, process via trained artifacts, generate risk scores,
                    and activate an AI strategist powered by LangGraph and RAG for intervention planning.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="home-card">
                <h4>Supported data now</h4>
                <p class="home-subtle">
                    Structured customer records (CSV/form fields), model artifacts, markdown strategy
                    documents, and FAISS vector indexes used for retrieval-augmented planning.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    vector_index_ready = any(
        os.path.exists(path)
        for path in [
            "vectorstore/db_faiss/index.faiss",
            "vectorstore/db_faiss/index 2.faiss",
        ]
    )
    kb_ready = os.path.exists("knowledge_base/retention_strategies.md")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="home-metric-label">Dataset Rows</div>', unsafe_allow_html=True)
        st.markdown(f"### {len(df):,}")
    with k2:
        st.markdown('<div class="home-metric-label">Model Features</div>', unsafe_allow_html=True)
        st.markdown(f"### {len(feature_columns)}")
    with k3:
        st.markdown('<div class="home-metric-label">Knowledge Base</div>', unsafe_allow_html=True)
        st.markdown(f"### {'Ready' if kb_ready else 'Missing'}")
    with k4:
        st.markdown('<div class="home-metric-label">Vector Index</div>', unsafe_allow_html=True)
        st.markdown(f"### {'Ready' if vector_index_ready else 'Missing'}")

    st.markdown(
        """
        <div class="home-flow">
            <h3 style="margin-bottom:0.5rem;">Product Flow</h3>
            <p class="home-subtle">1. Explore churn trends in the analytics dashboard.</p>
            <p class="home-subtle">2. Predict churn probability for any customer profile.</p>
            <p class="home-subtle">3. Generate an expert retention strategy and download the report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Enter Dashboard", use_container_width=True):
        st.session_state.show_dashboard = True
        st.rerun()


if not st.session_state.show_dashboard:
    render_home_page()
    st.stop()

with st.sidebar:
    if st.button("Back to Home", use_container_width=True):
        st.session_state.show_dashboard = False
        st.rerun()

st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">Telco Churn Command Center</h1>
        <p class="app-subtitle">
            Monitor churn signals, predict risk, and generate AI-powered retention actions from one workspace.
        </p>
        <div class="badge-row">
            <span class="badge">Dataset Analytics</span>
            <span class="badge">Predictive Scoring</span>
            <span class="badge">Agentic Retention Planning</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
tab1, tab2, tab3 = st.tabs(["Overview", "Churn Prediction", "AI Strategist"])


with tab1:
    with st.spinner("Loading dashboard metrics..."):
        total = len(df)
        churned = (df["Churn"] == "Yes").sum()
        churn_rate = churned / total * 100
        avg_monthly = df["MonthlyCharges"].mean()
        avg_tenure = df["tenure"].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Customers", f"{total:,}")
        m2.metric("Churned", f"{churned:,}")
        m3.metric("Churn Rate", f"{churn_rate:.1f}%")
        m4.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")

        st.divider()
        c1, c2 = st.columns(2)

        # Global dark mode chart settings
        plt.style.use('dark_background')
        CHART_COLORS = ["#38bdf8", "#f97316"]
        CHART_BG = "#0f172a"
        CHART_TEXT = "#e2e8f0"
        CHART_GRID = "#334155"
        CHART_EDGE = "#ffffff26"

        def style_axis(ax):
            ax.set_facecolor(CHART_BG)
            ax.tick_params(colors=CHART_TEXT, which='both', labelsize=9)
            ax.grid(True, linestyle='--', alpha=0.25, axis='y', color=CHART_GRID)
            ax.spines[["top", "right"]].set_visible(False)
            for side in ["left", "bottom"]:
                ax.spines[side].set_color(CHART_EDGE)

        with c1:
            st.subheader("Churn Distribution")
            fig, ax = plt.subplots(figsize=(3.2, 2.7))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor(CHART_BG)
            counts = df["Churn"].value_counts()
            ax.pie(
                counts,
                labels=["No Churn", "Churned"],
                autopct="%1.1f%%",
                colors=CHART_COLORS,
                startangle=90,
                pctdistance=0.72,
                textprops={"color": CHART_TEXT, "fontsize": 10, "fontweight": "medium"},
                wedgeprops={"edgecolor": "#ffffff33", "linewidth": 1.2, "width": 0.42},
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            st.subheader("Contract Type vs Churn")
            fig, ax = plt.subplots(figsize=(4.1, 2.9))
            fig.patch.set_alpha(0.0)
            style_axis(ax)
            contract_churn = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
            contract_churn.plot(
                kind="bar", ax=ax, color=CHART_COLORS, edgecolor=CHART_EDGE, width=0.68
            )
            ax.set_xlabel("Contract Type", fontsize=9, color=CHART_TEXT)
            ax.set_ylabel("Number of Customers", fontsize=9, color=CHART_TEXT)
            ax.legend(["No Churn", "Churned"], fontsize=8, framealpha=0.12, labelcolor=CHART_TEXT)
            plt.xticks(rotation=18, ha="right", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Tenure Distribution by Churn")
            fig, ax = plt.subplots(figsize=(4.1, 2.9))
            fig.patch.set_alpha(0.0)
            style_axis(ax)
            for label, color in [("No", CHART_COLORS[0]), ("Yes", CHART_COLORS[1])]:
                ax.hist(
                    df[df["Churn"] == label]["tenure"],
                    bins=28,
                    alpha=0.68,
                    color=color,
                    label=f"Churn: {label}",
                    edgecolor="#00000066",
                )
            ax.set_xlabel("Tenure (months)", fontsize=9, color=CHART_TEXT)
            ax.set_ylabel("Count", fontsize=9, color=CHART_TEXT)
            ax.legend(fontsize=8, framealpha=0.12, labelcolor=CHART_TEXT)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c4:
            st.subheader("Monthly Charges by Churn")
            fig, ax = plt.subplots(figsize=(4.1, 2.9))
            fig.patch.set_alpha(0.0)
            style_axis(ax)
            df.boxplot(
                column="MonthlyCharges",
                by="Churn",
                ax=ax,
                patch_artist=True,
                boxprops=dict(facecolor=CHART_COLORS[0], color="#e2e8f0", alpha=0.62),
                medianprops=dict(color=CHART_COLORS[1], linewidth=2),
                whiskerprops=dict(color="#e2e8f0"),
                capprops=dict(color="#e2e8f0"),
                flierprops=dict(markeredgecolor="#e2e8f0")
            )
            ax.set_title("")
            ax.set_xlabel("Churn", fontsize=9, color=CHART_TEXT)
            ax.set_ylabel("Monthly Charges ($)", fontsize=9, color=CHART_TEXT)
            plt.suptitle("")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.subheader("Internet Service Type vs Churn")
        fig, ax = plt.subplots(figsize=(6.3, 2.9))
        fig.patch.set_alpha(0.0)
        style_axis(ax)
        internet_churn = (
            df.groupby(["InternetService", "Churn"]).size().unstack(fill_value=0)
        )
        internet_churn.plot(
            kind="bar", ax=ax, color=CHART_COLORS, edgecolor=CHART_EDGE, width=0.65
        )
        ax.set_xlabel("Internet Service", fontsize=9, color=CHART_TEXT)
        ax.set_ylabel("Number of Customers", fontsize=9, color=CHART_TEXT)
        ax.legend(["No Churn", "Churned"], fontsize=8, framealpha=0.12, labelcolor=CHART_TEXT)
        plt.xticks(rotation=0, fontsize=9, color=CHART_TEXT)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.subheader("Sample Data (first 100 rows)")
    st.dataframe(df.head(100), width="stretch")


with tab2:
    st.subheader("Predict Customer Churn")
    st.caption("Complete the profile fields and run inference to estimate churn probability.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### Account Info")
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=120, value=12
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=65.0,
                step=0.5,
            )
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=780.0,
                step=1.0,
            )

        with col2:
            st.markdown("##### Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

            st.markdown("##### Phone")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )

        with col3:
            st.markdown("##### Internet Services")
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"]
            )

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        with st.status("Analyzing customer profile...", expanded=True) as status:
            st.write("Extracting demographic and service data...")
            time.sleep(0.5)
            st.write("Running data through preprocessing pipeline...")
            time.sleep(0.5)
            st.write("Executing predictive model...")
            
            input_data = {
                "gender": gender,
                "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }

            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)

            prediction = pipeline.predict(input_df)[0]
            proba = pipeline.predict_proba(input_df)[0]
            churn_prob = proba[1] * 100
            stay_prob = proba[0] * 100
            # Save to session state for the agent tab
            st.session_state.customer_data = input_data
            st.session_state.churn_prob = churn_prob
            st.session_state.agent_result = None # Reset agent result for new prediction
            
            time.sleep(0.5)
            status.update(label="Analysis complete!", state="complete", expanded=False)
            
        st.toast('Prediction generated successfully')

        st.divider()
        st.subheader("Prediction Result")

        r1, r2 = st.columns([1, 2])

        with r1:
            if prediction == 1:
                st.error("High Risk of Churn")
                st.write(f"Probability: {churn_prob:.1f}%")
            else:
                st.success("Likely to Stay")
                st.write(f"Probability: {stay_prob:.1f}%")

        with r2:
            fig, ax = plt.subplots(figsize=(4.8, 1.8))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("#0f172a")
            bars = ax.barh(
                ["Will Stay", "Will Churn"],
                [stay_prob, churn_prob],
                color=["#38bdf8", "#f97316"],
                edgecolor="#ffffff33",
                height=0.5,
            )
            ax.set_xlim(0, 105)
            ax.set_xlabel("Probability (%)", fontsize=9, color="#e2e8f0")
            ax.tick_params(colors="#e2e8f0", which='both', labelsize=9)
            ax.grid(True, linestyle='--', alpha=0.25, axis='x', color="#334155")
            for bar, val in zip(bars, [stay_prob, churn_prob]):
                ax.text(
                    val + 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="#e2e8f0"
                )
            ax.spines[["top", "right"]].set_visible(False)
            for spine in ax.spines.values():
                spine.set_color("#ffffff1a")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()
        st.subheader("Customer Insights & Recommendations")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("##### Key Risk Factors")
            factors = []
            if contract == "Month-to-month":
                factors.append("⚠️ **Month-to-month contract** users have the highest churn rate.")
            if internet_service == "Fiber optic":
                factors.append("⚠️ **Fiber optic** service has higher than average churn.")
            if tech_support == "No":
                factors.append("⚠️ **Lack of Tech Support** is a strong predictor of churn.")
            if tenure < 12:
                factors.append("⚠️ **Low Tenure** (< 1 year) represents a critical risk period.")
                
            if not factors:
                st.success("✅ Customer profile does not exhibit common churn risk factors.")
            else:
                for f in factors:
                    st.markdown(f)
                    
        with c2:
            st.markdown("##### Recommended Actions")
            if prediction == 1:
                st.markdown("- **Offer a Discount**: Provide a 10-20% discount to switch to a 1-year contract.")
                if tech_support == "No":
                    st.markdown("- **Value Add**: Offer 3 months of free Premium Tech Support.")
                if internet_service == "Fiber optic":
                    st.markdown("- **Service Check**: Trigger a proactive customer service call to ensure fiber optic network stability.")
            else:
                st.markdown("- **Upsell Opportunity**: Customer is stable. Consider offering hardware upgrades or additional streaming services.")
                st.markdown("- **Loyalty Reward**: Send a thank-you email offering a referral bonus.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Customer vs Average Metrics")
        fig, ax = plt.subplots(figsize=(6.2, 2.6))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("#0f172a")
        
        metrics = ["Tenure (Months)", "Monthly Charges ($)"]
        customer_vals = [tenure, monthly_charges]
        avg_vals = [df["tenure"].mean(), df["MonthlyCharges"].mean()]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, customer_vals, width, label='This Customer', color='#f97316' if prediction == 1 else '#38bdf8', edgecolor='#ffffff33')
        rects2 = ax.bar(x + width/2, avg_vals, width, label='Overall Average', color='#1e293b', edgecolor='#ffffff33')
        
        ax.set_ylabel('Value', color="#e2e8f0", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, color="#e2e8f0", fontsize=9)
        ax.tick_params(colors="#e2e8f0", which='both')
        ax.legend(framealpha=0.12, labelcolor="#e2e8f0", fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.25, axis='y', color="#334155")
        
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("#ffffff1a")
            
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.subheader("Export Results")
        
        # Prepare data for download
        export_data = input_data.copy()
        export_data["Prediction_Churn"] = "Yes" if prediction == 1 else "No"
        export_data["Churn_Probability"] = f"{churn_prob:.1f}%"
        export_data["Stay_Probability"] = f"{stay_prob:.1f}%"
        
        export_df = pd.DataFrame([export_data])
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Prediction as CSV",
            data=csv_data,
            file_name="churn_prediction_result.csv",
            mime="text/csv",
            use_container_width=True
        )

with tab3:
    st.subheader("AI-Driven Retention Strategy")
    
    if st.session_state.customer_data is None:
        st.info("Please run a prediction in the **Predict Churn** tab first to enable the AI Strategist.", icon="ℹ️")
    else:
        st.markdown("""
        This AI Agent uses **LangGraph** to process customer data, query a **RAG-based** knowledge base 
        of retention strategies, and generate a personalized intervention plan using **Gemini Flash**.
        """)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("### Target Customer Profile")
            st.json(st.session_state.customer_data)
        with c2:
            st.write("### Risk Context")
            st.metric("Churn Probability", f"{st.session_state.churn_prob:.1f}%")
            if st.session_state.churn_prob > 50:
                st.error("Priority: **CRITICAL** - High Churn Risk")
            else:
                st.warning("Priority: **MEDIUM** - Maintenance Mode")
        
        st.divider()
        
        if st.button("Start Expert AI Analysis", use_container_width=True):
            with st.status("Senior Agent thinking...", expanded=True) as status:
                try:
                    progress_bar = st.progress(0)
                    eta_text = st.empty()
                    estimated_total_seconds = 14
                    start_time = time.time()

                    progress_bar.progress(8)
                    eta_text.caption(f"Estimated time remaining: ~{estimated_total_seconds}s")

                    # Run the agent workflow
                    result = run_retention_agent(
                        st.session_state.customer_data, 
                        st.session_state.churn_prob
                    )

                    progress_bar.progress(65)
                    elapsed = time.time() - start_time
                    remaining = max(0, int(estimated_total_seconds - elapsed))
                    eta_text.caption(f"Estimated time remaining: ~{remaining}s")
                    
                    # Log the thought process
                    thought_log = result.get('thought_log', [])
                    total_logs = max(1, len(thought_log))
                    for i, log_entry in enumerate(thought_log, start=1):
                        st.write(log_entry)
                        time.sleep(0.3)
                        progress = min(95, 65 + int((i / total_logs) * 30))
                        progress_bar.progress(progress)
                        elapsed = time.time() - start_time
                        remaining = max(0, int(estimated_total_seconds - elapsed))
                        eta_text.caption(f"Estimated time remaining: ~{remaining}s")
                    
                    st.session_state.agent_result = result['final_report']
                    st.session_state.active_provider = result.get('active_provider', 'Unknown')
                    progress_bar.progress(100)
                    total_elapsed = time.time() - start_time
                    eta_text.caption(f"Completed in {total_elapsed:.1f}s")
                    
                    if st.session_state.active_provider == "Heuristic Mode":
                        status.update(label="Strategy Generated (Safe Mode Fallback)", state="complete", expanded=False)
                        st.toast("AI quota reached. Using expert heuristics fallback.")
                    else:
                        status.update(label=f"Strategy Formulated via {st.session_state.active_provider}!", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"Agent Workflow Error: {str(e)}")
                    st.info("Tip: This typically occurs when API keys are invalid or network access is unavailable.")
                    status.update(label="Operation Failed", state="error")

        if st.session_state.agent_result:
            if st.session_state.active_provider != "Heuristic Mode":
                st.success(f"Analysis Complete! (Generated via **{st.session_state.active_provider}**)")
            else:
                st.info("Expert Strategy Ready (Heuristic Fallback)")
                
            st.markdown(st.session_state.agent_result)
            
            # Export Agent Report
            report_text = f"TELCO CHURN STRATEGY REPORT\n\n{st.session_state.agent_result}"
            st.download_button(
                label="Download Full Retention Report",
                data=report_text,
                file_name="retention_strategy_report.md",
                mime="text/markdown",
                use_container_width=True
            )
