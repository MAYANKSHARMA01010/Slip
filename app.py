import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

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
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.6);
        color: white;
        border-color: transparent;
    }

    /* Tabs formatting */
    button[role="tab"] {
        font-weight: 500;
        padding-bottom: 0.5rem;
    }
    
    /* Custom divider line */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
import os
import joblib

ARTIFACTS = ["model_pipeline.pkl", "feature_columns.pkl"]

if not all(os.path.exists(f) for f in ARTIFACTS):
    st.error("Model artifacts not found. Please run the notebook first to generate `model_pipeline.pkl` and `feature_columns.pkl`.", icon="⚠️")
    st.info("The dashboard requires a trained model to function. Please execute `churn.ipynb` to train the Logistic Regression model.")
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

st.title("Telco Churn")
tab1, tab2 = st.tabs(["Data Overview", "Predict Churn"])


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
        CHART_COLORS = ["#00d2ff", "#E8534A"]  # Vivid Blue and Coral Red

        with c1:
            st.subheader("Churn Distribution")
            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            counts = df["Churn"].value_counts()
            ax.pie(
                counts,
                labels=["No Churn", "Churned"],
                autopct="%1.1f%%",
                colors=CHART_COLORS,
                startangle=90,
                wedgeprops={"edgecolor": "#ffffff33", "linewidth": 1.5},
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            st.subheader("Contract Type vs Churn")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            contract_churn = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
            contract_churn.plot(
                kind="bar", ax=ax, color=CHART_COLORS, edgecolor="#ffffff33"
            )
            ax.set_xlabel("Contract Type", fontsize=9)
            ax.set_ylabel("Number of Customers", fontsize=9)
            ax.legend(["No Churn", "Churned"], fontsize=8, framealpha=0.2)
            plt.xticks(rotation=20, ha="right", fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.2, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Tenure Distribution by Churn")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            for label, color in [("No", CHART_COLORS[0]), ("Yes", CHART_COLORS[1])]:
                ax.hist(
                    df[df["Churn"] == label]["tenure"],
                    bins=30,
                    alpha=0.7,
                    color=color,
                    label=f"Churn: {label}",
                    edgecolor="#00000080",
                )
            ax.set_xlabel("Tenure (months)", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.legend(fontsize=8, framealpha=0.2)
            ax.grid(True, linestyle='--', alpha=0.2, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c4:
            st.subheader("Monthly Charges by Churn")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            df.boxplot(
                column="MonthlyCharges",
                by="Churn",
                ax=ax,
                patch_artist=True,
                boxprops=dict(facecolor=CHART_COLORS[0], color="#fff", alpha=0.7),
                medianprops=dict(color=CHART_COLORS[1], linewidth=2),
                whiskerprops=dict(color="#fff"),
                capprops=dict(color="#fff"),
                flierprops=dict(markeredgecolor="#fff")
            )
            ax.set_title("")
            ax.set_xlabel("Churn", fontsize=9)
            ax.set_ylabel("Monthly Charges ($)", fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.2)
            plt.suptitle("")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.subheader("Internet Service Type vs Churn")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        internet_churn = (
            df.groupby(["InternetService", "Churn"]).size().unstack(fill_value=0)
        )
        internet_churn.plot(
            kind="bar", ax=ax, color=CHART_COLORS, edgecolor="#ffffff33"
        )
        ax.set_xlabel("Internet Service", fontsize=9)
        ax.set_ylabel("Number of Customers", fontsize=9)
        ax.legend(["No Churn", "Churned"], fontsize=8, framealpha=0.2)
        ax.grid(True, linestyle='--', alpha=0.2, axis='y')
        plt.xticks(rotation=0, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.subheader("Sample Data (first 100 rows)")
    st.dataframe(df.head(100), width="stretch")


with tab2:
    st.subheader("Predict Customer Churn")
    st.markdown("Fill in the customer details below and click **Predict**.")

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
        import time
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
            time.sleep(0.5)
            status.update(label="Analysis complete!", state="complete", expanded=False)
            
        st.toast('Prediction generated successfully!', icon='✅')

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
            fig, ax = plt.subplots(figsize=(6, 2))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            bars = ax.barh(
                ["Will Stay", "Will Churn"],
                [stay_prob, churn_prob],
                color=["#00d2ff", "#E8534A"],
                edgecolor="#ffffff33",
                height=0.5,
            )
            ax.set_xlim(0, 105)
            ax.set_xlabel("Probability (%)", fontsize=9, color="#e2e8f0")
            ax.tick_params(colors="#e2e8f0", which='both')
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
