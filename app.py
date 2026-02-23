import os
import warnings
import joblib
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
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
ARTIFACTS = ["model.pkl", "scaler.pkl", "feature_columns.pkl", "num_cols.pkl"]


def train_and_save():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv("telco_customer_churn.csv").drop(columns=["customerID"])
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    cat_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # Save artifacts
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feature_columns, "feature_columns.pkl")
    joblib.dump(NUM_COLS, "num_cols.pkl")
    return accuracy


if not all(os.path.exists(f) for f in ARTIFACTS):
    with st.spinner("First run — training model, please wait..."):
        acc = train_and_save()
    st.success(f"Model trained! Accuracy: {acc:.2%}")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns


@st.cache_data
def load_data():
    df = pd.read_csv("telco_customer_churn.csv").drop(columns=["customerID"])
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
    return df


model, scaler, feature_columns = load_artifacts()
df = load_data()

st.title("Telco Customer Churn Dashboard")
st.caption("Explore customer data and predict churn with Logistic Regression.")
tab1, tab2 = st.tabs(["Data Overview", "Predict Churn"])


with tab1:
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

    with c1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        counts = df["Churn"].value_counts()
        ax.pie(
            counts,
            labels=["No Churn", "Churned"],
            autopct="%1.1f%%",
            colors=["#4A90D9", "#E8534A"],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.subheader("Contract Type vs Churn")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        contract_churn = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
        contract_churn.plot(
            kind="bar", ax=ax, color=["#4A90D9", "#E8534A"], edgecolor="white"
        )
        ax.set_xlabel("Contract Type", fontsize=9)
        ax.set_ylabel("Number of Customers", fontsize=9)
        ax.legend(["No Churn", "Churned"], fontsize=8)
        plt.xticks(rotation=20, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Tenure Distribution by Churn")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for label, color in [("No", "#4A90D9"), ("Yes", "#E8534A")]:
            ax.hist(
                df[df["Churn"] == label]["tenure"],
                bins=30,
                alpha=0.6,
                color=color,
                label=f"Churn: {label}",
                edgecolor="white",
            )
        ax.set_xlabel("Tenure (months)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c4:
        st.subheader("Monthly Charges by Churn")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        df.boxplot(
            column="MonthlyCharges",
            by="Churn",
            ax=ax,
            patch_artist=True,
            boxprops=dict(facecolor="#4A90D9", color="#333"),
            medianprops=dict(color="#E8534A", linewidth=2),
        )
        ax.set_title("")
        ax.set_xlabel("Churn", fontsize=9)
        ax.set_ylabel("Monthly Charges ($)", fontsize=9)
        plt.suptitle("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Internet Service Type vs Churn")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    internet_churn = (
        df.groupby(["InternetService", "Churn"]).size().unstack(fill_value=0)
    )
    internet_churn.plot(
        kind="bar", ax=ax, color=["#4A90D9", "#E8534A"], edgecolor="white"
    )
    ax.set_xlabel("Internet Service", fontsize=9)
    ax.set_ylabel("Number of Customers", fontsize=9)
    ax.legend(["No Churn", "Churned"], fontsize=8)
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

        cat_cols_input = input_df.select_dtypes(include=["object"]).columns
        input_encoded = pd.get_dummies(
            input_df, columns=cat_cols_input, drop_first=True
        )
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        input_encoded[NUM_COLS] = scaler.transform(input_encoded[NUM_COLS])
        prediction = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0]
        churn_prob = proba[1] * 100
        stay_prob = proba[0] * 100

        st.divider()
        st.subheader("Prediction Result")

        r1, r2 = st.columns([1, 2])

        with r1:
            if prediction == 1:
                st.error("**High Churn Risk**")
                st.metric("Churn Probability", f"{churn_prob:.1f}%")
                st.caption("This customer is likely to leave.")
            else:
                st.success("**Low Churn Risk**")
                st.metric("Retention Probability", f"{stay_prob:.1f}%")
                st.caption("This customer is likely to stay.")

        with r2:
            fig, ax = plt.subplots(figsize=(6, 2))
            bars = ax.barh(
                ["Will Stay", "Will Churn"],
                [stay_prob, churn_prob],
                color=["#4A90D9", "#E8534A"],
                edgecolor="white",
                height=0.5,
            )
            ax.set_xlim(0, 105)
            ax.set_xlabel("Probability (%)", fontsize=9)
            for bar, val in zip(bars, [stay_prob, churn_prob]):
                ax.text(
                    val + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
