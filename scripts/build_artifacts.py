import sys
import os

# Ensure repo root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build():
    print("Building model artifacts from telco_customer_churn.csv...")
    csv_path = os.path.join(ROOT_DIR, "telco_customer_churn.csv")
    df = pd.read_csv(csv_path).drop(columns=["customerID"])
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["gender"] = df["gender"].astype(str)

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=50, random_state=42)),
    ])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, os.path.join(ROOT_DIR, "model_pipeline.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(ROOT_DIR, "feature_columns.pkl"))
    print("✅ Model artifacts (model_pipeline.pkl, feature_columns.pkl) created successfully!")

    print("Building FAISS vector database from knowledge_base/...")
    try:
        from agent.rag_utils import create_vector_db
        create_vector_db()
        print("✅ Vectorstore (vectorstore/db_faiss) created successfully!")
    except Exception as e:
        print(f"❌ Vectorstore creation failed: {e}")
        raise e

if __name__ == "__main__":
    build()
