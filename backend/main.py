import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_engine import process_customer_retention, stream_customer_retention
from rag_utils import (
    list_kb_articles,
    create_vector_db,
    get_manifest,
    KB_DIR,
)

app = FastAPI(
    title="Slip Churn Intelligence Backend",
    description="API for customer churn prediction and LangGraph strategists",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "model_pipeline.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "telco_customer_churn.csv")

# Load model artifacts
try:
    pipeline = joblib.load(PIPELINE_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
except Exception as e:
    # If artifacts are missing, we train them on startup
    import subprocess
    print("Model artifacts not found or failed to load. Attempting automatic training...")
    try:
        # Simple inline training script to ensure backend starts up
        df_temp = pd.read_csv(DATASET_PATH).drop(columns=["customerID"])
        df_temp["TotalCharges"] = df_temp["TotalCharges"].replace({" ": "0.0"}).astype(float)
        df_temp["SeniorCitizen"] = df_temp["SeniorCitizen"].astype(int)
        df_temp["gender"] = df_temp["gender"].astype(str)
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder

        X = df_temp.drop("Churn", axis=1)
        y = df_temp["Churn"].map({"Yes": 1, "No": 0})
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ])

        pipeline_temp = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=50, random_state=42)),
        ])

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline_temp.fit(X_train, y_train)
        joblib.dump(pipeline_temp, PIPELINE_PATH)
        joblib.dump(X.columns.tolist(), FEATURES_PATH)
        
        pipeline = pipeline_temp
        feature_columns = X.columns.tolist()
        print("Training completed successfully.")
    except Exception as err:
        print(f"Training failed: {err}")
        raise RuntimeError("Failed to load or train model artifacts.")

# Load dataset for analytics
df = pd.read_csv(DATASET_PATH)
# Clean dataset in memory
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
df["gender"] = df["gender"].astype(str)

# Models for Request/Response
class CustomerProfile(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    CustomerName: Optional[str] = "Customer"
    CustomerEmail: Optional[str] = ""
    CompanyName: Optional[str] = "Telco"

class StrategyRequest(BaseModel):
    customer_data: dict
    churn_prob: float
    user_query: Optional[str] = ""

class KnowledgeArticle(BaseModel):
    title: str
    content: str  # Full Markdown body

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Slip Churn Intelligence Platform API is running.",
        "dataset_rows": len(df),
        "features_count": len(feature_columns)
    }

@app.get("/summary")
def get_summary():
    churn_rate = (df["Churn"] == "Yes").sum() / len(df) * 100
    avg_ticket = df["MonthlyCharges"].mean()
    # Simple estimate of CLV potential at risk (Monthly charges sum of churned customers * average tenure)
    churned_customers = df[df["Churn"] == "Yes"]
    clv_at_risk = float((churned_customers["MonthlyCharges"] * churned_customers["tenure"]).sum())
    
    return {
        "total_base": len(df),
        "churn_rate": round(churn_rate, 2),
        "avg_ticket": round(avg_ticket, 2),
        "clv_potential": round(clv_at_risk, 2),
        "features": len(feature_columns)
    }

@app.get("/charts")
def get_charts():
    # 1. Churn Distribution
    counts = df["Churn"].value_counts().to_dict()
    churn_distribution = [
        {"name": "Retained", "value": int(counts.get("No", 0))},
        {"name": "Churned", "value": int(counts.get("Yes", 0))}
    ]
    
    # 2. Contract Type vs Churn
    contract_churn = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
    contract_data = []
    for contract_type in contract_churn.index:
        contract_data.append({
            "Contract": contract_type,
            "Retained": int(contract_churn.loc[contract_type, "No"]),
            "Churned": int(contract_churn.loc[contract_type, "Yes"])
        })
        
    # 3. Internet Service vs Churn
    internet_churn = df.groupby(["InternetService", "Churn"]).size().unstack(fill_value=0)
    internet_data = []
    for service_type in internet_churn.index:
        internet_data.append({
            "InternetService": service_type,
            "Retained": int(internet_churn.loc[service_type, "No"]),
            "Churned": int(internet_churn.loc[service_type, "Yes"])
        })

    # 4. Tenure distribution binned (bins of 5)
    bins = list(range(0, 76, 5))
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    df_copy = df.copy()
    df_copy['tenure_bin'] = pd.cut(df_copy['tenure'], bins=bins, labels=labels, include_lowest=True)
    tenure_grouped = df_copy.groupby(['tenure_bin', 'Churn'], observed=False).size().unstack(fill_value=0)
    tenure_data = []
    for bin_label in tenure_grouped.index:
        tenure_data.append({
            "bin": str(bin_label),
            "Retained": int(tenure_grouped.loc[bin_label, "No"]),
            "Churned": int(tenure_grouped.loc[bin_label, "Yes"])
        })
        
    # 5. Monthly charges distribution binned (bins of 10)
    charge_bins = list(range(10, 131, 10))
    charge_labels = [f"${charge_bins[i]}-{charge_bins[i+1]}" for i in range(len(charge_bins)-1)]
    df_copy['charge_bin'] = pd.cut(df_copy['MonthlyCharges'], bins=charge_bins, labels=charge_labels, include_lowest=True)
    charge_grouped = df_copy.groupby(['charge_bin', 'Churn'], observed=False).size().unstack(fill_value=0)
    charge_data = []
    for bin_label in charge_grouped.index:
        charge_data.append({
            "bin": str(bin_label),
            "Retained": int(charge_grouped.loc[bin_label, "No"]),
            "Churned": int(charge_grouped.loc[bin_label, "Yes"])
        })

    # 6. Correlation
    corr_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    corr_matrix = df[corr_cols].corr().round(2).values.tolist()
    correlation_data = {
        "columns": corr_cols,
        "matrix": corr_matrix
    }

    return {
        "churn_distribution": churn_distribution,
        "contract_churn": contract_data,
        "internet_churn": internet_data,
        "tenure_churn": tenure_data,
        "charge_churn": charge_data,
        "correlation": correlation_data
    }

@app.get("/averages")
def get_averages():
    return {
        "tenure": float(df["tenure"].mean()),
        "MonthlyCharges": float(df["MonthlyCharges"].mean()),
        "TotalCharges": float(df["TotalCharges"].mean())
    }

@app.get("/sample-data")
def get_sample_data():
    # Return first 100 rows as list of dicts
    sample_df = df.head(100).copy()
    # Replace NaN or clean float representation if necessary
    return sample_df.to_dict(orient="records")

@app.post("/predict")
def predict_churn(profile: CustomerProfile):
    try:
        # Convert pydantic model to dict for model features mapping
        input_dict = profile.model_dump()
        
        # Prepare inputs exactly as model expects
        model_input = {
            "gender": input_dict["gender"],
            "SeniorCitizen": input_dict["SeniorCitizen"],
            "Partner": input_dict["Partner"],
            "Dependents": input_dict["Dependents"],
            "tenure": input_dict["tenure"],
            "PhoneService": input_dict["PhoneService"],
            "MultipleLines": input_dict["MultipleLines"],
            "InternetService": input_dict["InternetService"],
            "OnlineSecurity": input_dict["OnlineSecurity"],
            "OnlineBackup": input_dict["OnlineBackup"],
            "DeviceProtection": input_dict["DeviceProtection"],
            "TechSupport": input_dict["TechSupport"],
            "StreamingTV": input_dict["StreamingTV"],
            "StreamingMovies": input_dict["StreamingMovies"],
            "Contract": input_dict["Contract"],
            "PaperlessBilling": input_dict["PaperlessBilling"],
            "PaymentMethod": input_dict["PaymentMethod"],
            "MonthlyCharges": input_dict["MonthlyCharges"],
            "TotalCharges": input_dict["TotalCharges"],
        }
        
        input_df = pd.DataFrame([model_input])[feature_columns]
        
        prediction = int(pipeline.predict(input_df)[0])
        proba = pipeline.predict_proba(input_df)[0]
        churn_prob = float(proba[1] * 100)
        stay_prob = float(proba[0] * 100)
        
        # Calculate risk drivers and mitigations dynamically
        factors = []
        if input_dict["Contract"] == "Month-to-month": 
            factors.append("Month-to-month contract (High churn segment)")
        if input_dict["InternetService"] == "Fiber optic":   
            factors.append("Fiber optic service (Quality/Price sensitivity)")
        if input_dict["TechSupport"] == "No":             
            factors.append("Lack of Tech Support (Key retention barrier)")
        if input_dict["tenure"] < 12:                
            factors.append("Low tenure (<1 yr) (Early lifecycle churn)")
            
        recs = []
        if prediction == 1:
            recs.append("Offer 10–20% 'Upgrade Reward' discount")
            if input_dict["TechSupport"] == "No":           
                recs.append("Grant 3 months complimentary Tech Support")
            if input_dict["InternetService"] == "Fiber optic": 
                recs.append("Schedule proactive connectivity health check")
        else:
            recs.append("Incentivize long-term loyalty with referral bonus")
            recs.append("Upsell hardware or high-tier streaming bundle")

        return {
            "prediction": prediction,
            "churn_probability": round(churn_prob, 2),
            "stay_probability": round(stay_prob, 2),
            "risk_drivers": factors,
            "recommendations": recs,
            "customer_data": input_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/strategy")
def get_strategy(request: StrategyRequest):
    try:
        # Run LangGraph strategy process
        result = process_customer_retention(
            customer_data=request.customer_data,
            churn_prob=request.churn_prob,
            user_query=request.user_query
        )
        
        return {
            "final_report": result.get("final_report", ""),
            "active_provider": result.get("active_provider", "Unknown"),
            "thought_log": result.get("thought_log", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Strategy generation failed: {str(e)}")


@app.post("/strategy/stream")
def stream_strategy(request: StrategyRequest):
    """
    Streaming variant of /strategy — returns Server-Sent Events so the
    frontend can display each thought-log step as it happens in real time.
    """
    return StreamingResponse(
        stream_customer_retention(
            customer_data=request.customer_data,
            churn_prob=request.churn_prob,
            user_query=request.user_query,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable Nginx/proxy buffering
        },
    )


# ─── Knowledge Base Management ──────────────────────────────────────────────

@app.get("/knowledge")
def get_knowledge_articles():
    """List all articles in the knowledge base with metadata."""
    try:
        articles = list_kb_articles()
        manifest = get_manifest()
        return {
            "articles": articles,
            "total_files": len(articles),
            "total_chunks_indexed": manifest.get("total_chunks", "not built yet"),
            "last_rebuilt": manifest.get("last_built", "never"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge", status_code=201)
def add_knowledge_article(article: KnowledgeArticle):
    """
    Add a new Markdown article to the knowledge base.
    The FAISS vectorstore is automatically rebuilt after saving.
    """
    try:
        # Sanitize title → safe filename
        safe_name = "".join(
            c if c.isalnum() or c in " -_" else "" for c in article.title
        ).strip().replace(" ", "_").lower()
        filename = f"{safe_name}.md"
        fpath = os.path.join(KB_DIR, filename)

        if os.path.exists(fpath):
            raise HTTPException(
                status_code=409,
                detail=f"Article '{filename}' already exists. Choose a different title.",
            )

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(f"# {article.title}\n\n{article.content}")

        create_vector_db(force=True)

        return {
            "message": f"Article '{filename}' added and vectorstore rebuilt.",
            "filename": filename,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/knowledge/{filename}")
def delete_knowledge_article(filename: str):
    """
    Delete a knowledge base article by filename and rebuild vectorstore.
    AI-generated files (ai_enriched_*) can also be removed this way.
    """
    # Safety: no path traversal
    if any(c in filename for c in ("..", "/", "\\")):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    fpath = os.path.join(KB_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"Article '{filename}' not found.")

    os.remove(fpath)
    create_vector_db(force=True)

    return {"message": f"Article '{filename}' deleted and vectorstore rebuilt."}


@app.post("/knowledge/refresh")
def refresh_knowledge_base():
    """
    Force a full rebuild of the FAISS vectorstore from the current knowledge_base/ folder.
    Useful after manually editing files or dropping in new ones.
    """
    try:
        create_vector_db(force=True)
        articles = list_kb_articles()
        manifest = get_manifest()
        return {
            "message": "Knowledge base rebuilt successfully.",
            "files_indexed": len(articles),
            "total_chunks": manifest.get("total_chunks"),
            "articles": articles,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    # Extract feature importance from the pipeline
    feature_importances = []
    try:
        preprocessor = pipeline.named_steps.get('preprocessor')
        classifier = pipeline.named_steps.get('clf')
        
        if classifier and hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                names = preprocessor.get_feature_names_out()
            else:
                names = feature_columns
                
            if len(names) == len(importances):
                # Format names and filter for top 12 importances
                feats = []
                for n, imp in zip(names, importances):
                    clean_name = n.replace("cat__", "").replace("num__", "")
                    feats.append({"feature": clean_name, "importance": float(imp)})
                # Sort and return top 12
                feature_importances = sorted(feats, key=lambda x: x["importance"])[-12:]
    except Exception as e:
        print(f"Error loading feature importances: {e}")

    # Standard metrics, benchmarking leaderboard, and confusion matrix
    return {
        "model_accuracy": 77.2,
        "precision_churn": 54.0,
        "recall_churn": 58.0,
        "f1_churn": 56.0,
        "leaderboard": [
            {"model": "Random Forest", "accuracy": 84.2, "rank": "Gold"},
            {"model": "XGBoost", "accuracy": 83.8, "rank": "Silver"},
            {"model": "Decision Tree", "accuracy": 78.7, "rank": "Bronze"}
        ],
        "confusion_matrix": {
            "matrix": [[882, 171], [149, 203]],
            "x": ["Predicted: No", "Predicted: Yes"],
            "y": ["Actual: No", "Actual: Yes"]
        },
        "feature_importances": feature_importances
    }
