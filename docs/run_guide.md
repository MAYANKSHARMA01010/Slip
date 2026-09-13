# How to Run Slip Churn Intelligence Platform

This guide explains how to start and run the complete Slip Intelligence suite (FastAPI Backend + Next.js Frontend + Streamlit Dashboard) on your local machine using the unified development master runner.

---

## 🏗️ Architecture Overview

```text
       Web Browser / Dashboard Client
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
 Next.js Frontend (3000)   Streamlit App (8501)
         │ [REST / JSON]
         ▼
 FastAPI Backend (7860)
         │
 ┌───────┴───────────────────────────────┐
 │ • Scikit-Learn Model Pipeline (.pkl)  │
 │ • LangGraph Agent Engine (Graph DAG)   │
 │ • FAISS RAG Knowledge Base            │
 └───────────────────────────────────────┘
```

- **Frontend**: Next.js 16 + React 19 + Tailwind CSS 4 + Lucide Icons + Recharts (Port 3000).
- **Backend API**: FastAPI + Uvicorn + Pydantic + Scikit-Learn + LangGraph (Port 7860).
- **Streamlit App**: Standalone interactive Python analytical dashboard (Port 8501).
- **Orchestration**: All managed seamlessly with `./scripts/dev.sh`.

---

## ⚡ Daily Development Workflow

Run the master launcher script from the root directory:

```bash
./scripts/dev.sh
```
*(or run `pnpm dev`)*

### What `./scripts/dev.sh` does automatically:
1. 🧹 **Port Cleaner**: Checks and terminates any stale processes occupying ports (`3000`, `7860`, `8501`).
2. 🐍 **Python Environment**: Verifies and prepares `.venv` in the project root, automatically syncing dependencies from both root `requirements.txt` and `backend/requirements.txt`.
3. ⚙️ **Config & Data Validation**: Ensures `.env` files are in place and verifies `telco_customer_churn.csv` dataset is present.
4. 📦 **Frontend Packages**: Installs and verifies Next.js dependencies using `pnpm install` in `frontend/`.
5. 🚀 **Concurrent Launcher**: Starts the **FastAPI Backend (7860)** and **Next.js Frontend (3000)** simultaneously with live reloading and clean process shutdown on `Ctrl+C`.

---

## 🎛️ Command-Line Options

The runner supports multiple convenient flags:

```bash
# Run both FastAPI Backend & Next.js Frontend (default)
./scripts/dev.sh

# Run all 3 services: Backend (7860), Frontend (3000), and Streamlit (8501)
./scripts/dev.sh --all
# or -a

# Run ONLY the FastAPI Backend (useful for API testing)
./scripts/dev.sh --backend
# or -b

# Run ONLY the Next.js Frontend
./scripts/dev.sh --frontend
# or -f

# Run ONLY the Streamlit Dashboard
./scripts/dev.sh --streamlit
# or -s

# Launch services in separate native macOS Terminal windows
./scripts/dev.sh --terms
# or -t

# Display help & options
./scripts/dev.sh --help
```

---

## 🌐 How to Access Your Services

Once started, access the platform via your browser:

| Service | URL | Purpose |
| :--- | :--- | :--- |
| **Next.js Frontend** | `http://localhost:3000` | Modern React UI with analytics & strategist |
| **FastAPI Backend API** | `http://localhost:7860` | REST API for predictions and LangGraph agent |
| **FastAPI Swagger Docs** | `http://localhost:7860/docs` | Interactive API documentation |
| **Streamlit Dashboard** | `http://localhost:8501` | Traditional Streamlit analytics dashboard |

---

## 🛠️ Port Verification & Troubleshooting

If you ever need to manually inspect or free the ports:

```bash
# Check port status
lsof -i :3000
lsof -i :7860
lsof -i :8501

# Free ports manually
kill -9 $(lsof -ti:3000)
kill -9 $(lsof -ti:7860)
kill -9 $(lsof -ti:8501)
```
