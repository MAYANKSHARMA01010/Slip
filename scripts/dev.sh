#!/usr/bin/env bash

# ==============================================================================
# Slip Intelligence Platform - Local Development & Services Master Runner
# 1. Frees occupied host ports (3000, 7860, 8501)
# 2. Prepares root Python Virtual Environment (.venv) & installs requirements
# 3. Synchronizes backend requirements (FastAPI, uvicorn, LangGraph, etc.)
# 4. Validates environment configurations & model data
# 5. Installs frontend dependencies via pnpm
# 6. Starts Backend, Frontend, and optional Streamlit services
# ==============================================================================

set -e

# ANSI Color Codes
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
MAGENTA="\033[0;35m"
RED="\033[0;31m"
NC="\033[0m" # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Auto-source NVM if available and Node version < 20
export NVM_DIR="$HOME/.nvm"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    . "$NVM_DIR/nvm.sh" >/dev/null 2>&1 || true
    CURRENT_NODE_MAJOR=$(node -v 2>/dev/null | cut -d'.' -f1 | sed 's/v//')
    if [ -n "$CURRENT_NODE_MAJOR" ] && [ "$CURRENT_NODE_MAJOR" -lt 20 ]; then
        nvm use 22 >/dev/null 2>&1 || nvm use 20 >/dev/null 2>&1 || nvm use default >/dev/null 2>&1 || true
    fi
fi

USE_TERMS=false
ONLY_BACKEND=false
ONLY_FRONTEND=false
ONLY_STREAMLIT=false
RUN_ALL=false

# Parse command line flags
for arg in "$@"; do
    case $arg in
        --terms|-t)
            USE_TERMS=true
            ;;
        --backend|-b)
            ONLY_BACKEND=true
            ;;
        --frontend|-f)
            ONLY_FRONTEND=true
            ;;
        --streamlit|-s)
            ONLY_STREAMLIT=true
            ;;
        --all|-a)
            RUN_ALL=true
            ;;
        --help|-h)
            echo -e "${BOLD}${CYAN}Slip Dev Master Script Usage:${NC}"
            echo -e "  ${GREEN}./scripts/dev.sh${NC}             Run FastAPI Backend (7860) + Next.js Frontend (3000)"
            echo -e "  ${GREEN}./scripts/dev.sh --all${NC}       Run Backend (7860) + Frontend (3000) + Streamlit (8501)"
            echo -e "  ${GREEN}./scripts/dev.sh --backend${NC}   Run ONLY FastAPI Backend (7860)"
            echo -e "  ${GREEN}./scripts/dev.sh --frontend${NC}  Run ONLY Next.js Frontend (3000)"
            echo -e "  ${GREEN}./scripts/dev.sh --streamlit${NC} Run ONLY Streamlit App (8501)"
            echo -e "  ${GREEN}./scripts/dev.sh --terms${NC}     Launch services in separate macOS Terminal windows"
            exit 0
            ;;
    esac
done

echo -e "${BOLD}${CYAN}======================================================================${NC}"
echo -e "${BOLD}${CYAN}          🚀 Slip Intelligence Platform - Local Development           ${NC}"
echo -e "${BOLD}${CYAN}======================================================================${NC}"

# ------------------------------------------------------------------------------
# 0. Port Management: Check and Release Occupied Ports
# ------------------------------------------------------------------------------
PORTS=()
if [ "$ONLY_BACKEND" = true ]; then
    PORTS=(7860)
elif [ "$ONLY_FRONTEND" = true ]; then
    PORTS=(3000)
elif [ "$ONLY_STREAMLIT" = true ]; then
    PORTS=(8501)
elif [ "$RUN_ALL" = true ]; then
    PORTS=(3000 7860 8501)
else
    # Default: Frontend (3000) + Backend (7860)
    PORTS=(3000 7860)
fi

echo -e "${YELLOW}🧹 [0/4] Checking and freeing ports: ${PORTS[*]}...${NC}"
for PORT in "${PORTS[@]}"; do
    PIDS=$(lsof -ti:$PORT 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo -e "  ${YELLOW}⚠️  Port $PORT is occupied by PID(s): $PIDS. Releasing...${NC}"
        kill -9 $PIDS 2>/dev/null || true
        sleep 0.5
        echo -e "  ${GREEN}✓ Port $PORT released successfully.${NC}"
    else
        echo -e "  ${GREEN}✓ Port $PORT is free.${NC}"
    fi
done

# ------------------------------------------------------------------------------
# 1. Python Environment Setup (.venv in root)
# ------------------------------------------------------------------------------
if [ "$ONLY_FRONTEND" = false ]; then
    echo ""
    echo -e "${BLUE}🐍 [1/4] Preparing Python Virtual Environment (.venv in root)...${NC}"
    cd "$ROOT_DIR"

    # Check for stale or broken .venv (e.g. if python version changed or moved)
    if [ -d ".venv" ]; then
        if ! .venv/bin/python -c "import sys" 2>/dev/null; then
            echo -e "  ${YELLOW}⚠️  Stale or broken .venv detected. Re-creating fresh .venv...${NC}"
            rm -rf .venv
        fi
    fi

    if [ ! -d ".venv" ]; then
        echo -e "  ${CYAN}Creating Python virtual environment (.venv)...${NC}"
        python3 -m venv .venv
    fi

    source "$ROOT_DIR/.venv/bin/activate"

    # Install Root Requirements
    if [ -f "$ROOT_DIR/requirements.txt" ]; then
        echo -e "  ${CYAN}Checking & syncing root Python dependencies...${NC}"
        pip install -q -r "$ROOT_DIR/requirements.txt"
    fi

    # Install Backend Requirements (FastAPI, uvicorn, pydantic, etc.)
    if [ -f "$ROOT_DIR/backend/requirements.txt" ]; then
        echo -e "  ${CYAN}Checking & syncing backend dependencies into root .venv...${NC}"
        pip install -q -r "$ROOT_DIR/backend/requirements.txt"
    fi
    echo -e "  ${GREEN}✓ Python virtual environment ready: $(python -V)${NC}"

    # --------------------------------------------------------------------------
    # 2. Environment Variables & Model Artifacts Validation
    # --------------------------------------------------------------------------
    echo ""
    echo -e "${BLUE}⚙️  [2/4] Validating configurations & datasets...${NC}"

    # Ensure backend .env is present
    if [ ! -f "$ROOT_DIR/backend/.env" ] && [ -f "$ROOT_DIR/backend/.env.example" ]; then
        echo -e "  ${YELLOW}⚠️  backend/.env missing. Creating from backend/.env.example...${NC}"
        cp "$ROOT_DIR/backend/.env.example" "$ROOT_DIR/backend/.env"
    fi

    # Ensure frontend .env is present
    if [ ! -f "$ROOT_DIR/frontend/.env" ] && [ -f "$ROOT_DIR/frontend/.env.example" ]; then
        echo -e "  ${CYAN}Creating frontend/.env from frontend/.env.example...${NC}"
        cp "$ROOT_DIR/frontend/.env.example" "$ROOT_DIR/frontend/.env"
    fi

    # Check Git LFS on CSV file
    if [ -f "$ROOT_DIR/telco_customer_churn.csv" ]; then
        if grep -q "git-lfs" "$ROOT_DIR/telco_customer_churn.csv" 2>/dev/null; then
            echo -e "  ${YELLOW}⚠️  Root dataset is a Git LFS pointer. Pulling data...${NC}"
            git lfs pull --include="telco_customer_churn.csv" || true
        fi
    fi

    # Sync CSV to backend if missing or pointer
    if [ ! -f "$ROOT_DIR/backend/telco_customer_churn.csv" ] || grep -q "git-lfs" "$ROOT_DIR/backend/telco_customer_churn.csv" 2>/dev/null; then
        echo -e "  ${CYAN}Syncing dataset to backend/telco_customer_churn.csv...${NC}"
        cp "$ROOT_DIR/telco_customer_churn.csv" "$ROOT_DIR/backend/telco_customer_churn.csv" 2>/dev/null || true
    fi

    echo -e "  ${GREEN}✓ Configurations and datasets verified.${NC}"
fi

# ------------------------------------------------------------------------------
# 3. Frontend Dependencies Setup (pnpm)
# ------------------------------------------------------------------------------
if [ "$ONLY_BACKEND" = false ] && [ "$ONLY_STREAMLIT" = false ]; then
    echo ""
    echo -e "${BLUE}📦 [3/4] Checking Frontend Dependencies (Next.js via pnpm)...${NC}"
    cd "$ROOT_DIR/frontend"

    if command -v pnpm &> /dev/null; then
        echo -e "  ${CYAN}Running pnpm install in frontend/...${NC}"
        pnpm install --dangerously-allow-all-builds || pnpm install
    else
        echo -e "  ${YELLOW}⚠️  pnpm not found globally. Falling back to npm install...${NC}"
        npm install
    fi
    echo -e "  ${GREEN}✓ Frontend packages installed and ready.${NC}"
fi

# ------------------------------------------------------------------------------
# 4. Service Launch Execution
# ------------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${MAGENTA}======================================================================${NC}"
echo -e "${BOLD}${MAGENTA}                 🎉 Slip Intelligence Platform Ready                  ${NC}"
echo -e "${BOLD}${MAGENTA}======================================================================${NC}"
if [ "$ONLY_FRONTEND" = true ]; then
    echo -e "  🛒 ${BOLD}Frontend Web App:${NC}       ${GREEN}http://localhost:3000${NC}"
elif [ "$ONLY_BACKEND" = true ]; then
    echo -e "  ⚙️  ${BOLD}FastAPI Backend API:${NC}    ${GREEN}http://localhost:7860${NC} (Docs: ${CYAN}http://localhost:7860/docs${NC})"
elif [ "$ONLY_STREAMLIT" = true ]; then
    echo -e "  📊 ${BOLD}Streamlit Dashboard:${NC}    ${GREEN}http://localhost:8501${NC}"
elif [ "$RUN_ALL" = true ]; then
    echo -e "  🛒 ${BOLD}Frontend Web App:${NC}       ${GREEN}http://localhost:3000${NC}"
    echo -e "  ⚙️  ${BOLD}FastAPI Backend API:${NC}    ${GREEN}http://localhost:7860${NC} (Docs: ${CYAN}http://localhost:7860/docs${NC})"
    echo -e "  📊 ${BOLD}Streamlit Dashboard:${NC}    ${GREEN}http://localhost:8501${NC}"
else
    echo -e "  🛒 ${BOLD}Frontend Web App:${NC}       ${GREEN}http://localhost:3000${NC}"
    echo -e "  ⚙️  ${BOLD}FastAPI Backend API:${NC}    ${GREEN}http://localhost:7860${NC} (Docs: ${CYAN}http://localhost:7860/docs${NC})"
    echo -e "  💡 ${YELLOW}Tip: Run with --all to also launch the Streamlit dashboard on :8501${NC}"
fi
echo -e "${BOLD}${MAGENTA}======================================================================${NC}"
echo ""

# Execution Modes:
# A) Single Service Modes
if [ "$ONLY_BACKEND" = true ]; then
    echo -e "${GREEN}✨ Launching Backend ONLY on http://localhost:7860 ...${NC}"
    cd "$ROOT_DIR/backend"
    source "$ROOT_DIR/.venv/bin/activate"
    exec uvicorn main:app --reload --host 0.0.0.0 --port 7860
fi

if [ "$ONLY_FRONTEND" = true ]; then
    echo -e "${GREEN}✨ Launching Frontend ONLY on http://localhost:3000 ...${NC}"
    cd "$ROOT_DIR/frontend"
    if command -v pnpm &> /dev/null; then
        exec pnpm dev
    else
        exec npm run dev
    fi
fi

if [ "$ONLY_STREAMLIT" = true ]; then
    echo -e "${GREEN}✨ Launching Streamlit App ONLY on http://localhost:8501 ...${NC}"
    cd "$ROOT_DIR"
    source "$ROOT_DIR/.venv/bin/activate"
    exec streamlit run app.py --server.port 8501
fi

# B) macOS Separate Terminal Windows Mode (--terms)
if [ "$USE_TERMS" = true ]; then
    echo -e "${CYAN}🖥️  Launching services in separate macOS Terminal windows...${NC}"
    osascript -e 'tell application "Terminal" to do script "cd \"'$ROOT_DIR'/backend\" && source \"'$ROOT_DIR'/.venv/bin/activate\" && uvicorn main:app --reload --host 0.0.0.0 --port 7860"' >/dev/null
    sleep 2
    if command -v pnpm &> /dev/null; then
        osascript -e 'tell application "Terminal" to do script "cd \"'$ROOT_DIR'/frontend\" && pnpm dev"' >/dev/null
    else
        osascript -e 'tell application "Terminal" to do script "cd \"'$ROOT_DIR'/frontend\" && npm run dev"' >/dev/null
    fi
    if [ "$RUN_ALL" = true ]; then
        sleep 1
        osascript -e 'tell application "Terminal" to do script "cd \"'$ROOT_DIR'\" && source \"'$ROOT_DIR'/.venv/bin/activate\" && streamlit run app.py --server.port 8501"' >/dev/null
    fi
    echo -e "${GREEN}✓ Services launched in separate terminal windows.${NC}"
    exit 0
fi

# C) Unified Concurrently / Trap Runner
cd "$ROOT_DIR"

# Check if concurrently is available or run via pnpm dlx / npx
if [ "$RUN_ALL" = true ]; then
    echo -e "${CYAN}✨ Launching Backend (7860), Frontend (3000), & Streamlit (8501) using concurrently...${NC}"
    if command -v pnpm &> /dev/null; then
        pnpm dlx concurrently@8 \
          --names "BACKEND,FRONTEND,STREAMLIT" \
          --prefix-colors "cyan,magenta,yellow" \
          --kill-others-on-fail \
          "cd backend && source ../.venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 7860" \
          "sleep 1 && cd frontend && pnpm dev" \
          "sleep 2 && source .venv/bin/activate && streamlit run app.py --server.port 8501"
    else
        npx -y concurrently@8 \
          --names "BACKEND,FRONTEND,STREAMLIT" \
          --prefix-colors "cyan,magenta,yellow" \
          --kill-others-on-fail \
          "cd backend && source ../.venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 7860" \
          "sleep 1 && cd frontend && npm run dev" \
          "sleep 2 && source .venv/bin/activate && streamlit run app.py --server.port 8501"
    fi
else
    echo -e "${CYAN}✨ Launching Backend (7860) & Frontend (3000) using concurrently...${NC}"
    if command -v pnpm &> /dev/null; then
        pnpm dlx concurrently@8 \
          --names "BACKEND,FRONTEND" \
          --prefix-colors "cyan,magenta" \
          --kill-others-on-fail \
          "cd backend && source ../.venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 7860" \
          "sleep 1 && cd frontend && pnpm dev"
    else
        npx -y concurrently@8 \
          --names "BACKEND,FRONTEND" \
          --prefix-colors "cyan,magenta" \
          --kill-others-on-fail \
          "cd backend && source ../.venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 7860" \
          "sleep 1 && cd frontend && npm run dev"
    fi
fi
