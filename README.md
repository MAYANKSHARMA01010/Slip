---
title: Slip - Churn Intelligence Platform
emoji: 📉
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Slip — Churn Intelligence Platform

## From Predictive Analytics to Strategic Intervention

[![Live Demo](https://img.shields.io/badge/Hugging%20Face-Space-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/Manku69/slip-churn-intelligence)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://slip-ai.streamlit.app/)

**Slip** is a production-grade churn intelligence platform. It bridges the gap between historical data analysis and real-world customer retention by combining classical Machine Learning with modern Agentic AI reasoning.

### 🌐 Live Experience
You can access the hosted version of the platform directly:
- **Hugging Face (Primary)**: [Slip Churn Command Center](https://huggingface.co/spaces/Manku69/slip-churn-intelligence)
- **Streamlit Cloud (Mirror)**: [slip-ai.streamlit.app](https://slip-ai.streamlit.app/)

### Project Vision
Customer churn is more than just a lost subscription; it’s a disruption in growth. This project tackles the churn problem in two distinct phases:
1.  **Phase 1 (Predictive)**: We built a robust Random Forest pipeline to identify *at-risk* customers with 84% accuracy.
2.  **Phase 2 (Agentic)**: We developed a **"Strategist" Agent** using LangGraph and RAG to reason through a customer's specific pain points and generate professional, data-backed retention plans.

---

### Core Intelligence Phases

#### Phase 1: Machine Learning Core
- **Analytical Verdicts**: Real-time inference on customer profiles.
- **Model Transparency**: Feature importance and performance metrics built-in.
- **Data Balance**: Utilizes SMOTE to ensure minority churn signals are never ignored.

#### Phase 2: Agentic Retention Layer
- **Cognitive Flow**: Built on **LangGraph** for multi-step "Chain of Thought" reasoning.
- **Dynamic Retrieval**: Queries an internal knowledge base of playbooks via **FAISS**.
- **Interactive Persona**: Accepts custom user instructions to refine the strategic focus.

---

### 🚀 Launching the Platform

Get **Slip** running on your local machine in seconds:

```bash
# 1. Setup your environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Synchronize dependencies
pip install -r requirements.txt

# 3. Launch the Intelligence Center
streamlit run app.py
```

### 📖 Exploratory Documentation
- **[System Blueprints](docs/architecture_diagram.md)**: Explore the architectural flow from data to strategy.
- **[Agent Workflow](docs/agent_workflow.md)**: A deep dive into the LangGraph DAG logic.
- **[Technology Stack](docs/TECH_STACK.md)**: Detailed breakdown of our weapons of choice.
- **[Installation Guide](docs/SETUP_GUIDE.md)**: Step-by-step setup and prerequisites.
- **[Presentation Script](docs/PRESENTATION_GUIDE.md)**: A visual guide for the Milestone 2 presentation.

---

### ☁️ Hugging Face Deployment & Usage

Slip is optimized for deployment on Hugging Face Spaces using Docker.

#### Cloning from Hugging Face
To clone the source code directly from the Hugging Face Space:
```bash
# Note: You will need Git LFS installed to pull the model artifacts
git clone https://huggingface.co/spaces/Manku69/slip-churn-intelligence
cd slip-churn-intelligence
git lfs pull
```

#### Deployment Notes
- **SDK**: Docker (Debian Slim)
- **Large Files**: Managed via **Git LFS** (tracking `.pkl` and `.csv` files).
- **Environment Secrets**: Requires `GOOGLE_API_KEY`, `GROQ_API_KEY`, and `MISTRAL_API_KEY` to be set in the Space settings.

### Team & Development
Built by the **Slip** core team. Visit our **[Contributors Guide](docs/CONTRIBUTORS.md)** to see the faces behind the code.
