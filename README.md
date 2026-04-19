# Slip — Churn Intelligence Platform

## From Predictive Analytics to Strategic Intervention

**Slip** is a production-grade churn intelligence platform. It bridges the gap between historical data analysis and real-world customer retention by combining classical Machine Learning with modern Agentic AI reasoning.

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

### Team & Development
Built by the **Slip** core team. Visit our **[Contributors Guide](docs/CONTRIBUTORS.md)** to see the faces behind the code.
