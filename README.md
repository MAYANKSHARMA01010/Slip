# Customer Churn Prediction & Agentic Retention Strategy

## From Predictive Analytics to Intelligent Intervention

### Project Overview

Customer churn is one of the biggest challenges telecom companies face today. Losing a customer isn't just about one cancelled subscription; it snowballs into lost revenue, higher acquisition costs, and a weakened brand. This project tackles that problem head on by building an AI powered system that not only predicts which customers are likely to leave, but eventually evolves into an intelligent agent that can suggest personalized retention strategies.

We worked with the **Telco Customer Churn** dataset, which captures real world customer behavior like how long they've been with the company, what services they use, how much they pay, and whether or not they ended up churning. The goal was to dig into this data, find meaningful patterns, and train a model that can flag at risk customers before it's too late.

**Milestone 1** focuses on classical machine learning. We built a full preprocessing and modeling pipeline using Scikit Learn, tackled class imbalance with **SMOTE**, and evaluated multiple classifiers (Decision Tree, Random Forest, XGBoost) via cross-validation before selecting **Random Forest** as the final model. The results are served through a polished, interactive Streamlit dashboard with a glassmorphism dark theme where users can explore the data visually, predict churn for individual customers, see personalized risk factors and retention recommendations, and even export prediction results as CSV.

**Milestone 2** takes things further by introducing an agent based AI layer. The idea here is to move beyond just predicting churn and actually reason about it. Using frameworks like LangGraph and retrieval augmented generation (RAG), the system will pull in retention best practices and generate structured intervention plans tailored to each customer's situation.

---

### Milsestones & Documentation

#### Milestone 1: ML-Based Churn Prediction (Mid-Sem)
- **Problem Context**: [Business Context & Problem Understanding](docs/business_context.md)
- **Model Training**: [Model Performance Evaluation Report](docs/model_report.md)
- **UI Dashboard**: Interactive Streamlit application with real-time inference.

#### Milestone 2: Agentic AI Retention Strategy (End-Sem)
- **Workflow & State**: [Agent Workflow Documentation](docs/agent_workflow.md)
- **System Design**: [Architecture Diagram](docs/architecture_diagram.md)
- **RAG Implementation**: Knowledge retrieval using FAISS and LangGraph.
- **Interactive Strategist**: Support for custom user queries and structured retention plans.

---

### 🚀 Quick Start

To run the platform locally:

```bash
# 1. Setup virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Intelligence Center
streamlit run app.py
```

### 📖 Full Project Documentation
- **[Business Context & Problem Statement](docs/business_context.md)**
- **[Model Performance & Selection Report](docs/model_report.md)**
- **[Agent Architecture & Workflow Guide](docs/agent_workflow.md)**
- **[System Architecture Diagram](docs/architecture_diagram.md)**
- **[Detailed Tech Stack & Libraries](docs/TECH_STACK.md)**
- **[Setup & Installation Guide](docs/SETUP_GUIDE.md)**

---

### Team & Development
Meet the core team and see how to get involved by visiting our **[Contributors Guide](docs/CONTRIBUTORS.md)**.
