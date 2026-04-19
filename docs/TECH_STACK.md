# Technology Stack & Libraries

**Slip** is built on a modern Python stack designed for high-performance machine learning and agentic AI. Here is a breakdown of our core technologies, why we chose them, and their role in the platform.

---

## UI & Application Framework
**[Streamlit](https://streamlit.io/)**
- **Role**: Powers the `app.py` dashboard, including the "Overview," "Churn Prediction," and "AI Strategist" phases.
- **Why**: It allows us to build a premium, interactive web interface entirely in Python, strictly focusing on data logic and UX without the overhead of a separate frontend stack.

---

## Machine Learning Core
**[Scikit-Learn](https://scikit-learn.org/)**
- **Role**: Manages the data preprocessing (`StandardScaler`, `OneHotEncoder`) and our primary **Random Forest** classification pipeline.
- **Why**: Industry-standard reliability and a mature ecosystem for model evaluation and validation.

**[Imbalanced-Learn (imblearn)](https://imbalanced-learn.org/)**
- **Role**: Addresses the distribution skew in the Telco dataset using **SMOTE** (Synthetic Minority Over-sampling Technique).
- **Why**: Ensures the model learns the true patterns of churning customers rather than simply optimizing for the majority class.

---

## Agentic AI Layer (Milestone 2)
**[LangGraph](https://langchain-ai.github.io/langgraph/)**
- **Role**: Orchestrates the "Agentic Strategist" workflow. It manages the stateful transitions between analysis, retrieval, and generation.
- **Why**: Provides the flexibility of a cyclic graph, allowing for sophisticated multi-step reasoning that traditional linear chains cannot achieve.

**[FAISS](https://github.com/facebookresearch/faiss)**
- **Role**: A high-performance vector database that stores our retention playbooks.
- **Why**: Enables instantaneous similarity searches, allowing the agent to retrieve the most relevant business logic for any customer profile.

---

## Data & Serialization
**[Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)**
- **Role**: The backbone of our data management. Pandas handles the tabular CSV records, while NumPy performs the efficient matrix operations required for inference.

**[Joblib](https://joblib.readthedocs.io/)**
- **Role**: Serializes our trained `Pipeline` and model artifacts into `.pkl` files.
- **Why**: Ensures the dashboard loads instantly by utilizing pre-trained intelligence instead of re-training on every boot.

---

## Final Project Structure
```text
.
├── agent/                   # Agentic AI logic (LangGraph nodes & RAG utils)
├── docs/                    # Human-centered project documentation
│   ├── PRESENTATION_GUIDE.md # Script & mapping for the Milestone 2 video
│   ├── SETUP_GUIDE.md        # How to clone and run Slip locally
│   ├── agent_workflow.md     # Detailed guide to the LangGraph DAG
│   └── architecture_diagram.md # High-level system blueprint
├── knowledge_base/          # Markdown retention playbooks for RAG
├── vectorstore/             # Pre-built FAISS vector index
├── app.py                   # Main Streamlit dashboard source code
├── churn.ipynb              # Training notebook & exploratory analysis
├── model_pipeline.pkl       # Serialized ML intelligence
├── requirements.txt         # Project dependencies
└── telco_customer_churn.csv # Core customer dataset (7,000+ records)
```
