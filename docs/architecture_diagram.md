# Slip — Architecture & Agent Workflow

The **Slip Intelligence Platform** follows a progressive AI architecture, evolving from a classical machine learning pipeline into a multi-agentic reasoning system.

```mermaid
graph TD
    A[Customer Data] --> B[Streamlit Dashboard]
    B --> C{Action}
    
    subgraph "Milestone 1: Predictive Core"
        C -->|Inference| D[Random Forest Pipeline]
        D --> E[Churn Probability %]
        D --> F[Feature Importance]
    end
    
    subgraph "Milestone 2: Agentic Layer"
        E --> G[LangGraph Agent]
        H[(FAISS Vector DB)] -->|Retrieval| G
        G --> I[LLM Analysis & Reasoning]
        I --> J[Structured Retention Report]
    end
    
    J --> K[Retention Email & Plan]
    K --> B
```

### Technical components:
- **UI Framework**: Streamlit
- **ML Core**: Scikit-learn (Random Forest Pipeline)
- **Agentic Framework**: LangGraph (Stateful Workflows)
- **Vector DB**: FAISS (Local storage of strategy playbooks)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Reasoning**: Google Gemini Flash (with Groq/Mistral fallbacks)
