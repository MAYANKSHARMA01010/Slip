# Slip Milestone 2: Presentation & Visual Guide

This guide maps each section of your **Milestone 2 Video Script** to the specific UI pages, code snippets, and diagrams available in the **Slip** codebase. Use this to ensure your visual narrative perfectly matches your spoken script.

---

## 📽️ Section I: The Evolution (0:00 - 0:45)
**Narrative Focus**: Moving from predictive ML to Agentic Intelligence.

| Visual Component | Resource to Show | Description |
| :--- | :--- | :--- |
| **Project Identity** | [app.py](file:///Users/amansoni/Documents/Projects/Projects/Slip/app.py) | Show the **"Slip" logo** (Sli<span>p</span>) on the Home Page. |
| **Landing UI** | [app.py](file:///Users/amansoni/Documents/Projects/Projects/Slip/app.py) | The **Hero Shell** with the tagline: "Predict churn early. Retain every customer." |
| **Platform Stats** | [app.py](file:///Users/amansoni/Documents/Projects/Projects/Slip/app.py) | The metric cards showing **Knowledge Base ✅ Ready** and **Vector Index ✅ Ready**. |

---

## 🧠 Section II: The Agent Architecture (0:45 - 1:45)
**Narrative Focus**: LangGraph, State Management, and the DAG.

| Visual Component | Resource to Show | Description |
| :--- | :--- | :--- |
| **The Workflow Diagram** | [agent_workflow.md](file:///Users/amansoni/Documents/Projects/Projects/Slip/docs/agent_workflow.md) | Show the **Mermaid DAG** (analyze -> retrieve -> generate). |
| **Agent State** | [agent_engine.py](file:///Users/amansoni/Documents/Projects/Projects/Slip/agent/agent_engine.py) | The `AgentState` TypedDict showing how context is preserved. |
| **Graph Definition** | [agent_engine.py](file:///Users/amansoni/Documents/Projects/Projects/Slip/agent/agent_engine.py) | The `create_agent()` function where nodes are compiled into a workflow. |

---

## 📚 Section III: The Knowledge Engine (1:45 - 2:45)
**Narrative Focus**: RAG, FAISS, and Retention Playbooks.

| Visual Component | Resource to Show | Description |
| :--- | :--- | :--- |
| **The Playbook Content** | [retention_strategies.md](file:///Users/amansoni/Documents/Projects/Projects/Slip/knowledge_base/retention_strategies.md) | Show the raw Markdown file containing the **Contract & Billing Strategies**. |
| **Vector Store Logic** | [rag_utils.py](file:///Users/amansoni/Documents/Projects/Projects/Slip/agent/rag_utils.py) | The code initializing `HuggingFaceEmbeddings` and `FAISS.save_local`. |
| **Retrieval Node** | [agent_engine.py](file:///Users/amansoni/Documents/Projects/Projects/Slip/agent/agent_engine.py) | The `retrieve_knowledge` function that queries the vector store. |

---

## ⚡ Section IV: Live Demo: The Strategist (2:45 - 4:15)
**Narrative Focus**: Interactive reasoning and personalization.

| Visual Component | Resource to Show | Description |
| :--- | :--- | :--- |
| **The Dashboard Tab** | **UI: AI Strategist** | Toggle to the **"AI Strategist"** tab in the sidebar. |
| **Input Section** | [app.py](file:///Users/amansoni/Documents/Projects/Projects/Slip/app.py) | Show the **"Specific Retention Query"** text area for custom instructions. |
| **Thinking Process** | [app.py](file:///Users/amansoni/Documents/Projects/Projects/Slip/app.py) | The **`st_status`** block that prints the agent's thought log in real-time. |
| **Final Strategy UI** | [app.py](file:///Users/amansoni/Documents/Projects/Projects/Slip/app.py) | The Markdown strategy output and the **"Send via Email Client"** button. |

---

## 🛡️ Section V: Technical Reliability (4:15 - 4:45)
**Narrative Focus**: Fallbacks and Heuristic Mode.

| Visual Component | Resource to Show | Description |
| :--- | :--- | :--- |
| **Multi-Provider Logic** | [agent_engine.py](file:///Users/amansoni/Documents/Projects/Projects/Slip/agent/agent_engine.py) | The `get_llm_response` function that iterates through Gemini, Groq, and Mistral. |
| **Heuristic Fallback** | [agent_engine.py](file:///Users/amansoni/Documents/Projects/Projects/Slip/agent/agent_engine.py) | The hardcoded **`heuristic_report`** used when AI providers fail. |
| **Fallback UI** | [app.py](file:///Users/amansoni/Documents/Projects/Projects/Slip/app.py) | The UI message: "Strategy Generated (Safe Mode Fallback)". |

---

## 🏁 Section VI: Closing Impact (4:45 - 5:00)
**Narrative Focus**: Predictive. Agentic. Intelligent.

| Visual Component | Resource to Show | Description |
| :--- | :--- | :--- |
| **Final Summary UI** | [app.py](file:///Users/amansoni/Documents/Projects/Projects/Slip/app.py) | The **Product Flow** summary box on the Home Page. |
| **Full Dashboard** | **UI: Overview Tab** | Switch back to the **Overview** tab showing all the vibrant Plotly charts. |

---

### 💡 Quick Presentation Tips:
1.  **Cursor Movement**: Move your cursor over the code snippets as you mention them (e.g., hover over `LangGraph` in the code).
2.  **Transition**: When jumping from "Code" to "UI," use a smooth slide transition to emphasize the connection between implementation and experience.
3.  **Real Data**: In the Live Demo, type a query like: *"Customer complained about Fiber pricing, focus on a value bundle."* to show the agent actually thinking.
