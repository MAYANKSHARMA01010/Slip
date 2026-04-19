import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from agent.rag_utils import get_vector_db

load_dotenv()

# We define the shape of our agent's memory here. 
# This state is passed between nodes to keep the 'Chain of Thought' intact.
class AgentState(TypedDict):
    customer_data: dict          # Raw profile from the dashboard
    churn_probability: float     # Score from our ML pipeline
    user_query: str              # Personalization focus from the user
    retrieved_strategies: str    # Context pulled from the FAISS database
    analysis: str               # The agent's reasoning about risk themes
    final_report: str           # The professional output for the user
    active_provider: str         # Which AI model (Gemini/Groq/etc.) succeeded
    thought_log: List[str]       # A step-by-step log for the UI status updates

# --- Core Reasoning Nodes ---

def get_llm_response(prompt: str, log: List[str], state: AgentState):
    """
    This is our fallback-safe execution layer. 
    It cycles through multiple AI providers to ensure the agent stays online.
    """
    providers = [
        {"name": "Google Gemini", "env_key": "GOOGLE_API_KEY", "class": ChatGoogleGenerativeAI, "model": "gemini-flash-latest"},
        {"name": "Groq (Llama 3)", "env_key": "GROQ_API_KEY", "class": ChatGroq, "model": "llama-3.3-70b-versatile"},
        {"name": "Mistral AI", "env_key": "MISTRAL_API_KEY", "class": ChatMistralAI, "model": "mistral-large-latest"},
    ]
    
    for provider in providers:
        api_key = os.getenv(provider["env_key"])
        if api_key and api_key != "your_api_key_here":
            try:
                # Configuring the specific LLM client
                if provider["name"] == "Google Gemini":
                    llm = provider["class"](model=provider["model"], google_api_key=api_key)
                else:
                    llm = provider["class"](model=provider["model"], api_key=api_key)
                
                log.append(f"🚀 Using {provider['name']} for generation...")
                response = llm.invoke(prompt)
                
                if hasattr(response, "content"):
                    if isinstance(response.content, list):
                        content = "".join([b.get("text", "") for b in response.content if isinstance(b, dict)])
                    else:
                        content = str(response.content)
                    return content, provider["name"]
            except Exception as e:
                log.append(f"⚠️ {provider['name']} failed: {str(e)[:100]}...")
                continue
    
    # If all AI providers fail, we signal the switch to Heuristic Mode.
    return None, "Heuristic Mode"

def analyze_customer(state: AgentState):
    """
    Node 1: Look at the customer's data and summarize why they might leave.
    """
    data = state['customer_data']
    prob = state['churn_probability']
    
    analysis_prompt = f"""
    You are a Senior Customer Churn Analyst. 
    Analyze this customer profile:
    - Churn Probability: {prob:.1f}%
    - Tenure: {data['tenure']} months
    - Contract: {data['Contract']}
    - Internet Service: {data['InternetService']}
    - Tech Support: {data['TechSupport']}
    - Monthly Charges: ${data['MonthlyCharges']}
    
    Identify the top 2-3 specific risk factors for this customer. 
    Be concise.
    """
    
    log = state.get('thought_log', [])
    log.append("🧠 Identifying primary risk factors from customer profile...")
    
    analysis, provider = get_llm_response(analysis_prompt, log, state)
    
    # Fallback reasoning if the AI is unreachable
    if not analysis:
        analysis = f"Strategic Analysis: High risk detected for {data['Contract']} customer with {data['InternetService']} service. Priorities: Contract stability and service value."
        log.append("💡 Switching to heuristic mode for this step.")
    
    return {"analysis": analysis, "active_provider": provider, "thought_log": log}

def retrieve_knowledge(state: AgentState):
    """
    Node 2: Search our internal playbook for strategies that match the analysis.
    """
    db = get_vector_db()
    query = f"Retention strategies for {state['analysis']}"
    
    # We pull the 3 most relevant sections from our Markdown knowledge base.
    docs = db.similarity_search(query, k=3)
    
    strategies = []
    for i, doc in enumerate(docs):
        strategies.append(f"Source {i+1}: {doc.page_content}")
    
    context = "\n\n".join(strategies)
    
    log = state.get('thought_log', [])
    log.append("🔍 Searching internal knowledge base for proven retention playbooks...")
    
    return {"retrieved_strategies": context, "thought_log": log}

def generate_report(state: AgentState):
    """
    Node 3: Synthesize the analysis and knowledge into a finalize report and email draft.
    """
    customer_name = state["customer_data"].get("CustomerName", "Customer")
    customer_email = state["customer_data"].get("CustomerEmail", "")
    company_name = state["customer_data"].get("CompanyName", "Telco")

    prompt = f"""
    You are a Senior Strategic Retention Agent. 
    Based on the following analysis and expert strategies, generate a professional retention plan.

    CUSTOMER IDENTITY CONTEXT:
    - Customer Name: {customer_name}
    - Customer Email: {customer_email}
    - Company Name: {company_name}
    
    USER SPECIAL QUERY:
    {state.get('user_query', 'No specific focus requested.')}
    
    CUSTOMER ANALYSIS:
    {state['analysis']}
    
    RETENTION PLAYBOOK EXCERPTS (WITH SOURCES):
    {state['retrieved_strategies']}
    
    YOUR OUTPUT MUST INCLUDE THE FOLLOWING SECTIONS IN ORDER WITH THESE EXACT HEADERS:
    ### 1. Executive Risk Summary
    ### 2. Actionable Intervention Plan
    ### 3. Draft Retention Email (Include 'Recipient:', 'Subject:', and 'Body:' inside this section)
    ### 4. Sources & References
    ### 5. Business & Ethical Disclaimer

    IMPORTANT: Do not use placeholders like [Customer Name]. Use real names provided.
    """
    
    log = state.get('thought_log', [])
    log.append("📝 Synthesizing insights and drafting personalized intervention plan...")
    
    final_report, provider = get_llm_response(prompt, log, state)
    
    # If the generator fails, we provide a structured strategic response from expert rules.
    if not final_report:
        heuristic_report = f"""
**1. Executive Risk Summary**
High risk detected for {customer_name} due to contract type and service charges.

**2. Actionable Intervention Plan**
* Migrate to a long-term contract.
* Offer a connectivity health check.

**3. Draft Retention Email**
Sent to: {customer_email}
Subject: Enhancing your experience at {company_name}

**4. Sources & References**
Internal Playbook v2.1

**5. Business & Ethical Disclaimer**
AI-generated. Review by professional required.
"""
        final_report = f"> 💡 **Expert Heuristics**: Using strategic fallback logic.\n\n" + heuristic_report
        log.append("💡 Switched to Heuristic Mode for final report.")

    return {"final_report": final_report, "active_provider": provider, "thought_log": log}

# --- Graph Assembly ---

def create_agent():
    """
    We assemble the nodes into a workflow (DAG) using LangGraph.
    """
    workflow = StateGraph(AgentState)
    
    # Define our three-step logic
    workflow.add_node("analyze", analyze_customer)
    workflow.add_node("retrieve", retrieve_knowledge)
    workflow.add_node("generate", generate_report)
    
    # Define the sequence
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# --- Execution Entry Point ---
def process_customer_retention(customer_data: dict, churn_prob: float, user_query: str = ""):
    """
    The main entry point for the Streamlit dashboard to interact with the AI Strategist.
    """
    agent = create_agent()
    initial_state = {
        "customer_data": customer_data,
        "churn_probability": churn_prob,
        "user_query": user_query,
        "thought_log": []
    }
    return agent.invoke(initial_state)
