import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from agent.rag_utils import get_vector_db

load_dotenv()

# --- State Definition ---
class AgentState(TypedDict):
    customer_data: dict
    churn_probability: float
    user_query: str
    retrieved_strategies: str
    analysis: str
    final_report: str
    active_provider: str
    thought_log: List[str]

# --- Nodes ---

def get_llm_response(prompt: str, log: List[str], state: AgentState):
    """
    Attempts to get a response from multiple AI providers in order of preference.
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
    
    return None, "Heuristic Mode"

def analyze_customer(state: AgentState):
    """
    Analyzes the raw customer data and churn probability to identify key risk themes.
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
    
    if not analysis:
        analysis = f"Strategic Analysis: High risk detected for {data['Contract']} customer with {data['InternetService']} service. Priorities: Contract stability and service value."
        log.append("💡 Switching to heuristic mode for this step.")
    
    return {"analysis": analysis, "active_provider": provider, "thought_log": log}

def retrieve_knowledge(state: AgentState):
    """
    Queries the vector database for relevant retention strategies.
    """
    db = get_vector_db()
    query = f"Retention strategies for {state['analysis']}"
    
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
    Synthesizes analysis and knowledge into a professional report.
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

# --- Graph Construction ---

def create_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("analyze", analyze_customer)
    workflow.add_node("retrieve", retrieve_knowledge)
    workflow.add_node("generate", generate_report)
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

# --- Execution Entry ---
def process_customer_retention(customer_data: dict, churn_prob: float, user_query: str = ""):
    """
    Final entry point name to break all cached module links.
    """
    agent = create_agent()
    initial_state = {
        "customer_data": customer_data,
        "churn_probability": churn_prob,
        "user_query": user_query,
        "thought_log": []
    }
    return agent.invoke(initial_state)
