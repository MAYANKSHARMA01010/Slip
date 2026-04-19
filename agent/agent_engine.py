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
    user_query: str  # Added for Milestone 2 alignment
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
                # Initialize LLM based on provider class
                if provider["name"] == "Google Gemini":
                    llm = provider["class"](model=provider["model"], google_api_key=api_key)
                else:
                    llm = provider["class"](model=provider["model"], api_key=api_key)
                
                log.append(f"🚀 Using {provider['name']} for generation...")
                response = llm.invoke(prompt)
                
                # Extract content
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
    
    # Extract content and metadata (including sources for alignment)
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
    
    YOUR OUTPUT MUST INCLUDE THE FOLLOWING SECTIONS IN ORDER:
    1. Executive Risk Summary (Professional tone).
    2. Actionable Intervention Plan (Bullet points, tailored to analysis).
    3. Draft Retention Email (Personalized, empathetic).
    4. Sources & References (List the sources cited in the playbook excerpts).
    5. Business & Ethical Disclaimer (Mandatory: explain that AI-generated suggestions should be reviewed by a human professional).

    IMPORTANT EMAIL REQUIREMENTS:
    - Use the exact customer name and company name provided above.
    - Include the customer email in a short line like "To: ..." before the subject.
    - Do not use placeholders such as [Customer Name], [Company Name], etc.
    """
    
    log = state.get('thought_log', [])
    log.append("📝 Synthesizing insights and drafting personalized intervention plan...")
    
    final_report, provider = get_llm_response(prompt, log, state)
    
    if not final_report:
        # --- Heuristic Fallback Content ---
        heuristic_report = f"""
**1. Executive Risk Summary**
The customer is at risk due to their **{state['customer_data']['Contract']}** contract and **{state['customer_data']['InternetService']}** service. With a churn probability of **{state['churn_probability']:.1f}%**, immediate intervention is recommended to secure long-term value.

**2. Actionable Intervention Plan**
*   **Contract Migration**: Propose a targeted discount (10-20%) to move the customer from month-to-month to a secured 1-year or 2-year plan.
*   **Service Optimization**: Since the customer uses **{state['customer_data']['InternetService']}**, ensure they are aware of all features and provide a complimentary service check.
*   **Loyalty Engagement**: Schedule a proactive touchpoint to address any latent dissatisfaction before it leads to churn.

**3. Draft Retention Email**
To: {customer_email}
Subject: We value your partnership - let's find your perfect plan

Dear {customer_name},

As a valued member of {company_name}, we noticed you've been with us for {state['customer_data']['tenure']} months, and we want to ensure you're getting the best experience possible. We've reviewed your current service profile and would like to offer you an exclusive loyalty discount on our annual secured plans.

This upgrade would provide you with both price stability and enhanced support features. Are you available for a brief call this week to discuss how we can better serve you?

Best regards,
Retention Team
{company_name}

**4. Sources & References**
*   *Retention Playbook v2.1*: Strategy for high-risk contract migration segments.
*   *Internal Knowledge Base*: Multi-channel engagement protocols for Fiber Optic customers.

**5. Business & Ethical Disclaimer**
*Disclaimer: This retention plan is generated by an AI assistant for guidance purposes. All suggestions and email templates should be reviewed by a qualified customer success professional to ensure accuracy and compliance with company policy.*
"""
        final_report = f"> 💡 **Safe Mode Active**: All AI providers reached quota limits. Using expert heuristics.\n\n" + heuristic_report
        log.append("💡 Switched to Heuristic Mode for final report.")

    if final_report:
        final_report = (
            final_report
            .replace("[Customer Name]", customer_name)
            .replace("[Company Name]", company_name)
            .replace("[Your Name]", "Retention Team")
            .replace("[Link: Secure My Savings]", "Please reply to this email to activate your offer.")
        )
    
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
def run_retention_agent(customer_data: dict, churn_prob: float, user_query: str = ""):
    agent = create_agent()
    initial_state = {
        "customer_data": customer_data,
        "churn_probability": churn_prob,
        "user_query": user_query,
        "thought_log": []
    }
    return agent.invoke(initial_state)
