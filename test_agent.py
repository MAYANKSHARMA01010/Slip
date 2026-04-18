from agent.agent_engine import run_retention_agent

customer_data = {
    "tenure": 5,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "TechSupport": "No",
    "MonthlyCharges": 90.0
}
churn_prob = 85.5

print("Running Agent Test...")
result = run_retention_agent(customer_data, churn_prob)

print("\n--- Thought Log ---")
for log in result['thought_log']:
    print(log)

print("\n--- Final Report ---")
print(f"Active Provider: {result.get('active_provider')}")
print(result['final_report'])
