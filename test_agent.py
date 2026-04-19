from agent.agent_engine import process_customer_retention

customer_data = {
    "tenure": 5,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "TechSupport": "No",
    "MonthlyCharges": 90.0,
    "CustomerName": "Test Customer",
    "CustomerEmail": "test@example.com",
    "CompanyName": "Slip Telco"
}
churn_prob = 85.5
user_query = "Focus on fiber optic reliability."

print("Running Agent Test...")
result = process_customer_retention(customer_data, churn_prob, user_query)

print("\n--- Thought Log ---")
for log in result['thought_log']:
    print(log)

print("\n--- Final Report ---")
print(f"Active Provider: {result.get('active_provider')}")
print(result['final_report'])
