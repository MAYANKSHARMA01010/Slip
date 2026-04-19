import sys
import os

# Add the current directory to sys.path to simulate app.py environment
sys.path.append(os.getcwd())

try:
    from agent.agent_engine import process_customer_retention
    import inspect
    
    sig = inspect.signature(process_customer_retention)
    print(f"Function signature: {sig}")
    
    if 'user_query' in sig.parameters:
        print("SUCCESS: 'user_query' found in signature.")
    else:
        print("FAILURE: 'user_query' NOT found in signature.")
        
except Exception as e:
    print(f"ERROR: {str(e)}")
