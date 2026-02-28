import sys
import os
sys.path.append(os.getcwd())

try:
    import app.graph
    print(f"Exported names in app.graph: {app.graph.__all__}")
    if 'agent_graph' in app.graph.__all__:
        print("SUCCESS: agent_graph is exported")
    else:
        print("FAILURE: agent_graph is NOT exported")
except Exception as e:
    print(f"Error: {e}")
