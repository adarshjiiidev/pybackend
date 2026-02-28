try:
    from app.graph import agent_graph
    print("Successfully imported agent_graph")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
