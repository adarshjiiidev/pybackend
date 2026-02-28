import time
import asyncio
import logging

# Disable logging to focus on timing
logging.basicConfig(level=logging.ERROR)

def benchmark():
    from app.graph.workflow import create_agent_graph
    print("Benchmarking create_agent_graph()...")
    start = time.time()
    for i in range(5):
        create_agent_graph()
    end = time.time()
    print(f"Average time to create_agent_graph: {(end - start) / 5:.4f}s")

if __name__ == "__main__":
    benchmark()
