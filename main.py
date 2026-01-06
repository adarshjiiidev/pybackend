import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent.graph import build_graph
from agent.utils import setup_logger

# Load environment variables
load_dotenv(override=True)

logger = setup_logger()

def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return

    print("Initializing Financial Agent...")
    try:
        graph = build_graph()
        print("Agent ready. Type 'quit' to exit.")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            print("\nProcessing...")
            
            # Initial state
            initial_state = {"messages": [HumanMessage(content=user_input)]}
            
            # Run the graph
            result = graph.invoke(initial_state)
            
            # Extract response
            final_response = result.get("final_response")
            error = result.get("error")
            
            if error:
                print(f"\n[!] Error: {error}")
            elif final_response:
                print(f"\nAgent:\n{final_response}")
            else:
                print("\n[?] No response generated.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[!] Unexpected error: {e}")

if __name__ == "__main__":
    main()
