"""
Terminal Chat Client for Daaddys AI
Allows interactive chat with the AI backend via command line.
"""

import httpx
import asyncio
import json
import sys
from typing import Optional

# ANSI color codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TerminalChatClient:
    """Interactive terminal chat client."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id: Optional[str] = None
    
    async def create_session(self):
        """Create a new chat session."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(f"{self.base_url}/chat/session")
                data = response.json()
                self.session_id = data["session_id"]
                print(f"{Colors.GREEN}[OK] Session created: {self.session_id[:8]}...{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}[ERROR] Failed to create session: {e}{Colors.ENDC}")
                self.session_id = None
    
    async def chat_stream(self, query: str, mode: str = "auto"):
        """Send a chat message and stream the response."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/stream",
                    json={
                        "query": query,
                        "mode": mode,
                        "session_id": self.session_id
                    }
                ) as response:
                    print(f"\n{Colors.CYAN}[AI]:{Colors.ENDC} ", end="", flush=True)
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            try:
                                data = json.loads(data_str)
                                if "content" in data:
                                    print(data["content"], end="", flush=True)
                                if data.get("done"):
                                    if "metadata" in data:
                                        metadata = data["metadata"]
                                        print(f"\n\n{Colors.YELLOW}[Agent: {metadata.get('agent', 'unknown')}]{Colors.ENDC}")
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    print("\n")  # New line after response
                    
            except httpx.ConnectError:
                print(f"{Colors.RED}[ERROR] Connection failed. Is the server running on {self.base_url}?{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}[ERROR]: {e}{Colors.ENDC}")
    
    async def start(self):
        """Start the interactive chat session."""
        # Print header (ASCII-safe)
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}    Daaddys AI - Financial Intelligence Terminal{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
        
        # Check server health
        print(f"{Colors.CYAN}Connecting to server...{Colors.ENDC}")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                health = response.json()
                print(f"{Colors.GREEN}[OK] Server is healthy{Colors.ENDC}")
                print(f"{Colors.YELLOW}  Database: {health.get('database', 'unknown')}{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.RED}[ERROR] Server health check failed: {e}{Colors.ENDC}")
                print(f"{Colors.YELLOW}Make sure the server is running: python -m uvicorn app.main:app --reload{Colors.ENDC}")
                return
        
        # Create session
        await self.create_session()
        
        # Print usage instructions
        print(f"\n{Colors.BOLD}Commands:{Colors.ENDC}")
        print(f"  {Colors.GREEN}/mode [auto|market|realtime|portfolio|explainer|crypto]{Colors.ENDC} - Change agent mode")
        print(f"  {Colors.GREEN}/clear{Colors.ENDC} - Create new session")
        print(f"  {Colors.GREEN}/quit{Colors.ENDC} or {Colors.GREEN}/exit{Colors.ENDC} - Exit")
        print(f"\n{Colors.BOLD}Examples:{Colors.ENDC}")
        print(f"  {Colors.CYAN}Analyze Reliance stock{Colors.ENDC}")
        print(f"  {Colors.CYAN}What's the latest news on Nifty 50?{Colors.ENDC}")
        print(f"  {Colors.CYAN}How to build a balanced portfolio?{Colors.ENDC}")
        print(f"  {Colors.CYAN}Explain PE ratio in simple terms{Colors.ENDC}\n")
        
        mode = "auto"
        
        # Main chat loop
        while True:
            try:
                # Get user input
                user_input = input(f"{Colors.BOLD}{Colors.BLUE}You:{Colors.ENDC} ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ["/quit", "/exit", "quit", "exit"]:
                    print(f"\n{Colors.YELLOW}Goodbye!{Colors.ENDC}\n")
                    break
                
                elif user_input.lower() == "/clear":
                    await self.create_session()
                    print(f"{Colors.GREEN}[OK] New session started!{Colors.ENDC}\n")
                    continue
                
                elif user_input.lower().startswith("/mode"):
                    parts = user_input.split()
                    if len(parts) > 1:
                        new_mode = parts[1].lower()
                        valid_modes = ["auto", "market", "realtime", "portfolio", "explainer", "crypto"]
                        if new_mode in valid_modes:
                            mode = new_mode
                            print(f"{Colors.GREEN}[OK] Mode changed to: {mode}{Colors.ENDC}\n")
                        else:
                            print(f"{Colors.RED}Invalid mode. Use: {', '.join(valid_modes)}{Colors.ENDC}\n")
                    else:
                        print(f"{Colors.YELLOW}Current mode: {mode}{Colors.ENDC}\n")
                    continue
                
                # Send chat message
                await self.chat_stream(user_input, mode)
                
            except KeyboardInterrupt:
                print(f"\n\n{Colors.YELLOW}Interrupted. Type /quit to exit.{Colors.ENDC}\n")
            except EOFError:
                print(f"\n{Colors.YELLOW}Goodbye! 👋{Colors.ENDC}\n")
                break


async def main():
    """Main entry point."""
    # Check for custom server URL
    server_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    
    client = TerminalChatClient(base_url=server_url)
    await client.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Goodbye! 👋{Colors.ENDC}\n")
