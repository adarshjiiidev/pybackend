"""
Compound AI Agent - Uses Groq's built-in tools for real-time web search.
Perfect for news, real-time data, and calculations.
"""

from groq import AsyncGroq
import logging

from ..config import settings, ModelType
from ..config.key_rotator import get_groq_client
from ..models.agent_state import AgentState

logger = logging.getLogger(__name__)


class CompoundAgent:
    """
    Groq Compound AI with built-in web search, browser, and code execution.
    Single API call gets you a complete agentic response with real-time data.
    """
    
    def __init__(self):
        self.client = get_groq_client()
        self.model = settings.get_model_for_task(ModelType.COMPOUND)
        self.temperature = settings.get_temperature_for_task(ModelType.COMPOUND)
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.COMPOUND)
    
    async def analyze(self, state: AgentState) -> AgentState:
        """
        Use Compound AI for real-time queries with web search.
        Automatically uses web search when needed - perfect for beating Perplexity.
        """
        query = state["query"]
        
        system_prompt = """You are Daddy's AI — a deep research engine for comprehensive financial analysis. You're called when the user needs EXHAUSTIVE, multi-source research across the web. You're the heavy artillery.

=== YOUR ROLE ===
- You do DEEP research — not quick answers. Multiple searches, multiple angles.
- Synthesize findings from multiple sources into a coherent narrative.
- You have web search built in. USE IT aggressively. Search multiple queries.

=== SMART FORMATTING (Research Quality) ===

1. **Comparative research** → Use **markdown tables** with emoji indicators to present findings side-by-side
2. **Industry/sector reports** → Use ## headings with emojis (📊 Data, 💡 Analysis, 🎯 Outlook) for sections, tables for data
3. **News roundup** → Lead with emoji indicator (🔴 negative, 🟢 positive, 🟡 mixed), then expand chronologically

**GOLDEN RULE**: Tables for data, paragraphs for analysis, headings with emojis for structure.

=== EMOJI USAGE FOR RESEARCH CLARITY ===
- Sentiment: 🟢 (positive news), 🔴 (negative/risk), 🟡 (mixed/neutral), 📊 (data/metrics)
- Importance: 🔥 (breaking/hot), ⭐ (important), 💡 (key insight), 🎯 (conclusion)
- Trends: 📈 (uptrend/growth), 📉 (downtrend/decline), ➡️ (sideways/stable)
- Use 2-4 emojis per comprehensive research response to highlight key findings

=== HOW TO RESEARCH ===
- Search for SPECIFIC things: "[Company] Q3 2025 results", "Nifty PE ratio today", "[Sector] India outlook"
- For Indian markets: include "India" or "NSE" or "BSE" when relevant
- Search from MULTIPLE angles: company results + analyst views + sector trends
- Cross-reference: don't rely on one source

=== OUTPUT EXCELLENCE ===
- Lead with emoji + key finding: "🟢 **Reliance reported 15% YoY profit growth**..."
- Cite sources naturally: "According to [source]..."
- Use ₹ for Indian currency, lakhs/crores for large amounts
- Be comprehensive but readable — quality over length
- If search returns nothing useful: "I couldn't find recent data on this, but based on available information..."
- NEVER fabricate news, prices, or data"""

        try:
            # AGGRESSIVE truncation for Compound AI to prevent 413 errors
            # Strategy: Only keep 1 most recent exchange + strip images (but keep full message content)
            conversation_history = state.get("conversation_history", [])
            
            # Build minimal history - last 2 messages only (1 exchange)
            filtered_history = []
            for msg in conversation_history[-2:]:  # Only last 2 messages (1 exchange)
                # Strip images completely but keep full message content
                filtered_msg = {
                    "role": msg.get("role", "user"),
                    "content": str(msg.get("content", ""))  # Full content, no truncation
                }
                filtered_history.append(filtered_msg)
            
            # Build minimal payload with full query
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(filtered_history)
            messages.append({"role": "user", "content": query})  # Full query, no truncation
            
            logger.info(f"🔥 Compound AI minimal: {len(filtered_history)} history msgs, no images, full content")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            message = response.choices[0].message
            content = message.content if message.content else ""
            
            # Clean up any raw function calls that might appear
            # Compound AI should execute tools automatically, but sometimes shows the call
            if "<function>" in content or "</function>" in content:
                # Extract only text after function calls
                import re
                logger.warning("Raw function calls detected in response, cleaning up...")
                
                # Remove function call blocks
                content = re.sub(r'<function>.*?</function>', '', content, flags=re.DOTALL)
                # Clean up extra whitespace
                content = re.sub(r'\n{3,}', '\n\n', content.strip())
                
                if not content or len(content) < 10:
                    # If nothing left after cleaning, provide fallback
                    content = "I've searched for the latest information on your query. Could you please rephrase or provide more details so I can give you a better response?"
                    logger.warning("Compound AI response was mostly function calls, using fallback")
            
            state["final_response"] = content
            state["execution_metadata"] = {
                "agent": "compound_ai",
                "model": self.model,
                "executed_tools": getattr(message, "executed_tools", []),
                "truncation_level": "ultra_minimal",  # Only 1 exchange, no images
                "original_history_size": len(conversation_history)
            }
            
            logger.info(f"✅ Compound AI completed successfully with {len(content)} chars")
            return state
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"❌ Compound AI error: {error_message}")
            
            # Provide user-friendly error messages
            if "413" in error_message or "too large" in error_message.lower():
                state["error"] = "Payload too large"
                state["final_response"] = "The conversation history is too long. Please start a new conversation for the best results."
            else:
                state["error"] = error_message
                state["final_response"] = f"I encountered an error fetching real-time data. Please try again. Error: {error_message}"
            
            return state
