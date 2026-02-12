"""
Crypto Intelligence Agent - Cryptocurrency analysis and insights.
Covers Bitcoin, Ethereum, and major altcoins with market cycle awareness.
Uses REASONING model for complex market analysis.
"""

from groq import AsyncGroq
import logging

from ..config import settings, ModelType
from ..models.agent_state import AgentState
from ..tools import get_crypto_data, format_for_llm
from ..database import MarketDataCacheManager

logger = logging.getLogger(__name__)


class CryptoAgent:
    """Provides cryptocurrency market analysis and insights using reasoning model."""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        # Use deep reasoning model for crypto market analysis
        self.model = settings.get_model_for_task(ModelType.REASONING_DEEP)
        self.temperature = settings.get_temperature_for_task(ModelType.REASONING_DEEP)
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.REASONING_DEEP)
        self.cache = MarketDataCacheManager()
    
    async def analyze(self, state: AgentState) -> AgentState:
        """
        Perform cryptocurrency market analysis.
        Covers BTC, ETH, and major altcoins with cycle and sentiment analysis.
        """
        query = state["query"]
        entities = state.get("extracted_entities") or {}
        symbols = entities.get("symbols", [])
        
        # Default crypto symbols if none extracted
        if not symbols:
            symbols = ["BTC", "ETH"]
        
        # Map common terms to symbols
        symbol_map = {
            "BTC": "BTC", "BITCOIN": "BTC",
            "ETH": "ETH", "ETHEREUM": "ETH",
            "CRYPTO": "BTC"  # Default to BTC for generic crypto queries
        }
        
        symbols = [symbol_map.get(s.upper(), s.upper()) for s in symbols]
        
        # Fetch crypto data
        tool_results = {}
        for symbol in symbols[:3]:  # Limit to 3
            crypto_data = await self._fetch_crypto_data(symbol)
            tool_results[symbol] = crypto_data
        
        state["tool_results"] = tool_results
        
        # Generate analysis
        analysis = await self._generate_analysis(query, tool_results, state.get("conversation_history", []))
        
        state["final_response"] = analysis
        state["execution_metadata"] = {
            "agent": "crypto",
            "symbols_analyzed": list(tool_results.keys())
        }
        
        return state
    
    async def _fetch_crypto_data(self, symbol: str) -> dict:
        """Fetch cryptocurrency data with caching."""
        async def fetch_fresh():
            return await get_crypto_data(symbol)
        
        return await self.cache.get_or_fetch(
            symbol=symbol,
            data_type="crypto",
            fetch_func=fetch_fresh,
            ttl_seconds=120  # 2 minutes cache
        )
    
    async def _generate_analysis(
        self,
        query: str,
        tool_results: dict,
        conversation_history: list
    ) -> str:
        """Generate cryptocurrency market analysis."""
        
        # Build context
        context = "Cryptocurrency Data:\n\n"
        for symbol, data in tool_results.items():
            context += format_for_llm(data, "crypto") + "\n"
        
        system_prompt = """You are a cryptocurrency analyst. You have price data for the user. Your job: analyze it, explain what it means, and ALWAYS emphasize risk. Think of yourself as a teacher who must warn students before they do something risky.

=== STEP 1: USE THE DATA YOU'RE GIVEN ===
You will receive cryptocurrency price data. Use those numbers. Don't make up prices. Cite: "BTC is currently at $X based on the data..."

=== STEP 2: STRUCTURE YOUR RESPONSE ===

SECTION 1 - Current Market Overview
- What's the overall picture? Bullish/bearish/sideways?
- Use the price data provided. Be specific.

SECTION 2 - Asset-Specific Analysis (for each coin in the data)
- Price level, trend, key support/resistance if obvious
- What's driving it? (narratives, news - if you know)

SECTION 3 - Market Cycle Assessment
- Where are we? Bull run, bear market, accumulation?
- Explain in simple terms. "We may be in an accumulation phase where..."

SECTION 4 - Key Narratives & Sentiment
- What stories are moving the market? (ETF, regulations, halving, etc.)
- Keep it high-level - we're not doing deep on-chain here

SECTION 5 - Risk Factors
- MUST include: extreme volatility, 24/7 market, no circuit breakers
- Regulatory risk, especially in India
- "Never invest more than you can afford to lose"

SECTION 6 - India Context (REQUIRED)
- Crypto is legal but regulatory clarity is limited
- 30% tax + 1% TDS on gains (as of our knowledge)
- RBI has expressed concerns. Regulatory changes possible.
- This is educational - not advice to buy/sell

=== STEP 3: WHAT YOU MUST NEVER DO ===
- Don't say "Buy BTC" or "Sell ETH" - we educate, we don't recommend
- Don't downplay risk - crypto can go -80% in weeks
- Don't encourage speculation or FOMO
- Don't promise returns

=== STEP 4: TONE ===
- Balanced. Opportunities AND risks.
- Educational. "Understanding that..." "It's important to know..."
- Cautionary. "High risk." "Volatile." "Do your own research."
- Use $ for crypto (global market). Use ₹ when talking India tax."""

        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in conversation_history[-5:]:
            messages.append(msg)
        
        user_message = f"{context}\nUser Query: {query}\n\nProvide cryptocurrency market analysis."
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            analysis = response.choices[0].message.content
            logger.info("Crypto analysis generated successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"Crypto agent error: {e}")
            return f"I encountered an error while analyzing cryptocurrency data. Please try again. Error: {str(e)}"
