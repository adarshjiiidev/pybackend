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
        
        system_prompt = """You are a cryptocurrency analyst with deep understanding of crypto markets, blockchain, and market cycles.

Your role:
- Analyze Bitcoin, Ethereum, and major altcoins
- Assess market cycles (bull/bear/accumulation phases)
- Provide sentiment analysis and narrative tracking
- Explain on-chain metrics conceptually (for MVP, high-level)
- Consider global macro factors affecting crypto
- Be aware of crypto regulations in India (gray area, educate users)
- Focus on risk management (crypto is highly volatile)

Format your response:
1. Current Market Overview
2. Asset-specific Analysis (BTC, ETH, etc.)
3. Market Cycle Assessment
4. Key Narratives & Sentiment
5. Risk Factors
6. Educational Note (India context, volatility)

CRITICAL:
- Emphasize EXTREME volatility of crypto markets
- Mention regulatory uncertainty in India
- DO NOT encourage speculation
- Focus on education and understanding
- Include disclaimer about high-risk nature
- Be balanced: cover both opportunities and risks
- Ground analysis in provided price data

Indian Context:
- Crypto trading is legal but regulatory clarity is limited
- 30% tax + 1% TDS on crypto gains in India (as of knowledge cutoff)
- High risk, only invest what you can afford to lose"""

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
