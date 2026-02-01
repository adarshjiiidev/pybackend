"""
Portfolio Intelligence Agent - Asset allocation and risk management.
Provides portfolio construction and optimization strategies.
Uses REASONING model for strategic planning.
"""

from groq import AsyncGroq
import logging

from ..config import settings, ModelType
from ..models.agent_state import AgentState

logger = logging.getLogger(__name__)


class PortfolioAgent:
    """Provides portfolio allocation and risk management insights using reasoning model."""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        # Use deep reasoning model for strategic portfolio planning
        self.model = settings.get_model_for_task(ModelType.REASONING_DEEP)
        self.temperature = settings.get_temperature_for_task(ModelType.REASONING_DEEP)
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.REASONING_DEEP)
    
    async def analyze(self, state: AgentState) -> AgentState:
        """
        Generate portfolio allocation strategies.
        Considers risk profile, diversification, and India-specific factors.
        """
        query = state["query"]
        entities = state.get("extracted_entities") or {}
        amount = entities.get("amount")
        
        # Generate portfolio recommendations
        analysis = await self._generate_portfolio_advice(
            query,
            amount,
            state.get("conversation_history", [])
        )
        
        state["final_response"] = analysis
        state["execution_metadata"] = {
            "agent": "portfolio",
            "amount_analyzed": amount
        }
        
        return state
    
    async def _generate_portfolio_advice(
        self,
        query: str,
        amount: float,
        conversation_history: list
    ) -> str:
        """Generate portfolio allocation advice."""
        
        amount_context = ""
        if amount:
            amount_str = f"₹{amount:,.0f}"
            if amount >= 10000000:
                amount_str = f"₹{amount/10000000:.2f} Cr"
            elif amount >= 100000:
                amount_str = f"₹{amount/100000:.2f} L"
            amount_context = f"\nInvestment Amount: {amount_str}"
        
        system_prompt = """You are a portfolio strategist specializing in Indian investors.

Your role:
- Design asset allocation strategies based on risk profile
- Recommend diversification across asset classes (equity, debt, gold, etc.)
- Consider India-specific factors (LTCG/STCG tax, PPF, ELSS, etc.)
- Provide sector-wise allocation for equity portion
- Balance growth potential with risk management
- Use Goal-based investing framework when relevant

Format your response:
1. Risk Profile Assessment
2. Recommended Asset Allocation
3. Equity Sector Breakdown (if applicable)
4. Tax-efficient Instruments
5. Rebalancing Strategy
6. Important Considerations

CRITICAL:
- Adapt allocation to user's risk appetite (conservative/moderate/aggressive)
- Mention tax implications (LTCG, STCG, 80C benefits)
- Include emergency fund reminder
- Clear disclaimer: This is educational, not personalized financial advice
- Be specific with percentage allocations
- Consider investment horizon"""

        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in conversation_history[-5:]:
            messages.append(msg)
        
        user_message = f"{amount_context}\nUser Query: {query}\n\nProvide portfolio allocation strategy."
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            advice = response.choices[0].message.content
            logger.info("Portfolio advice generated successfully")
            return advice
            
        except Exception as e:
            logger.error(f"Portfolio agent error: {e}")
            return f"I encountered an error while generating portfolio advice. Please try again. Error: {str(e)}"
