"""
Router/Planner Agent - Intent classification and mode selection.
Analyzes user queries and routes to appropriate specialist agent.
Uses FAST model for quick classification.
"""

from groq import AsyncGroq
from typing import Optional, Any
import logging
import re

from ..config import settings, ModelType
from ..models.agent_state import AgentState, AgentMode

logger = logging.getLogger(__name__)


class RouterAgent:
    """Routes user queries to appropriate specialized agents using fast model."""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.get_model_for_task(ModelType.ROUTER)
        self.temperature = settings.get_temperature_for_task(ModelType.ROUTER)
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.ROUTER)
    
    async def classify_intent(self, state: AgentState) -> AgentState:
        """
        Classify user intent and select appropriate agent mode.
        Also extract entities with conversation context awareness.
        If images are present, set flag for vision processing.
        """
        query = state["query"]
        images = state.get("images", [])
        conversation_history = state.get("conversation_history", [])
        
        # If images present, mark for vision processing
        if images:
            logger.info(f"🖼️ Images detected ({len(images)}), marking for vision processing")
            state["has_vision_content"] = True
            # For image queries, prefer explainer mode for better descriptions
            state["selected_mode"] = AgentMode.EXPLAINER.value
            state["extracted_entities"] = {}
            return state
        
        # Detect if deep search / research loop should be activated
        should_research = self._should_trigger_research(query, state)
        state["enable_research_loop"] = should_research
        
        system_prompt = """You are the routing brain of Daddys AI, a financial intelligence system.

Your job: Analyze the query and select the appropriate specialist mode.

**Mode Selection** - Route to the right specialist:
   - market_research: Deep fundamental analysis, company research
   - realtime_analysis: Price movements, technical analysis, current trends, latest market news
   - portfolio: Portfolio optimization, asset allocation
   - explainer: Educational content, concept clarification, definitions
   - crypto: Cryptocurrency analysis

Respond ONLY with the mode name (no JSON, just the mode string).

Examples:
- "what is happening in japan elections" → explainer
- "what is RSI in trading" → explainer
- "latest news on reliance stock" → realtime_analysis
- "explain portfolio diversification" → explainer
- "analyze TCS fundamentals" → market_research"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this query: {query}"}
                ],
                temperature=self.temperature,
                max_tokens=100
            )
            
            import json
            result_text = response.choices[0].message.content.strip()
            
            # Parse response - now it's just a mode string
            mode = result_text.lower()
            
            # Validate mode
            valid_modes = {
                "market_research": AgentMode.MARKET_RESEARCH,
                "realtime_analysis": AgentMode.REALTIME_ANALYSIS,
                "portfolio": AgentMode.PORTFOLIO,
                "explainer": AgentMode.EXPLAINER,
                "crypto": AgentMode.CRYPTO
            }
            
            selected_mode = valid_modes.get(mode, AgentMode.MARKET_RESEARCH)
            
            # Extract entities with conversation context
            entities = await self._extract_entities_with_context(query, conversation_history)
            state["extracted_entities"] = entities
            
            # Override mode to explainer if it's a financial term query with no stock symbols
            from ..tools.financial_terms import is_financial_term
            if is_financial_term(query) and not entities.get("symbols"):
                logger.info(f"📚 Detected financial term query")
                selected_mode = AgentMode.EXPLAINER
            
            state["selected_mode"] = selected_mode.value
            
            logger.info(f"🤖 Router Decision → Mode: {selected_mode.value}, Research: {should_research}, Entities: {entities}")
            return state
            
        except Exception as e:
            logger.error(f"Router classification error: {e}")
            state["selected_mode"] = AgentMode.MARKET_RESEARCH.value
            state["extracted_entities"] = {}
            return state
    
    
    async def _extract_entities_with_context(self, query: str, conversation_history: list) -> dict:
        """Extract entities using conversation history for context (resolves pronouns like 'it')."""
        from ..tools.financial_terms import is_financial_term
        
        # Check if query is about a financial term (not a stock symbol)
        if is_financial_term(query):
            logger.info(f"📚 Detected financial term query: {query}")
            return {"symbols": [], "timeframe": None, "amount": None}
        
        # Build context from recent conversation
        context = ""
        if conversation_history:
            recent_msgs = conversation_history[-4:]  # Last 2 exchanges
            for msg in recent_msgs:
                role = "User" if msg["role"] == "user" else "AI"
                context += f"{role}: {msg['content'][:150]}\n"
        
        prompt = f"""Extract stock symbols from the query. If query uses pronouns (it, that, this), check conversation history.

IMPORTANT: Do NOT extract symbols if the query is asking for:
- Definitions or explanations of financial terms/acronyms
- Trading tools like LTP Calculator, WTB/WTT, COA, SOC, EOR, Max Pain
- Features, strategies, or trading concepts
- Educational content about market terminology

Conversation History:
{context}

Current Query: {query}

Respond with JSON: {{"symbols": ["SYMBOL1"], "timeframe": "1y", "amount": null}}
If no symbols or if asking for term definition/tool explanation, return {{"symbols": [], "timeframe": null, "amount": null}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            
            import json
            entities = json.loads(response.choices[0].message.content)
            
            # Normalize symbols to uppercase and filter out financial terms
            if "symbols" in entities and entities["symbols"]:
                filtered_symbols = []
                for s in entities["symbols"]:
                    s_upper = s.upper()
                    if not is_financial_term(s_upper):
                        filtered_symbols.append(s_upper)
                    else:
                        logger.info(f"🚫 Filtered out financial term: {s_upper}")
                
                entities["symbols"] = filtered_symbols
            
            return entities
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            # Fallback to regex extraction
            return await self._extract_entities(query)
    
    def _should_trigger_research(self, query: str, state: AgentState) -> bool:
        """
        Detect if autonomous research loop should be triggered.
        
        Triggers on:
        1. Manual enable_deep_search flag
        2. Complex multi-part questions
        3. Comparison queries  
        4. Deep analysis keywords
        """
        # Check manual flag first
        if state.get("enable_deep_search", False):
            return True
        
        query_lower = query.lower()
        
        # Complexity indicators
        complexity_keywords = [
            "compare", "versus", "vs", "analyze deeply", "research",
            "comprehensive analysis", "detailed study", "investigate",
            "which is better", "what are the differences"
        ]
        
        # Check for keywords
        has_complexity = any(kw in query_lower for kw in complexity_keywords)
        
        # Check for multiple questions (multi-part)
        question_marks = query.count("?")
        is_multipart = question_marks > 1
        
        # Check query length (longer = more complex)
        word_count = len(query.split())
        is_long = word_count > 15
        
        # Trigger if any indicator is met
        return has_complexity or is_multipart or is_long
    
    async def _extract_entities(self, query: str) -> dict[str, Any]:
        """Extract stock symbols, timeframes, and other entities from query."""
        entities = {
            "symbols": [],
            "timeframe": None,
            "amount": None
        }
        
        #Extract common Indian stock symbols
        stock_patterns = [
            r'\b(RELIANCE|TCS|INFY|HDFC|ICICI|SBI|TATA|ITC|WIPRO|HUL)\b',
            r'\b([A-Z]{2,}\.NS|[A-Z]{2,}\.BO)\b'
        ]
        
        for pattern in stock_patterns:
            matches = re.findall(pattern, query.upper())
            entities["symbols"].extend(matches)
        
        # Extract crypto symbols
        crypto_pattern = r'\b(BTC|ETH|BITCOIN|ETHEREUM|CRYPTO)\b'
        crypto_matches = re.findall(crypto_pattern, query.upper())
        if crypto_matches:
            entities["symbols"].extend(crypto_matches)
        
        # Extract timeframe
        if any(word in query.lower() for word in ["today", "intraday", "now"]):
            entities["timeframe"] = "1d"
        elif any(word in query.lower() for word in ["week", "weekly"]):
            entities["timeframe"] = "1wk"
        elif any(word in query.lower() for word in ["month", "monthly"]):
            entities["timeframe"] = "1mo"
        elif any(word in query.lower() for word in ["year", "yearly", "annual"]):
            entities["timeframe"] = "1y"
        
        # Extract amount (in lakhs/crores)
        amount_pattern = r'(\d+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores|L|Cr)'
        amount_match = re.search(amount_pattern, query, re.IGNORECASE)
        if amount_match:
            value = float(amount_match.group(1))
            unit = amount_match.group(2).lower()
            if 'lakh' in unit or unit == 'l':
                entities["amount"] = value * 100000
            elif 'crore' in unit or unit == 'cr':
                entities["amount"] = value * 10000000
        
        return entities
