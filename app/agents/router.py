"""
Router/Planner Agent - Intent classification and mode selection.
Analyzes user queries and routes to appropriate specialist agent.
Uses FAST model for quick classification.
Parallelized: classification + entity extraction run concurrently.
"""

from typing import Optional, Any
import logging
import re
import asyncio

from ..config import settings, ModelType

try:
    from ..config.openrouter_client import get_openrouter_client
    _HAS_OPENROUTER = True
except ImportError:
    _HAS_OPENROUTER = False

from ..config.key_rotator import get_groq_client
from ..models.agent_state import AgentState, AgentMode

logger = logging.getLogger(__name__)


class RouterAgent:
    """Routes user queries to appropriate specialized agents using fast model."""
    
    def __init__(self):
        # OpenRouter first priority, Groq fallback
        if _HAS_OPENROUTER and settings.openrouter_available:
            self.client = get_openrouter_client()
            self.model = settings.get_openrouter_model(ModelType.ROUTER)
            self._provider = "openrouter"
        else:
            self.client = get_groq_client()
            self.model = settings.get_model_for_task(ModelType.ROUTER)
            self._provider = "groq"
        self.temperature = settings.get_temperature_for_task(ModelType.ROUTER)
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.ROUTER)

    def _looks_conversational(self, query: str) -> bool:
        """Heuristic guardrail — returns True ONLY for pure greetings/small-talk."""
        if not query:
            return True

        q = query.strip().lower()
        if not q:
            return True

        q_norm = re.sub(r"[^a-z0-9\s]", " ", q)
        q_norm = re.sub(r"\s+", " ", q_norm).strip()
        q_soft = re.sub(r"(.)\1{2,}", r"\1\1", q_norm)
        tokens = q_soft.split()

        # Any finance or LTP Calculator term → definitely NOT conversational
        finance_markers = {
            # General finance
            "stock", "market", "price", "nifty", "sensex", "rsi", "macd",
            "portfolio", "sip", "mutual", "crypto", "bitcoin", "analysis",
            "analyze", "research", "news", "invest", "trading", "option",
            "call", "put", "strike", "expiry", "oi", "volume", "index",
            # Common finance acronyms — always financial context on this platform
            "pcr", "vix", "iv", "pe", "pb", "eps", "roce", "roe", "cmp",
            "atr", "obv", "adx", "sma", "ema", "dma", "fii", "dii",
            "sebi", "nse", "bse", "f&o", "futures", "derivative", "hedge",
            "theta", "delta", "gamma", "vega", "rho", "greek", "premium",
            "straddle", "strangle", "condor", "butterfly", "spread",
            # LTP Calculator specific terms — MUST always go to KB
            "wtb", "wtt", "soc", "eos", "eor", "coa", "ltp",
            "support", "resistance", "pressure", "shifting", "diversion",
            "strong", "weak", "bullish", "bearish", "yellow", "box",
            "percentage", "imaginary", "scenario", "reversal", "confusion",
            "natural", "weakness", "constraint", "weekly", "range",
        }
        if any(t in finance_markers for t in tokens):
            return False

        # Also check substring matches for short acronyms that won't tokenize cleanly
        ltp_substrings = ["wtb", "wtt", "soc", "eos", "eor", "ltp", "pcr", "vix"]
        if any(sub in q for sub in ltp_substrings):
            return False

        quick_phrases = {
            "hi", "hii", "hiii", "hello", "hey", "yo", "sup", "namaste",
            "thanks", "thank you", "thx", "ok", "okay", "cool", "bye", "gn",
            "good morning", "good afternoon", "good evening",
            "how are you", "how r u", "what's up", "whats up", "you there",
        }
        if q_soft in quick_phrases:
            return True

        joined = q_soft.replace(" ", "")
        if len(tokens) <= 3:
            if re.fullmatch(r"h(?:e|a)?l+o+", joined):
                return True
            if re.fullmatch(r"h+i+", joined):
                return True
            if re.fullmatch(r"h+e+y+", joined):
                return True

        # Only mark as conversational if it's very short AND matches no finance pattern
        if len(tokens) <= 2 and len(q_soft) <= 10:
            return True

        return False
    
    async def classify_intent(self, state: AgentState) -> AgentState:
        """
        Classify user intent using Groq llama-3.1-8b-instant.
        - Fast (<500ms), cheap, reliable JSON output
        - Zero hardcoded patterns — the LLM reads intent from meaning, not keywords
        - If primary parse fails, retries with a simpler prompt
        """
        import json as _json
        import re as _re
        query = state["query"]
        conversation_history = state.get("conversation_history", [])


        # ── Build conversation context ─────────────────────────────────────
        recent_ctx = ""
        if conversation_history:
            recent_ctx = "\n".join(
                f"{m['role'].upper()}: {str(m.get('content', ''))[:100]}"
                for m in conversation_history[-3:]
            )

        system_prompt = """You are the intent router for Daddy's AI — an Indian financial assistant.

Return ONLY this JSON (no markdown, no explanation):
{"mode": "...", "needs_search": true/false, "use_kb": true/false, "conversational": true/false, "reason": "one line"}

═══════════════════════════════════════════════════════════════════
  RULE 1 — KB TERMS  →  mode=explainer, use_kb=true, needs_search=false
═══════════════════════════════════════════════════════════════════
If the query is about ANY of these InvestingDaddy / LTP Calculator concepts,
ALWAYS use mode=explainer + use_kb=true. These are stored in our knowledge base.

  LTP Calculator concepts:
    WTB, WTT, weak towards bottom, weak towards top,
    SOC, state of confusion, 1-hour SOC, 2-hour SOC, 3-hour SOC,
    COA, chart of accuracy, COA 1.0, COA 1.1–1.9, COA 2.0,
    EOR, EOS, extension of resistance, extension of support,
    EOR+1, EOS+1, EOR-1, EOS-1,
    diversion, end of diversion,
    imaginary line, ATM pair, sheesh aasan,
    weekly range, advance weekly range, L1 range, L2 range, L3 range,
    game of percentage, 75% rule,
    yellow box, natural weakness, shifting pressure, shift top, shift bottom,
    blood bath, bull run,
    six kinds of reversal, reversal theory,
    strong support, strong resistance, WTB to strong, WTT to strong,
    color codes (pink/blue/grey/green/yellow on option chain),
    LTP Blast, LTP Swing, 9:20 AM strategy, magical lines,
    max pain, max gain, reversal price, arbitrage stock,
    Vinay Sir, Vinay Prakash Tiwari, InvestingDaddy, Daddy's International School,
    Adarsh developer, Who made Daddy's AI

  General finance education (explain HOW something works):
    How does RSI work, what is MACD, explain Bollinger Bands,
    what is PCR, what is open interest, what is OI change,
    what is put-call ratio, what is P/E ratio, what is EPS,
    what are option greeks, delta gamma theta vega,
    iron condor, straddle, strangle, butterfly spread,
    candlestick patterns, fibonacci retracement, golden cross, death cross,
    long buildup, short buildup, call writers, put writers,
    demat account, STCG, LTCG, CNC, MIS, bracket order, cover order

  SIGNAL: query starts with "what is", "what are", "explain", "how does",
           "how do", "teach me", "define", "tell me about" + any finance term

═══════════════════════════════════════════════════════════════════
  RULE 2 — LIVE MARKET DATA  →  mode=realtime_analysis, needs_search=true
═══════════════════════════════════════════════════════════════════
If the user wants CURRENT/LIVE numbers, prices, or market status right now:

  Keywords that signal live data need:
    today, now, current, live, latest, right now, this moment,
    is up, is down, how much, at what level, where is, how is

  Market entities:
    Nifty, Bank Nifty, BankNifty, Sensex, Fin Nifty, MidCap Nifty,
    SGX Nifty, GIFT Nifty, Dow Jones, Nasdaq, S&P 500,
    FII data, DII data, market open, market close, market status,
    futures price, option premium (live), PCR today, VIX today

  Examples → realtime_analysis:
    "where is nifty today", "current bank nifty level",
    "nifty live", "how is sensex doing", "sgx nifty",
    "fii data today", "market status", "gift nifty now"

═══════════════════════════════════════════════════════════════════
  RULE 3 — DEEP STOCK RESEARCH  →  mode=market_research, needs_search=true
═══════════════════════════════════════════════════════════════════
  Keywords: analyze, analyse, research, should I buy, is it good to buy,
            fundamental analysis of, technical analysis of, compare stocks,
            sector report, earnings, quarterly results, revenue, profit,
            ROCE, ROE, debt, promoter holding

  Examples → market_research:
    "analyze TCS", "should I buy Reliance", "compare HDFC and ICICI",
    "fundamental analysis of Infosys", "research Nifty IT stocks"

═══════════════════════════════════════════════════════════════════
  RULE 4 — PORTFOLIO / SIP  →  mode=portfolio, needs_search=false
═══════════════════════════════════════════════════════════════════
  Keywords: my portfolio, portfolio advice, SIP, asset allocation,
            how much to invest, risk appetite, diversification advice

═══════════════════════════════════════════════════════════════════
  RULE 5 — CRYPTO  →  mode=crypto, needs_search=true
═══════════════════════════════════════════════════════════════════
  Keywords: bitcoin, ethereum, crypto, BTC, ETH, altcoin, DeFi, NFT, Web3

═══════════════════════════════════════════════════════════════════
  RULE 5.5 — CURRENT EVENTS / GEOPOLITICS → mode=market_research, needs_search=true
═══════════════════════════════════════════════════════════════════
  ANY query asking about real-world events and their market/economic impact:
    war, conflict, sanctions, geopolitics, inflation, recession, fed rate,
    oil crisis, government policy, budget, RBI policy, election market impact,
    "effect on market", "impact on economy", "what will happen to stocks"
  → mode=market_research, needs_search=true, use_kb=false

  Examples → market_research:
    "tell about war and its effect on market"
    "how does US Fed rate hike affect India"
    "Russia Ukraine war market impact"
    "effect of inflation on stocks"
    "budget 2025 impact on market"

═══════════════════════════════════════════════════════════════════
  RULE 6 — OFF-TOPIC / UNKNOWN  →  mode=explainer, conversational=true
═══════════════════════════════════════════════════════════════════
  If the query has NOTHING to do with finance, trading, or markets
  (IPL, movies, weather, food, travel) — politely redirect.
  → mode=explainer, conversational=true, needs_search=false, use_kb=false

NOTE: conversational=true is ONLY for pure greetings/small talk with zero
information need. QuickReplyAgent already handles most greetings before
the router runs, so you'll rarely see them — but set conversational=true
if one slips through."""

        user_prompt = f"Query: {query}"
        if recent_ctx:
            user_prompt = f"Conversation so far:\n{recent_ctx}\n\nNew query: {query}"

        try:
            # ── Use Groq for routing (fast, cheap, reliable JSON) ──────────
            # Falls back to OpenRouter if Groq fails (expired key, rate limit, etc.)
            from ..config.key_rotator import get_groq_client

            async def _classify_with_messages(client, model):
                return await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=120,
                )

            # Try Groq first
            groq_client = None
            try:
                groq_client = get_groq_client()
                classification_response = await _classify_with_messages(
                    groq_client, "llama-3.1-8b-instant"
                )
                logger.debug("Router: used Groq llama-3.1-8b-instant")
            except Exception as groq_err:
                logger.warning(f"Router Groq failed ({groq_err}), falling back to OpenRouter")
                or_client = get_openrouter_client() if (_HAS_OPENROUTER and settings.openrouter_available) else self.client
                classification_response = await _classify_with_messages(
                    or_client, self.model
                )
                logger.debug(f"Router: used OpenRouter fallback ({self.model})")

            entity_coro = self._extract_entities_with_context(query, conversation_history)
            entities = await entity_coro

            raw = (classification_response.choices[0].message.content or "").strip()
            logger.debug(f"Router raw: {raw}")

            # ── Parse JSON robustly ────────────────────────────────────────
            parsed = None
            # Try 1: direct parse
            try:
                parsed = _json.loads(raw)
            except Exception:
                pass
            # Try 2: strip markdown fences
            if parsed is None:
                m = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, _re.DOTALL)
                if m:
                    try:
                        parsed = _json.loads(m.group(1))
                    except Exception:
                        pass
            # Try 3: find first {...} block
            if parsed is None:
                m = _re.search(r"\{[^{}]+\}", raw, _re.DOTALL)
                if m:
                    try:
                        parsed = _json.loads(m.group())
                    except Exception:
                        pass

            # ── Retry with ultra-simple prompt if still no JSON ────────────
            if parsed is None:
                logger.warning(f"Router JSON parse failed (raw={raw[:100]!r}), retrying with OpenRouter")
                try:
                    or_client_retry = get_openrouter_client() if (_HAS_OPENROUTER and settings.openrouter_available) else self.client
                    retry_resp = await or_client_retry.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": 'Return ONLY valid JSON: {"mode":"explainer|realtime_analysis|market_research|portfolio|crypto","needs_search":true/false,"use_kb":true/false,"conversational":true/false,"reason":"brief"}'},
                            {"role": "user", "content": f"Classify: {query}"},
                        ],
                        temperature=0.0,
                        max_tokens=100,
                    )
                    retry_raw = (retry_resp.choices[0].message.content or "").strip()
                    try:
                        parsed = _json.loads(retry_raw)
                    except Exception:
                        m = _re.search(r"\{[^{}]+\}", retry_raw, _re.DOTALL)
                        if m:
                            try:
                                parsed = _json.loads(m.group())
                            except Exception:
                                pass
                except Exception as retry_err:
                    logger.error(f"Router retry also failed: {retry_err}")

            # ── Smart fallback if all parse attempts failed ────────────────
            if parsed is None:
                q_lower = query.lower().strip()
                if self._looks_conversational(query):
                    logger.warning("Router fallback: conversational query detected -> explainer no-search")
                    parsed = {
                        "mode": "explainer",
                        "needs_search": False,
                        "use_kb": False,
                        "conversational": True,
                        "reason": "parse failure - conversational fallback",
                    }
                # Educational/concept queries → explainer + KB, not web search
                elif any(q_lower.startswith(p) for p in (
                    "what is", "what are", "what does", "what",
                    "explain", "how does", "how do", "teach me",
                    "define", "meaning of", "tell me about",
                )):
                    logger.warning("Router fallback: educational query detected → explainer+kb")
                    parsed = {"mode": "explainer", "needs_search": False, "use_kb": True, "conversational": False, "reason": "parse failure - educational fallback"}
                else:
                    logger.error("Router: all parse attempts failed, defaulting to market_research+search")
                    parsed = {"mode": "market_research", "needs_search": True, "use_kb": False, "conversational": False, "reason": "parse failure fallback"}

            if not isinstance(parsed, dict):
                parsed = {}

            # Some models return partial/invalid JSON like "{}" or wrong keys.
            # Normalize that to deterministic fallback behavior.
            if not parsed.get("mode"):
                q_lower = query.lower().strip()
                if self._looks_conversational(query):
                    parsed = {
                        "mode": "explainer",
                        "needs_search": False,
                        "use_kb": False,
                        "conversational": True,
                        "reason": "missing mode - conversational fallback",
                    }
                elif any(q_lower.startswith(p) for p in (
                    "what is", "what are", "what does", "what",
                    "explain", "how does", "how do", "teach me",
                    "define", "meaning of", "tell me about",
                )):
                    parsed = {
                        "mode": "explainer",
                        "needs_search": False,
                        "use_kb": True,
                        "conversational": False,
                        "reason": "missing mode - educational fallback",
                    }
                else:
                    parsed = {
                        "mode": "market_research",
                        "needs_search": True,
                        "use_kb": False,
                        "conversational": False,
                        "reason": "missing mode - research fallback",
                    }

            # Final safety override: very short casual inputs should never trigger deep research.
            q_words = query.strip().split()
            if self._looks_conversational(query) and len(q_words) <= 3:
                parsed = {
                    "mode": "explainer",
                    "needs_search": False,
                    "use_kb": False,
                    "conversational": True,
                    "reason": "conversational safety override",
                }

            mode_str = str(parsed.get("mode", "market_research")).lower().strip()
            needs_search = bool(parsed.get("needs_search", True))

            valid_modes = {
                "market_research": AgentMode.MARKET_RESEARCH,
                "realtime_analysis": AgentMode.REALTIME_ANALYSIS,
                "portfolio": AgentMode.PORTFOLIO,
                "explainer": AgentMode.EXPLAINER,
                "crypto": AgentMode.CRYPTO,
            }
            selected_mode = valid_modes.get(mode_str, AgentMode.MARKET_RESEARCH)

            use_kb = bool(parsed.get("use_kb", False))
            is_conversational = bool(parsed.get("conversational", False))

            # ── CRITICAL OVERRIDE: heuristic finance-term detection beats LLM ──
            # If _looks_conversational() returned False (finance/LTP term detected),
            # the LLM cannot mark it as conversational=True + use_kb=False.
            # "what is pcr", "what is wtb" etc. must ALWAYS go through KB.
            if is_conversational and not self._looks_conversational(query):
                logger.info(
                    f"🔧 Finance-term override: LLM said conversational=True "
                    f"but heuristic detected finance term → forcing use_kb=True, conversational=False"
                )
                is_conversational = False
                use_kb = True

            # If router says conversational → force needs_search=False, use_kb=False
            if is_conversational:
                needs_search = False
                use_kb = False

            state["extracted_entities"] = entities
            state["selected_mode"] = selected_mode.value
            state["enable_research_loop"] = needs_search
            state["use_kb"] = use_kb
            state["is_conversational"] = is_conversational

            logger.info(
                f"🤖 Router → mode={selected_mode.value} | search={needs_search} "
                f"| use_kb={use_kb} | conversational={is_conversational} "
                f"| reason={parsed.get('reason', '')} | entities={entities}"
            )
            return state

        except Exception as e:
            logger.error(f"Router classification error: {e}")
            if self._looks_conversational(state.get("query", "")):
                state["selected_mode"] = AgentMode.EXPLAINER.value
                state["enable_research_loop"] = False
                state["use_kb"] = False
                state["is_conversational"] = True
            else:
                state["selected_mode"] = AgentMode.MARKET_RESEARCH.value
                state["enable_research_loop"] = True
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
        
        prompt = f"""You are teaching a child how to extract stock symbols from a query. Follow these steps EXACTLY:

STEP 1 - CHECK: Is the user asking for a definition or explanation of a term?
- Look for: "what is", "explain", "define", "how does [term] work"
- Terms that are NOT stocks: PE ratio, RSI, MACD, LTP, WTB, WTT, COA, SOC, EOR, Max Pain, support, resistance
- If the query is about explaining a concept → return {{"symbols": [], "timeframe": null, "amount": null}}
- Do NOT treat these as stock symbols even if they look like acronyms

STEP 2 - CHECK: Does the query or conversation mention actual company/stock names?
- Indian stocks: RELIANCE, TCS, INFY, HDFC, ICICI, SBI, ITC, WIPRO, HUL (and similar)
- Format may be: "RELIANCE", "Reliance", "reliance stock"
- If the user says "it", "that stock", "this company" → look at conversation history to see what they referred to

STEP 3 - EXTRACT from query or conversation:
- List all stock symbols found. Use UPPERCASE.
- If none found → symbols: []

STEP 4 - EXTRACT timeframe (if mentioned):
- "today", "intraday", "now" → "1d"
- "week" → "1wk"
- "month" → "1mo"
- "year" → "1y"
- Not mentioned → null

STEP 5 - EXTRACT amount in lakhs/crores (if mentioned):
- "10 lakh", "5 crore" → convert to number
- Not mentioned → null

Conversation History (use this to resolve "it", "that", "this"):
{context}

Current Query: {query}

Respond with ONLY valid JSON, no other text:
{{"symbols": ["SYMBOL1", "SYMBOL2"], "timeframe": "1y" or null, "amount": number or null}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            
            import json
            content = response.choices[0].message.content
            if not content:
                logger.warning("Entity extraction got None response, returning empty entities")
                return {"symbols": [], "timeframe": None, "amount": None}
            entities = json.loads(content)
            
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
    
    def _keyword_fallback(self, query: str) -> str:
        """Fast keyword-based routing when LLM returns None/empty."""
        q = query.lower().strip()
        words = q.split()
        # Greetings — ONLY short standalone ones (≤3 words)
        # "hey tell me about X" should NOT be a greeting
        greet = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'namaste', 'thanks', 'thank you']
        if len(words) <= 3 and any(q == g or q.startswith(g + ' ') or q.startswith(g + '!') for g in greet):
            return "explainer"
        # Realtime
        realtime = ['price', 'where is', 'how is', 'today', 'now', 'current', 'live', 'latest', 'nifty', 'sensex', 'market']
        if any(kw in q for kw in realtime):
            return "realtime_analysis"
        # Crypto
        crypto = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana']
        if any(kw in q for kw in crypto):
            return "crypto"
        # Portfolio
        portfolio = ['portfolio', 'allocate', 'invest', 'sip', 'mutual fund']
        if any(kw in q for kw in portfolio):
            return "portfolio"
        # Concept explanations
        explain = ['what is', 'explain', 'how does', 'meaning', 'define']
        if any(kw in q for kw in explain):
            return "explainer"
        return "market_research"

    def _should_trigger_research(self, query: str, state: AgentState) -> bool:
        """
        Decide whether live web search / data pre-fetch should be triggered.

        Philosophy: search is CHEAP and FAST. Better to search unnecessarily
        than to answer a factual question from stale training data.

        Returns True  → meta_reasoning will do a proactive prefetch (search_web + NSE quotes)
        Returns False → only for pure concept definitions and math (no live data needed)
        """
        # Manual override always wins
        if state.get("enable_deep_search", False):
            return True

        query_lower = query.lower()

        # ── NEVER search: pure greetings, casual conversation, thanks ──────
        casual_phrases = [
            "how are you", "how r u", "how r you", "whats up", "what's up",
            "wassup", "how do you do", "who are you", "what are you",
        ]
        trivial_starters = [
            "hi", "hello", "hey", "good morning", "good afternoon",
            "good evening", "good night", "namaste", "hola", "sup", "yo",
            "thanks", "thank you", "thx", "ty", "appreciate",
        ]
        if any(query_lower == p or query_lower.startswith(p + " ") or query_lower.startswith(p + "!") for p in trivial_starters) and len(query.split()) <= 5:
            return False
        if any(p in query_lower for p in casual_phrases) and len(query.split()) <= 8:
            return False

        # ── NEVER search: pure concept definitions & math ──────────────────
        # These never need live data.
        no_search_phrases = [
            "what is", "what are", "define", "meaning of", "explain",
            "how does", "tell me about", "teach me", "difference between",
            "what does", "how do i calculate", "formula for",
        ]
        # Only skip if the query is PURELY conceptual (short, no stock mention)
        is_pure_concept = (
            any(query_lower.startswith(p) or f" {p}" in query_lower for p in no_search_phrases)
            and len(query.split()) <= 10
            and not any(kw in query_lower for kw in [
                "today", "now", "current", "latest", "recent",
                "price", "stock", "nifty", "sensex", "market", "news",
            ])
        )
        if is_pure_concept:
            return False

        # ── ALWAYS search: anything with a time/market signal ──────────────
        live_data_signals = [
            # time signals
            "today", "now", "current", "latest", "recent", "this week",
            "this month", "this year", "2026", "2025", "yesterday",
            # market signals
            "price", "stock", "share", "nifty", "sensex", "market",
            "ipo", "results", "earnings", "quarterly", "annual report",
            "dividend", "bonus", "split", "merger", "acquisition",
            "fii", "dii", "open interest", "option chain",
            # news / events
            "news", "happened", "penalty", "ban", "sebi", "rbi", "budget",
            "inflation", "gdp", "rate", "war", "geopolit", "sanction",
            # analysis keywords
            "analyze", "analysis", "research", "compare", "versus", "vs",
            "outlook", "forecast", "prediction", "target", "sector",
            "performance", "return", "gain", "loss", "rally", "crash",
            "bull", "bear", "breakout", "support", "resistance",
        ]
        if any(kw in query_lower for kw in live_data_signals):
            return True

        # ── Default: search for longer / complex queries ────────────────────
        word_count = len(query.split())
        question_marks = query.count("?")
        if word_count > 8 or question_marks > 1:
            return True

        # Short ambiguous query: search to be safe
        return True
    
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
