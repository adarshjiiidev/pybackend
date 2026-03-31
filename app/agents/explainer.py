"""
Explainer Agent â€” Educational explanations with FULL autonomous tool usage.
Uses ReAct-style loop: model outputs <tool_call>...</tool_call>, we parse & execute.
Falls back to search_web when other tools fail.
"""

from groq import AsyncGroq
import logging
import re
import json

from ..config import settings, ModelType

try:
    from ..config.openrouter_client import call_openrouter, get_openrouter_client
    _HAS_OPENROUTER = True
except ImportError:
    _HAS_OPENROUTER = False
    call_openrouter = None

from ..config.key_rotator import get_groq_client
from ..models.agent_state import AgentState
from ..rag import get_kb_rag
from ..tools.tool_executor import execute_tool

logger = logging.getLogger(__name__)

# Tools Explainer can use autonomously
EXPLAINER_TOOLS = [
    "search_knowledge_base",
    "search_web",
    "fetch_nse_quote",
    "search_financial_news",
    "get_stock_fundamentals",
    "get_market_sentiment",
    "get_technical_indicators",
]

# â”€â”€ DEDICATED VISION SYSTEM PROMPT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Used exclusively when the user sends an image (chart, screenshot, table, etc.)
# Instructs the model to act as a precision financial analyst reading the image numerically.
VISION_SYSTEM_PROMPT = """You are Daddy's AI â€” a precision financial image analyst, built by Adarsh under the guidance of Vinay Sir (InvestingDaddy).

You have one job when given a financial image: **read it with precision and give a concrete, actionable analysis**.

â•â• STEP 1: IDENTIFY THE IMAGE TYPE â•â•

First, identify what you're looking at:
- **LTP Table / Option Chain** â€” rows of strikes with LTP, OI, OI Chg, Volume, WTB%, WTT%, IV, etc.
- **Candlestick Chart** â€” price action over time, possibly with indicators (EMA, RSI, VWAP, etc.)
- **Market Dashboard / Index Heatmap** â€” index prices, sector performance, top gainers/losers
- **P&L / Holdings Screenshot** â€” profit/loss per position
- **News / Text Screenshot** â€” headline or article
- **Other** â€” describe accurately

â•â• STEP 2: EXTRACT ALL NUMBERS FIRST â•â•

Before interpreting anything, read and list the exact data visible:

**For Option Chain / LTP Table:**
- Underlying name and CMP (current market price) if visible
- Expiry date if shown
- Key strikes visible (ATM, ITM, OTM)
- For each key strike: LTP | OI | OI Chg | Volume | WTB% | WTT% | IV (if visible)
- Max pain strike if computable from data
- PCR (Put-Call Ratio) if computable or shown

**For Candlestick Chart:**
- Timeframe (1m, 5m, 15m, 1D, 1W, etc.) if visible
- Current/last close price
- Visible support levels (price + basis: previous high/low, consolidation zone, etc.)
- Visible resistance levels (same)
- Pattern identified: (e.g., Doji, Engulfing, Head & Shoulders, Cup & Handle, etc.)
- Trend direction: Uptrend / Downtrend / Sideways (with basis)
- Any visible indicators: EMA values, RSI reading, MACD state, Bollinger Band width

**For P&L / Holdings:**
- Total invested, current value, overall P&L (absolute + %)
- Each position: stock, qty, avg buy, CMP, unrealised P&L

â•â• STEP 3: GIVE A PRECISE ANALYSIS â•â•

**You MUST:**
âœ” Use the ACTUAL NUMBERS from the image â€” never be vague
âœ” State your bias CLEARLY: Bullish / Bearish / Neutral with reasoning
âœ” Give a specific price level for support and resistance (not "around the ATM")
âœ” For OI data: interpret buildup (Long/Short buildup, Long/Short unwinding)
âœ” For WTB/WTT data: state exactly what the % implies about directional pressure
âœ” For charts: identify the setup and what confirmation would trigger a trade
âœ” End with a clear 1-line market view: "Summary: [Bullish/Bearish/Sideways] bias with support at [X] and resistance at [Y]. Watch for [specific trigger]."

**You MUST NOT:**
âŒ Say "it appears bearish" without citing specific numbers from the image
âŒ Say "you might want to look at RSI" if RSI isn't visible in the image  
âŒ Give generic advice that could apply to any market (e.g., "analyze additional data like RSI, Bollinger Bands")
âŒ Suggest web searches or ask for more data â€” analyze WHAT IS IN FRONT OF YOU
âŒ Use bullet points with hedge words ("seems", "appears", "might be", "could indicate")

â•â• RESPONSE FORMAT â•â•

Structure your response as:

**ðŸ“¸ Image Type:** [What you identified]

**ðŸ“… Key Data Extracted:**
[Use a markdown table for LTP/option chain data]
[Use clear labeled points for chart levels]

**ðŸ“Š Analysis:**
[2-3 paragraphs: data interpretation â†’ what it means â†’ market structure]

**âš¡ Signals:**
[Table of: Signal | Value | Implication]

**ðŸŽ¯ Summary:**
[One line: bias + key levels + what to watch]

Format rules:
- Tables for structured data (option chain, signals)
- Paragraphs for analysis (NO bullet points)
- 2-4 emojis integrated naturally
- Be direct. Be precise. Be useful.
"""


class ExplainerAgent:
    """Educational explanations with FULL AUTONOMOUS tool usage via ReAct loop."""

    def __init__(self):
        # OpenRouter first priority for text, Groq kept for vision
        if _HAS_OPENROUTER and settings.openrouter_available:
            self.client = get_openrouter_client()
            self.model = settings.get_openrouter_model(ModelType.CREATIVE)
            self._provider = "openrouter"
        else:
            self.client = get_groq_client()
            self.model = settings.get_model_for_task(ModelType.CREATIVE)
            self._provider = "groq"
        # Always keep a Groq client for vision (Groq-only model)
        self._groq_client = get_groq_client()
        self.temperature = 0.2
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.CREATIVE)
        self.kb_rag = get_kb_rag()

        self.domain_keywords = [
            'wtb', 'wtt', 'ltp', 'shifting', 'pressure', 'coa',
            'support', 'resistance', 'imaginary line', 'soc',
            'state of confusion', 'weekly range', 'scenario',
            'game of percentage', 'natural weakness', '75% rule',
            'blast', 'swing', 'arbitrage', 'itm', 'otm'
        ]

    def _check_domain_query(self, query: str) -> bool:
        from ..tools.financial_terms import is_financial_term
        if is_financial_term(query):
            return True
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.domain_keywords)

    def _is_definition_query(self, query: str) -> bool:
        q = (query or "").strip().lower()
        return bool(
            re.match(
                r"^(what is|what are|define|explain|meaning of|tell me about|how does)\b",
                q,
            )
        )

    def _extract_core_term(self, query: str) -> str:
        """Best-effort extraction of term/concept from definition-style prompts."""
        q = re.sub(r"\s+", " ", (query or "").strip()).strip("?.! ")
        if not q:
            return ""
        patterns = [
            r"^(what is|what are|define|explain|meaning of|tell me about)\s+",
            r"^(can you explain|please explain)\s+",
        ]
        lowered = q.lower()
        for p in patterns:
            m = re.match(p, lowered)
            if m:
                return q[m.end():].strip("?.! ")
        return q

    def _is_constraints_result(self, result: dict) -> bool:
        filename = str(result.get("filename", "")).strip().lower()
        title = str(result.get("title", "")).strip().lower()
        return filename == "constraints.txt" or title == "constraints"

    def _filter_substantive_kb_results(self, results: list[dict]) -> list[dict]:
        filtered: list[dict] = []
        for result in results or []:
            if not isinstance(result, dict):
                continue
            if self._is_constraints_result(result):
                continue
            if float(result.get("score", 0.0) or 0.0) < 0.25:
                continue
            if not str(result.get("content", "")).strip():
                continue
            filtered.append(result)
        return filtered

    def _extract_excerpt(self, text: str, max_chars: int = 420) -> str:
        cleaned = re.sub(r"```.*?```", " ", text or "", flags=re.DOTALL)
        cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.replace("**", " ").replace("`", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        chosen: list[str] = []
        total = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            projected = total + len(sentence) + (1 if chosen else 0)
            if projected > max_chars and chosen:
                break
            if projected > max_chars:
                chosen.append(sentence[:max_chars].rstrip(" ,;:") + "...")
                break
            chosen.append(sentence)
            total = projected
            if len(chosen) >= 3:
                break

        excerpt = " ".join(chosen).strip()
        if excerpt:
            return excerpt
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."

    def _build_context_grounded_fallback(
        self,
        query: str,
        kb_results: list[dict] | None = None,
        web_result: dict | None = None,
    ) -> str:
        term = self._extract_core_term(query) or query.strip() or "this topic"

        substantive_kb = self._filter_substantive_kb_results(kb_results or [])
        if substantive_kb:
            top = substantive_kb[0]
            excerpt = self._extract_excerpt(str(top.get("content", "")))
            title = str(top.get("title", "")).strip()
            if excerpt and title and title.lower() not in excerpt.lower():
                return f"{title}: {excerpt}"
            if excerpt:
                return excerpt

        web_text = ""
        if isinstance(web_result, dict):
            web_text = str(web_result.get("result", "") or "")
        elif isinstance(web_result, str):
            web_text = web_result
        excerpt = self._extract_excerpt(web_text)
        if excerpt and "search temporarily unavailable" not in excerpt.lower():
            return f"Here is the best available explanation for {term}: {excerpt}"

        return self._build_direct_no_search_fallback(
            query=query,
            is_conversational=False,
        )

    async def _call_primary_chat(
        self,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ):
        if (
            self._provider == "openrouter"
            and _HAS_OPENROUTER
            and settings.openrouter_available
            and call_openrouter is not None
        ):
            return await call_openrouter(
                self.model,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _build_direct_no_search_fallback(self, query: str, is_conversational: bool) -> str:
        """
        Deterministic non-empty fallback for direct no-search mode.
        Keeps responses helpful without forcing users to rephrase.
        """
        if is_conversational:
            return "Hello! I am here and ready to help with markets, stocks, and finance whenever you are."

        term = self._extract_core_term(query)
        if term:
            return (
                f"The phrase \"{term}\" does not clearly match a standard financial term yet. "
                "It may be a typo. Share the exact term and I will explain it in simple words with a practical example."
            )

        return (
            "A clear explanation can still be provided without live search. "
            "Share the exact term or question and I will break it down step by step."
        )

    async def analyze(self, state: AgentState) -> AgentState:
        query = state["query"]
        from ..tools.financial_terms import is_financial_term

        # ── Trust the router's decision completely ────────────────────────
        # The router already classified: mode, needs_search, use_kb, is_conversational.
        # We don't re-classify here — we just act on what the router decided.
        enable_search = state.get("enable_research_loop", True)
        is_conversational = state.get("is_conversational", False)

        # Pre-initialise here so the except handler always has these in scope,
        # regardless of which branch raises (fixes potential NameError).
        kb_context = ""
        web_context = ""
        structured_kb_results: list[dict] = []
        web_result_payload: dict | None = None

        if not enable_search:
            try:
                import asyncio as _asyncio

                # ── CASE 1: Pure conversational / off-topic ──────────────────────
                # Router flagged this as non-financial. Reply warmly; if off-topic,
                # acknowledge briefly and redirect to finance. No KB, no web search.
                if is_conversational:
                    conv_system = """You are Daddy's AI — a friendly Indian financial assistant.

The user just sent a message. It may be a greeting, thanks, small talk, OR a question that has nothing to do with finance (food, travel, cricket, general knowledge, etc.).

HOW TO RESPOND:
- Greeting / thanks / small talk → Reply warmly in 1-2 sentences. Let them know you are ready to help with stocks, markets, or investing.
- Off-topic question (food, sports, recipe, travel, etc.) → Acknowledge it briefly and warmly (1 sentence), then redirect to finance in 1 sentence. Sound friendly, NOT robotic or dismissive.

RULES:
- Maximum 2-3 sentences — never longer
- NEVER use bullet points, headers, bold, or markdown
- Always respond in English, even if the user wrote in Hindi or Hinglish
- Identity is strict: you are Daddy's AI for InvestingDaddy, created by Adarsh under Vinay Sir's guidance
- Never claim to be from OpenAI, Groq, Arcee, OpenRouter, or any model/provider

Examples:
User: "hi" → "Hey! 😊 What financial topic can I help you with today?"
User: "thanks" → "Happy to help! Ask me anything about markets or investing anytime. 🙌"
User: "gulab jamun kaise banate hai" → "Ha, gulab jamun sounds delicious! 😄 But cooking is a bit outside my lane — I specialise in Indian stocks, Nifty, and investing. Ask me anything about markets and I am all yours! 📈"
User: "who won IPL 2024" → "Great match! 🏏 But cricket is not my territory — I live in the stock market. Ask me about Nifty, fundamentals, or any investing concept! 📈"
"""
                    answer = ""
                    try:
                        resp = await _asyncio.wait_for(
                            self._call_primary_chat(
                                messages=[
                                    {"role": "system", "content": conv_system},
                                    {"role": "user", "content": query},
                                ],
                                temperature=0.8,
                                max_tokens=100,
                            ),
                            timeout=10.0,
                        )
                        answer = (resp.choices[0].message.content or "").strip()
                    except Exception as primary_err:
                        logger.warning(
                            f"Conversational primary client failed ({type(primary_err).__name__}: {primary_err}), trying Groq fallback"
                        )
                        try:
                            _groq = get_groq_client()
                            resp = await _asyncio.wait_for(
                                _groq.chat.completions.create(
                                    model="llama-3.1-8b-instant",
                                    messages=[
                                        {"role": "system", "content": conv_system},
                                        {"role": "user", "content": query},
                                    ],
                                    temperature=0.8,
                                    max_tokens=100,
                                ),
                                timeout=10.0,
                            )
                            answer = (resp.choices[0].message.content or "").strip()
                        except Exception as fallback_err:
                            logger.warning(
                                f"Conversational Groq fallback failed ({type(fallback_err).__name__}: {fallback_err})"
                            )
                    if not answer:
                        answer = self._build_direct_no_search_fallback(
                            query=query,
                            is_conversational=True,
                        )
                    state["final_response"] = answer
                    state["execution_metadata"] = {
                        "agent": "explainer",
                        "mode": "conversational",
                    }
                    logger.info(f"💬 Conversational reply (router-driven) for: {query[:50]!r}")
                    return state

                # ── CASE 2: Concept / educational question — no live data needed ──
                # For definition-style prompts, always try KB even if router did not set use_kb.
                kb_context = ""
                web_context = ""
                structured_kb_results: list[dict] = []
                web_result_payload: dict | None = None
                is_definition_query = self._is_definition_query(query)
                kb_requested = bool(state.get("use_kb", False) or is_definition_query)
                if kb_requested:
                    # ── Fast path: use async-prefetched KB context from workflow ──
                    prefetched = (state.get("kb_context") or "").strip()
                    if prefetched:
                        kb_context = "\n\n---\n📚 **KNOWLEDGE BASE CONTEXT:**\n" + prefetched
                        logger.info(f"⚡ Using async-prefetched KB: {len(prefetched)} chars")
                    else:
                        # ── Slow path: do KB search now (prefetch didn't run) ──────
                        try:
                            from ..rag.qdrant_kb import get_qdrant_rag
                            qdrant_rag = get_qdrant_rag()

                            # Smart KB search: extract focused topics before searching
                            search_queries = [query]  # fallback if extraction fails
                            try:
                                from ..config.key_rotator import get_groq_client as _get_groq_kb
                                _groq_kb = _get_groq_kb()
                                _extract_resp = await _asyncio.wait_for(
                                    _groq_kb.chat.completions.create(
                                        model="llama-3.1-8b-instant",
                                        messages=[
                                            {
                                                "role": "system",
                                                "content": (
                                                    "You are a search-query extractor for an LTP Calculator knowledge base.\n"
                                                    "KB topics: WTB, WTT, SOC (State of Confusion), Game of Percentage, "
                                                    "Shifting Pressure, Six Kinds of Reversal, Strong Support/Resistance, "
                                                    "COA Scenarios, Diversion, EOR, EOS, Natural Weakness, Color Codes, "
                                                    "Open Interest, Imaginary Line, Constraints, Weekly Range.\n"
                                                    "Extract 1-3 short, focused search queries that will find the right KB docs.\n"
                                                    "Return ONLY a JSON array of strings. No explanation.\n"
                                                    'Example: ["WTB at support", "shifting pressure rules"]'
                                                ),
                                            },
                                            {"role": "user", "content": query},
                                        ],
                                        temperature=0.0,
                                        max_tokens=80,
                                    ),
                                    timeout=5.0,
                                )
                                import json as _json_kb
                                raw_arr = (_extract_resp.choices[0].message.content or "").strip()
                                parsed_arr = _json_kb.loads(raw_arr)
                                if isinstance(parsed_arr, list) and parsed_arr:
                                    search_queries = [str(q) for q in parsed_arr if q][:3]
                                    logger.info(f"🔍 Smart KB queries: {search_queries}")
                            except Exception as _extract_err:
                                logger.debug(f"KB topic extraction skipped, using raw query: {_extract_err}")

                            # Search with each extracted query, dedup by filename
                            seen_files: set = set()
                            all_kb_results: list = []
                            for _sq in search_queries:
                                try:
                                    _res = qdrant_rag.search(_sq, top_k=2)
                                    for _r in (_res or []):
                                        _fname = _r.get("filename", _r.get("id", str(id(_r))))
                                        if _fname not in seen_files:
                                            seen_files.add(_fname)
                                            all_kb_results.append(_r)
                                except Exception:
                                    pass

                            if all_kb_results:
                                structured_kb_results = self._filter_substantive_kb_results(all_kb_results)
                                if structured_kb_results:
                                    kb_parts = []
                                    for r in structured_kb_results[:4]:  # cap at 4 docs
                                        kb_parts.append(
                                            f"## {r['title']} ({r['filename']})\n{r['content'][:1200]}"
                                        )
                                    kb_context = "\n\n---\n📚 **KNOWLEDGE BASE CONTEXT:**\n" + "\n\n".join(kb_parts)
                                    logger.info(f"📚 On-demand KB: {len(structured_kb_results)} docs injected for '{query[:50]}'")
                                else:
                                    logger.info(f"🔍 On-demand KB returned no substantive documents for '{query[:50]}'")
                        except Exception as kb_err:
                            logger.debug(f"KB lookup skipped: {kb_err}")


                # ── Web fallback when KB misses ─────────────────────────────────────
                # Fast path: use async-prefetched result from workflow
                if is_definition_query and not kb_context:
                    prefetched_web = (state.get("web_context") or "").strip()
                    if prefetched_web:
                        web_context = "\n\n---\n🌐 **WEB CONTEXT:**\n" + prefetched_web[:2000]
                        logger.info(f"⚡ Using async-prefetched web: {len(prefetched_web)} chars")
                    else:
                        # Slow path: own search (prefetch not available)
                        try:
                            import json as _json
                            core_term = self._extract_core_term(query) or query
                            web_result = await execute_tool(
                                "search_web",
                                {"query": f"{core_term} finance meaning"},
                            )
                            web_result_payload = web_result
                            web_context = (
                                "\n\n---\n🌐 **WEB CONTEXT:**\n"
                                + _json.dumps(web_result, default=str)[:2000]
                            )
                            logger.info(f"🌐 On-demand web fallback: '{query[:60]}'")
                        except Exception as web_err:
                            logger.debug(f"Web fallback skipped: {web_err}")

                # â”€â”€ Build system prompt (with or without KB context) â”€â”€â”€â”€â”€â”€â”€
                kb_section = ""
                if kb_context:
                    kb_section = f"""
{kb_context}

â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
IMPORTANT: Use the Knowledge Base Context above as your PRIMARY source for this answer.
Explain it clearly and accurately based strictly on the KB content.
STRICT RULES: Answer ONLY from the KB content above. Do NOT invent rules, percentages, or scenario names not present. Do NOT use general trading knowledge to fill gaps - LTP Calculator has unique theory. SOC = State of Confusion (NOT Shift of Control). Strong S/R = yellow box disappeared (not price rejection). If KB doesn't have the full answer, say what is covered and tell user to check the LTP Calculator course.
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
"""

                web_section = ""
                if web_context:
                    web_section = f"""
{web_context}

â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
IMPORTANT: KB confidence was low, so web context was fetched for grounding.
If the queried term seems misspelled, say that clearly and still help the user.
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
"""

                context_section = f"{kb_section}{web_section}"

                direct_system = f"""You are Daddy's AI — a senior financial analyst, educator, and storyteller built for InvestingDaddy, India's premier stock market education platform by Vinay Sir.

Answer this query with DEPTH, PRECISION, and NARRATIVE POWER.{context_section}

=== RESPONSE FORMAT — MANDATORY (WIKIPEDIA + ANALYST STYLE) ===

⛔ BULLETS ARE BANNED. Not a single hyphen (-), dot (●), or asterisk (*) used as a bullet point.
⛔ NEVER start a line with "- " or "● " or "• "
✅ WRITE IN FLOWING PARAGRAPHS under clear section headings.

STRUCTURE (adapt based on query type):

## [Title with key insight or number]
[2-3 sentence opening that delivers the MOST IMPORTANT insight immediately. Story-first.]

## Background
[Dense narrative paragraph. Historical context. Cause-and-effect told chronologically in prose.]

## How It Works / Key Mechanics
[Explain the concept/event clearly. Use real numbers, real examples, real names from web context if available.]

## 📊 Key Data (if comparing multiple things)
[ONE table max — side-by-side comparisons only]
| Metric | Value | Impact |
|--------|-------|--------|

## Practical Impact / What This Means for Investors
[Direct 2-3 sentence takeaway. Phase-based if needed: immediate → medium → long-term.]

⚠️ *Not financial advice. Consult a SEBI-registered advisor.*

FOR SHORT QUERIES: 2-3 paragraphs, no headings needed.
LENGTH: 250-500 words. Precise beats padded.

GUARDRAILS:
- Do NOT start with "I" as the first word
- Do NOT use generic openers like "Great question!" or "Certainly!"
- Do NOT mention live data limitations
- Do NOT claim to be made by OpenAI, Groq, Arcee, or any model provider
- Use ₹, lakhs, crores for Indian context; $ for global topics
- If web context is available above, cite real examples and numbers from it naturally
- Identity: you are Daddy's AI for InvestingDaddy"""

                # ── LLM call: primary model first, Groq instant fallback ──────
                # Inject conversation history so follow-up questions have context
                _conv_history = state.get("conversation_history", []) or []
                _history_msgs = []
                for _msg in _conv_history[-6:]:  # last 6 turns (3 exchanges)
                    _role = _msg.get("role", "user")
                    _body = _msg.get("content", "")
                    if _role in ("user", "assistant") and _body:
                        _history_msgs.append({"role": _role, "content": str(_body)[:800]})
                messages_payload = [
                    {"role": "system", "content": direct_system},
                    *_history_msgs,
                    {"role": "user", "content": query},
                ]
                resp = None
                try:
                    resp = await _asyncio.wait_for(
                        self._call_primary_chat(
                            messages=messages_payload,
                            temperature=0.7,
                            max_tokens=min(self.max_tokens, 1800),
                        ),
                        timeout=16.0,
                    )
                    logger.info(
                        f"âš¡ Direct reply via primary model (kb={bool(kb_context)}, web_fallback={bool(web_context)})"
                    )
                except Exception as primary_err:
                    logger.warning(
                        f"Direct reply primary model failed ({type(primary_err).__name__}: {primary_err}), retrying with Groq fast model"
                    )
                    # Retry with Groq llama-3.1-8b-instant (fast, never times out at 8s)
                    try:
                        from ..config.key_rotator import get_groq_client as _get_groq
                        _groq_fb = _get_groq()
                        resp = await _asyncio.wait_for(
                            _groq_fb.chat.completions.create(
                                model="llama-3.1-8b-instant",
                                messages=messages_payload,
                                temperature=0.6,
                                max_tokens=min(self.max_tokens, 1200),
                            ),
                            timeout=10.0,
                        )
                        logger.info("Direct reply recovered via Groq fast fallback")
                    except Exception as groq_fb_err:
                        logger.warning(f"Groq fast fallback also failed: {groq_fb_err}")
                        resp = None

                answer = (resp.choices[0].message.content or "").strip() if resp else ""
                if not answer and resp is not None:
                    logger.warning("Direct reply returned empty content, trying strict non-empty retry")
                    try:
                        retry_resp = await _asyncio.wait_for(
                            self._call_primary_chat(
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "Return a non-empty helpful answer. "
                                            "If the term is unclear or misspelled, say that briefly and ask for exact spelling."
                                        ),
                                    },
                                    {"role": "user", "content": query},
                                ],
                                temperature=0.3,
                                max_tokens=min(self.max_tokens, 320),
                            ),
                            timeout=8.0,
                        )
                        answer = (retry_resp.choices[0].message.content or "").strip()
                    except Exception as retry_err:
                        logger.warning(
                            f"Direct reply non-empty retry failed ({type(retry_err).__name__}: {retry_err})"
                        )
                if not answer:
                    answer = self._build_context_grounded_fallback(
                        query=query,
                        kb_results=structured_kb_results,
                        web_result=web_result_payload,
                    )
                state["final_response"] = answer
                state["execution_metadata"] = {
                    "agent": "explainer",
                    "mode": "direct_reply",
                    "kb_used": bool(kb_context),
                    "web_fallback_used": bool(web_context),
                }
                return state
            except Exception as e:
                # Router explicitly said no search. Never escalate into tool-search mode here.
                logger.warning(
                    f"Direct no-search reply failed ({type(e).__name__}: {e}), using deterministic no-search fallback"
                )
                if is_conversational:
                    fallback = self._build_direct_no_search_fallback(
                        query=query,
                        is_conversational=True,
                    )
                else:
                    fallback = self._build_context_grounded_fallback(
                        query=query,
                        kb_results=structured_kb_results,
                        web_result=web_result_payload,
                    )
                state["final_response"] = fallback
                state["execution_metadata"] = {
                    "agent": "explainer",
                    "mode": "direct_reply_fallback",
                    "kb_used": bool(structured_kb_results),
                    "web_fallback_used": bool(web_result_payload),
                    "no_search_enforced": True,
                }
                return state


        # --- Knowledge Base pre-fetch for domain terms ---
        kb_context = ""
        kb_sources = []
        is_domain_query = self._check_domain_query(query) or is_financial_term(query)

        if is_domain_query:
            results = self.kb_rag.search(query, top_k=2)
            if results:
                logger.info(f"ðŸ“š Found {len(results)} KB files for query")
                kb_sources = [f"{r['title']} ({r['filename']})" for r in results]
                kb_parts = []
                for result in results:
                    kb_parts.append(f"\n## Source: {result['title']} ({result['filename']})\n{result['content'][:1500]}")
                kb_context = "\n---\n**Knowledge Base Context:**\n" + "\n".join(kb_parts)

        # --- Build system prompt ---
        # When KB content is available, embed it IN the system prompt with a hard lock.
        # This prevents the model from ignoring the context and hallucinating.
        if kb_context:
            system_prompt = f"""ðŸŒ LANGUAGE RULE: ALWAYS respond in English. Never respond in Hindi, Chinese, or any other language, even if the query appears to be in another language.

You are Daddy's AI â€” a precise financial teacher built by Adarsh under Vinay Sir's guidance (InvestingDaddy).

â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘  ðŸ”’  KNOWLEDGE BASE LOCK â€” MANDATORY READING BEFORE YOU ANSWER  â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

The following content is from InvestingDaddy's proprietary knowledge base.
This is the ONLY authoritative source for this topic.

RULES:
1. Answer ONLY using the knowledge base content below.
2. Do NOT use your own training data for the core definition/explanation.
3. Do NOT substitute a different topic (e.g., if asked about WTB, answer about WTB â€” not Demat, not general trading).
4. If the KB content is sufficient â†’ answer directly from it.
5. If KB content is partial â†’ answer what's there, then say "For more detail, here's what I know..." and add general context.
6. NEVER say "I don't have information" if the content below covers the topic.

â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
ðŸ“š KNOWLEDGE BASE CONTENT (SOURCE OF TRUTH):
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
{kb_context}
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

=== PERSONALITY & FORMAT ===
Warm, direct, knowledgeable teacher. Mirror user's language (Hinglish if needed). Use â‚¹ for Indian currency.

RESPONSE FORMAT:
- Flowing paragraphs under ## headings for structured topics
- Tables for comparisons (NOT bullet points)
- Carousels for step-by-step content
- 2-4 emojis naturally integrated
- âŒ ZERO bullet points (use tables or prose instead)
- Lead with the most important fact from the KB content

=== IDENTITY ===
Creator: Adarsh (Class 8, Daddy's International School, Chandauli) under Dr. Vinay Prakash Tiwari (Vinay Sir), Founder of InvestingDaddy."""
        else:
            system_prompt = f"""ðŸŒ LANGUAGE RULE: ALWAYS respond in English. Never respond in Hindi, Chinese, or any other language.

You are Daddy's AI â€” a brilliant financial teacher who makes complex concepts crystal clear.

=== CORE IDENTITY ===
**Creator**: Adarsh, a Class 8 student at Daddy's International School, Chandauli
**Founder**: Dr. Vinay Prakash Tiwari (Vinay Sir) â€” Founder of InvestingDaddy
**Platform**: InvestingDaddy â€” India's premier stock market education platform

When asked who created you, proudly share that Adarsh built this under Vinay Sir's guidance.

=== YOUR INTELLIGENCE ===
You're the teacher everyone wishes they had â€” knowledgeable, patient, engaging.
You take complex financial concepts and make them feel simple using real-world analogies,
Indian market examples (Reliance, TCS, HDFC), and a warm conversational tone.

=== PERSONALITY ===
Warm, direct, confident. Talk like a human â€” never robotic. Mirror user's language (Hinglish if they use it).
Simple question â†’ simple answer. Use â‚¹ for Indian currency.

=== AUTONOMOUS TOOL PROTOCOL ===
Before answering, ask: "Do I need to verify this or get fresh data?"

**WHEN TO USE TOOLS:**
â€¢ "latest", "today", "current" â†’ search_web("[topic] India")
â€¢ Financial term not in memory â†’ search_knowledge_base("[term]")
â€¢ "price of X" â†’ fetch_nse_quote("SYMBOL")
â€¢ Company earnings â†’ search_financial_news("[company]")

**ðŸš¨ FALLBACK:** If ANY tool fails â†’ search_web("[query] India").

=== HOW TO CALL A TOOL ===
<tool_call>
{{"tool": "TOOL_NAME", "arguments": {{"param": "value"}}}}
</tool_call>

Tools: search_knowledge_base, search_web, fetch_nse_quote, search_financial_news, get_stock_fundamentals, get_market_sentiment, get_technical_indicators

=== RESPONSE FORMAT ===
- Flowing paragraphs with ## headings for structured topics
- Tables for comparisons (NOT bullet points)
- Carousels for step-by-step content
- âŒ NO bullet points anywhere
- âœ… 2-4 emojis naturally integrated"""

        messages = [{"role": "system", "content": system_prompt}]
        conversation_history = state.get("conversation_history", [])
        if conversation_history:
            for msg in conversation_history[-6:]:
                m = dict(msg)
                m.pop("images", None)
                messages.append({"role": m["role"], "content": m.get("content", "")})

        if kb_context:
            # KB content is already in system prompt â€” just send the query
            user_message = (
                f"User Query: {query}\n\n"
                "Answer this question using ONLY the knowledge base content provided in your system prompt. "
                "Write in flowing paragraphs with ## headings. Zero bullet points."
            )
        else:
            user_message = (
                f"User Query: {query}\n\n"
                "Provide a clear explanation. USE TOOLS autonomously if you need more info."
            )

        # â”€â”€ Handle IMAGE queries with dedicated vision analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        images = state.get("images") or []
        if images:
            logger.info(f"ðŸ–¼ï¸ Explainer using dedicated VISION mode for {len(images)} image(s)")

            # Build vision-specific user message incorporating the user's question
            if query and query.strip():
                vision_user_msg = (
                    f"**User's question:** {query}\n\n"
                    "Please analyze the financial image(s) attached above. "
                    "Read all numbers precisely, identify chart patterns or data tables, "
                    "and give a concrete, actionable analysis as per your instructions."
                )
            else:
                vision_user_msg = (
                    "Please analyze the financial image(s) attached above. "
                    "Identify what type of image it is (option chain, chart, P&L, dashboard, etc.), "
                    "extract all visible data precisely, and provide a complete analysis "
                    "with your market bias and key levels."
                )

            vision_messages = [{"role": "system", "content": VISION_SYSTEM_PROMPT}]

            # Add conversation context (text only, no images from history)
            conversation_history = state.get("conversation_history", [])
            for msg in conversation_history[-4:]:
                m = dict(msg)
                m.pop("images", None)
                if m.get("content"):
                    vision_messages.append({"role": m["role"], "content": m["content"]})

            # Attach images + question as user turn
            content_parts = [{"type": "text", "text": vision_user_msg}]
            for img in images[:5]:
                url = img if img.startswith("data:") else f"data:image/jpeg;base64,{img}"
                content_parts.append({"type": "image_url", "image_url": {"url": url}})
            vision_messages.append({"role": "user", "content": content_parts})

            try:
                vision_response = await self._groq_client.chat.completions.create(
                    model=settings.model_vision,
                    messages=vision_messages,
                    temperature=0.1,   # Low temperature = precise, consistent reads
                    max_tokens=2000
                )
                state["final_response"] = vision_response.choices[0].message.content or "Could not analyze the image. Please try again."
                state["execution_metadata"] = {
                    "agent": "explainer",
                    "mode": "vision_financial_analyst",
                    "model": settings.model_vision,
                    "images_analyzed": len(images)
                }
                logger.info("âœ… Vision analysis complete")
            except Exception as ve:
                logger.error(f"Vision analysis failed: {ve}")
                state["final_response"] = "I had trouble reading the image. Could you describe what's in the screenshot so I can help you analyze it?"
                state["execution_metadata"] = {"agent": "explainer", "mode": "vision_error", "skip_verifier": True}
            return state

        # â”€â”€ TEXT queries: use the standard explainer prompt with tools â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        tools_used = []
        max_tool_rounds = 4
        active_temperature = 0.3 if kb_context else self.temperature

        try:
            for round_num in range(max_tool_rounds + 1):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=active_temperature,
                    max_tokens=self.max_tokens
                )
                content = response.choices[0].message.content or ""

                tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)

                if not tool_call_match:
                    explanation = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
                    if explanation and len(explanation) > 30:
                        state["final_response"] = explanation
                    else:
                        state["final_response"] = content.strip() or "I apologize, I couldn't generate a proper explanation. Could you rephrase?"
                    break

                try:
                    tc = json.loads(tool_call_match.group(1))
                    tool_name = tc.get("tool", "")
                    tool_args = tc.get("arguments", {})
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool call JSON, treating as final response")
                    state["final_response"] = content.strip()
                    break

                if tool_name not in EXPLAINER_TOOLS:
                    logger.warning(f"Explainer requested unknown tool: {tool_name}, trying search_web instead")
                    tool_name = "search_web"
                    tool_args = {"query": f"{query} India"}

                logger.info(f"ðŸ› ï¸ Explainer autonomous tool: {tool_name}({tool_args})")
                tools_used.append(tool_name)

                try:
                    result = await execute_tool(tool_name, tool_args)
                except Exception as tool_err:
                    logger.warning(f"Tool {tool_name} failed: {tool_err}, trying search_web")
                    try:
                        result = await execute_tool("search_web", {"query": f"{tool_args.get('query', tool_args.get('symbol', query))} India"})
                        tools_used.append("search_web")
                    except Exception:
                        result = {"error": f"Tool {tool_name} failed and search_web also failed"}

                result_str = json.dumps(result, default=str)[:2000]

                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"[Tool Result for {tool_name}]:\n{result_str}\n\nNow synthesize and provide your complete answer. Cite this data when relevant."})

            else:
                final_text = re.sub(r'<tool_call>.*?</tool_call>', '', content or '', flags=re.DOTALL).strip()
                state["final_response"] = final_text or "I gathered some data but couldn't fully synthesize. Please try a more specific question."

            state["execution_metadata"] = {
                "agent": "explainer",
                "model": self.model,
                "used_knowledge_base": bool(kb_context),
                "autonomous_tools_used": tools_used,
                "tool_rounds": len(tools_used)
            }
            logger.info(f"âœ… Explainer: KB={bool(kb_context)}, Tools={tools_used}")
            return state

        except Exception as e:
            logger.error(f"Explainer agent error: {e}")
            # Last resort: try search_web
            try:
                web_result = await execute_tool("search_web", {"query": f"{query} explained India"})
                fallback_msgs = [
                    {"role": "system", "content": "You are a friendly financial teacher. Explain this topic clearly using the search results."},
                    {"role": "user", "content": f"Query: {query}\n\nSearch results:\n{json.dumps(web_result, default=str)[:3000]}"}
                ]
                resp = await self.client.chat.completions.create(
                    model=self.model, messages=fallback_msgs,
                    temperature=0.4, max_tokens=self.max_tokens
                )
                state["final_response"] = resp.choices[0].message.content
                state["execution_metadata"] = {"agent": "explainer", "fallback": "search_web", "skip_verifier": True}
            except Exception as e2:
                logger.error(f"Explainer fallback also failed: {e2}")
                state["error"] = str(e)
                state["final_response"] = "I encountered an error while generating the explanation. Please try again."
            return state

