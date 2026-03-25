"""
Explainer Agent — Educational explanations with FULL autonomous tool usage.
Uses ReAct-style loop: model outputs <tool_call>...</tool_call>, we parse & execute.
Falls back to search_web when other tools fail.
"""

from groq import AsyncGroq
import logging
import re
import json

from ..config import settings, ModelType

try:
    from ..config.openrouter_client import get_openrouter_client
    _HAS_OPENROUTER = True
except ImportError:
    _HAS_OPENROUTER = False

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

# ── DEDICATED VISION SYSTEM PROMPT ─────────────────────────────────────────────
# Used exclusively when the user sends an image (chart, screenshot, table, etc.)
# Instructs the model to act as a precision financial analyst reading the image numerically.
VISION_SYSTEM_PROMPT = """You are Daddy's AI — a precision financial image analyst, built by Adarsh under the guidance of Vinay Sir (InvestingDaddy).

You have one job when given a financial image: **read it with precision and give a concrete, actionable analysis**.

══ STEP 1: IDENTIFY THE IMAGE TYPE ══

First, identify what you're looking at:
- **LTP Table / Option Chain** — rows of strikes with LTP, OI, OI Chg, Volume, WTB%, WTT%, IV, etc.
- **Candlestick Chart** — price action over time, possibly with indicators (EMA, RSI, VWAP, etc.)
- **Market Dashboard / Index Heatmap** — index prices, sector performance, top gainers/losers
- **P&L / Holdings Screenshot** — profit/loss per position
- **News / Text Screenshot** — headline or article
- **Other** — describe accurately

══ STEP 2: EXTRACT ALL NUMBERS FIRST ══

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

══ STEP 3: GIVE A PRECISE ANALYSIS ══

**You MUST:**
✔ Use the ACTUAL NUMBERS from the image — never be vague
✔ State your bias CLEARLY: Bullish / Bearish / Neutral with reasoning
✔ Give a specific price level for support and resistance (not "around the ATM")
✔ For OI data: interpret buildup (Long/Short buildup, Long/Short unwinding)
✔ For WTB/WTT data: state exactly what the % implies about directional pressure
✔ For charts: identify the setup and what confirmation would trigger a trade
✔ End with a clear 1-line market view: "Summary: [Bullish/Bearish/Sideways] bias with support at [X] and resistance at [Y]. Watch for [specific trigger]."

**You MUST NOT:**
❌ Say "it appears bearish" without citing specific numbers from the image
❌ Say "you might want to look at RSI" if RSI isn't visible in the image  
❌ Give generic advice that could apply to any market (e.g., "analyze additional data like RSI, Bollinger Bands")
❌ Suggest web searches or ask for more data — analyze WHAT IS IN FRONT OF YOU
❌ Use bullet points with hedge words ("seems", "appears", "might be", "could indicate")

══ RESPONSE FORMAT ══

Structure your response as:

**📸 Image Type:** [What you identified]

**📅 Key Data Extracted:**
[Use a markdown table for LTP/option chain data]
[Use clear labeled points for chart levels]

**📊 Analysis:**
[2-3 paragraphs: data interpretation → what it means → market structure]

**⚡ Signals:**
[Table of: Signal | Value | Implication]

**🎯 Summary:**
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

    async def analyze(self, state: AgentState) -> AgentState:
        query = state["query"]
        from ..tools.financial_terms import is_financial_term

        # ── Trust the router's decision completely ─────────────────────────────
        # The router already classified: mode, needs_search, use_kb, is_conversational.
        # We don't re-classify here — we just act on what the router decided.
        enable_search = state.get("enable_research_loop", True)
        is_conversational = state.get("is_conversational", False)

        if not enable_search:
            try:
                import asyncio as _asyncio

                # ── CASE 1: Pure conversational (greeting / small talk / thanks) ──
                # Router flagged this as small talk. Reply with a warm 1-2 sentence response.
                # NO KB lookup, NO web search, NO thought process shown.
                if is_conversational:
                    conv_system = """You are Daddy's AI — a friendly Indian financial assistant.

The user just sent a casual message (greeting, thanks, small talk, etc.).
Reply warmly and naturally in 1-2 sentences MAX.
- Be friendly and personable
- Mention you're ready to help with stocks, markets, or finance
- NEVER explain concepts, NEVER use headers, NEVER write paragraphs
- Keep it casual, like a knowledgeable friend saying hello back

Examples of good responses:
- "Hey! 😊 Doing great — what financial topic can I help you with today?"
- "Hello there! Ready to dive into stocks, Nifty, or any investing concept whenever you are. 📈"
- "Thanks! Happy to help. Ask me anything about markets or investing. 🙌"
"""
                    from ..config.key_rotator import get_groq_client as _get_groq
                    _groq = _get_groq()
                    resp = await _asyncio.wait_for(
                        _groq.chat.completions.create(
                            model="llama-3.1-8b-instant",  # Fast model — it's just a greeting
                            messages=[
                                {"role": "system", "content": conv_system},
                                {"role": "user", "content": query},
                            ],
                            temperature=0.8,
                            max_tokens=80,  # Hard cap — keep it SHORT
                        ),
                        timeout=10.0,
                    )
                    answer = (resp.choices[0].message.content or "").strip()
                    state["final_response"] = answer
                    state["execution_metadata"] = {
                        "agent": "explainer",
                        "mode": "conversational",
                        "skip_verifier": True,
                    }
                    logger.info(f"💬 Conversational reply (router-driven) for: {query[:50]!r}")
                    return state

                # ── CASE 2: Concept / educational question — no live data needed ──
                # Query Qdrant KB if router decided it's needed (use_kb=True)
                kb_context = ""
                if state.get("use_kb", False):
                    try:
                        from ..rag.qdrant_kb import get_qdrant_rag
                        qdrant_rag = get_qdrant_rag()
                        kb_results = qdrant_rag.search(query, top_k=2)
                        if kb_results:
                            good_results = [r for r in kb_results if r.get("score", 1.0) >= 0.25]
                            if good_results:
                                kb_parts = []
                                for r in good_results:
                                    kb_parts.append(
                                        f"## {r['title']} ({r['filename']})\n{r['content'][:1200]}"
                                    )
                                kb_context = "\n\n---\n📚 **KNOWLEDGE BASE CONTEXT (use this first):**\n" + "\n\n".join(kb_parts)
                                logger.info(f"📚 Direct-reply KB: {len(good_results)} results injected for '{query[:50]}'")
                    except Exception as kb_err:
                        logger.debug(f"KB lookup skipped: {kb_err}")

                # ── Build system prompt (with or without KB context) ───────
                kb_section = ""
                if kb_context:
                    kb_section = f"""
{kb_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT: Use the Knowledge Base Context above as your PRIMARY source for this answer.
Expand on it, add Indian market examples, and make it easy to understand.
If the KB doesn't fully cover the topic, supplement with your own knowledge.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

                direct_system = f"""You are Daddy's AI — an intelligent financial educator and assistant built for InvestingDaddy, an Indian financial education platform founded by Vinay Sir.

You have been routed here because this query does NOT need live market data. Answer directly from your knowledge.{kb_section}

════════════════════════════════════════
HOW TO HANDLE DIFFERENT QUERY TYPES:
════════════════════════════════════════

1. GREETINGS / SMALL TALK (any language — Hindi, Hinglish, English, etc.)
   → Respond warmly and naturally in English. Keep it short (2-3 sentences).
   → Mention you're ready to help with stocks, markets, or financial concepts.
   → Examples: "how are you" → "Doing great! 😊 Ready to help with stocks, charts, or any financial concept."

2. CONCEPT / DEFINITION QUESTIONS (what is X, explain X, teach me X, how does X work)
   → Teach clearly, progressively, like a mentor explaining to a serious learner.
   → Structure your answer:
      • Start with a 1-line crisp definition
      • Explain it in simple terms with a real Indian market example
      • Add a practical use case (how traders/investors use it)
      • End with a pro tip or common mistake to avoid
   → Use Indian examples: Reliance, Nifty, TCS, Infosys, HDFC, NSE, BSE, Zerodha, etc.
   → For indicators (RSI, MACD, Supertrend, Bollinger Bands):
      Explain: what it measures → how it's calculated (simply) → how to use it → signal interpretation
   → Length: 150-400 words for concept explanations.

3. SELF-INTRODUCTION (who are you, what can you do, tell me about yourself)
   → Introduce yourself as Daddy's AI, built for InvestingDaddy by Adarsh under Vinay Sir's guidance.
   → Explain your capabilities: real-time stock data, fundamental analysis, technical indicators, financial education, portfolio advice.

4. GENERAL FINANCIAL EDUCATION (how to invest, types of mutual funds, stock market basics)
   → Teach like an experienced mentor, not a textbook.
   → Always relate to Indian markets and instruments (NSE, BSE, SEBI, Zerodha, Groww, etc.)

════════════════════════════════════════
FORMATTING RULES:
════════════════════════════════════════
• Always respond in English (even if query is in Hindi/Hinglish)
• Use markdown: **bold** for key terms, bullet points for structured answers
• For greetings: conversational, no headers needed
• For teaching: structured paragraphs, bullet points, occasional emoji for warmth
• Do NOT mention live prices, search results, or "I don't have current data"
• Do NOT start with "I" — start with a strong statement or the answer itself"""

                # ── LLM call: Groq first (fast), OpenRouter fallback ──────
                messages_payload = [
                    {"role": "system", "content": direct_system},
                    {"role": "user", "content": query},
                ]
                resp = None
                try:
                    from ..config.key_rotator import get_groq_client as _get_groq
                    _groq = _get_groq()
                    resp = await _asyncio.wait_for(
                        _groq.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=messages_payload,
                            temperature=0.7,
                            max_tokens=700,
                        ),
                        timeout=20.0,
                    )
                    logger.info(f"⚡ Direct reply via Groq (kb={bool(kb_context)})")
                except Exception as groq_err:
                    logger.warning(f"Direct reply Groq failed ({type(groq_err).__name__}: {groq_err}), using OpenRouter")
                    resp = await _asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            messages=messages_payload,
                            temperature=0.7,
                            max_tokens=700,
                        ),
                        timeout=30.0,
                    )
                    logger.info(f"⚡ Direct reply via OpenRouter fallback (kb={bool(kb_context)})")

                answer = (resp.choices[0].message.content or "").strip()
                state["final_response"] = answer
                state["execution_metadata"] = {
                    "agent": "explainer",
                    "mode": "direct_reply",
                    "kb_used": bool(kb_context),
                    "skip_verifier": True,
                }
                return state
            except Exception as e:
                logger.warning(f"Direct reply failed ({type(e).__name__}: {e}), falling through to full pipeline")
                # Fall through to the normal explainer pipeline below


        # --- Knowledge Base pre-fetch for domain terms ---
        kb_context = ""
        kb_sources = []
        is_domain_query = self._check_domain_query(query) or is_financial_term(query)

        if is_domain_query:
            results = self.kb_rag.search(query, top_k=2)
            if results:
                logger.info(f"📚 Found {len(results)} KB files for query")
                kb_sources = [f"{r['title']} ({r['filename']})" for r in results]
                kb_parts = []
                for result in results:
                    kb_parts.append(f"\n## Source: {result['title']} ({result['filename']})\n{result['content'][:1500]}")
                kb_context = "\n---\n**Knowledge Base Context:**\n" + "\n".join(kb_parts)

        # --- Build system prompt ---
        # When KB content is available, embed it IN the system prompt with a hard lock.
        # This prevents the model from ignoring the context and hallucinating.
        if kb_context:
            system_prompt = f"""🌐 LANGUAGE RULE: ALWAYS respond in English. Never respond in Hindi, Chinese, or any other language, even if the query appears to be in another language.

You are Daddy's AI — a precise financial teacher built by Adarsh under Vinay Sir's guidance (InvestingDaddy).

╔══════════════════════════════════════════════════════════════════╗
║  🔒  KNOWLEDGE BASE LOCK — MANDATORY READING BEFORE YOU ANSWER  ║
╚══════════════════════════════════════════════════════════════════╝

The following content is from InvestingDaddy's proprietary knowledge base.
This is the ONLY authoritative source for this topic.

RULES:
1. Answer ONLY using the knowledge base content below.
2. Do NOT use your own training data for the core definition/explanation.
3. Do NOT substitute a different topic (e.g., if asked about WTB, answer about WTB — not Demat, not general trading).
4. If the KB content is sufficient → answer directly from it.
5. If KB content is partial → answer what's there, then say "For more detail, here's what I know..." and add general context.
6. NEVER say "I don't have information" if the content below covers the topic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 KNOWLEDGE BASE CONTENT (SOURCE OF TRUTH):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{kb_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

=== PERSONALITY & FORMAT ===
Warm, direct, knowledgeable teacher. Mirror user's language (Hinglish if needed). Use ₹ for Indian currency.

RESPONSE FORMAT:
- Flowing paragraphs under ## headings for structured topics
- Tables for comparisons (NOT bullet points)
- Carousels for step-by-step content
- 2-4 emojis naturally integrated
- ❌ ZERO bullet points (use tables or prose instead)
- Lead with the most important fact from the KB content

=== IDENTITY ===
Creator: Adarsh (Class 8, Daddy's International School, Chandauli) under Dr. Vinay Prakash Tiwari (Vinay Sir), Founder of InvestingDaddy."""
        else:
            system_prompt = f"""🌐 LANGUAGE RULE: ALWAYS respond in English. Never respond in Hindi, Chinese, or any other language.

You are Daddy's AI — a brilliant financial teacher who makes complex concepts crystal clear.

=== CORE IDENTITY ===
**Creator**: Adarsh, a Class 8 student at Daddy's International School, Chandauli
**Founder**: Dr. Vinay Prakash Tiwari (Vinay Sir) — Founder of InvestingDaddy
**Platform**: InvestingDaddy — India's premier stock market education platform

When asked who created you, proudly share that Adarsh built this under Vinay Sir's guidance.

=== YOUR INTELLIGENCE ===
You're the teacher everyone wishes they had — knowledgeable, patient, engaging.
You take complex financial concepts and make them feel simple using real-world analogies,
Indian market examples (Reliance, TCS, HDFC), and a warm conversational tone.

=== PERSONALITY ===
Warm, direct, confident. Talk like a human — never robotic. Mirror user's language (Hinglish if they use it).
Simple question → simple answer. Use ₹ for Indian currency.

=== AUTONOMOUS TOOL PROTOCOL ===
Before answering, ask: "Do I need to verify this or get fresh data?"

**WHEN TO USE TOOLS:**
• "latest", "today", "current" → search_web("[topic] India")
• Financial term not in memory → search_knowledge_base("[term]")
• "price of X" → fetch_nse_quote("SYMBOL")
• Company earnings → search_financial_news("[company]")

**🚨 FALLBACK:** If ANY tool fails → search_web("[query] India").

=== HOW TO CALL A TOOL ===
<tool_call>
{{"tool": "TOOL_NAME", "arguments": {{"param": "value"}}}}
</tool_call>

Tools: search_knowledge_base, search_web, fetch_nse_quote, search_financial_news, get_stock_fundamentals, get_market_sentiment, get_technical_indicators

=== RESPONSE FORMAT ===
- Flowing paragraphs with ## headings for structured topics
- Tables for comparisons (NOT bullet points)
- Carousels for step-by-step content
- ❌ NO bullet points anywhere
- ✅ 2-4 emojis naturally integrated"""

        messages = [{"role": "system", "content": system_prompt}]
        conversation_history = state.get("conversation_history", [])
        if conversation_history:
            for msg in conversation_history[-6:]:
                m = dict(msg)
                m.pop("images", None)
                messages.append({"role": m["role"], "content": m.get("content", "")})

        if kb_context:
            # KB content is already in system prompt — just send the query
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

        # ── Handle IMAGE queries with dedicated vision analysis ────────────────────────────
        images = state.get("images") or []
        if images:
            logger.info(f"🖼️ Explainer using dedicated VISION mode for {len(images)} image(s)")

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
                logger.info("✅ Vision analysis complete")
            except Exception as ve:
                logger.error(f"Vision analysis failed: {ve}")
                state["final_response"] = "I had trouble reading the image. Could you describe what's in the screenshot so I can help you analyze it?"
                state["execution_metadata"] = {"agent": "explainer", "mode": "vision_error"}
            return state

        # ── TEXT queries: use the standard explainer prompt with tools ──────────────────

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

                logger.info(f"🛠️ Explainer autonomous tool: {tool_name}({tool_args})")
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
            logger.info(f"✅ Explainer: KB={bool(kb_context)}, Tools={tools_used}")
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
                state["execution_metadata"] = {"agent": "explainer", "fallback": "search_web"}
            except Exception as e2:
                logger.error(f"Explainer fallback also failed: {e2}")
                state["error"] = str(e)
                state["final_response"] = "I encountered an error while generating the explanation. Please try again."
            return state
