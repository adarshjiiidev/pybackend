"""
Deep Reasoning Agent using GPT-OSS-120B.
Provides step-by-step reasoning with high effort for complex financial analysis.
Supports multi-step tool calling for comprehensive deep research.
"""

from typing import Optional, Any
import logging
import asyncio
import json

from ..config import settings, ModelType

try:
    from ..config.openrouter_client import get_openrouter_client
    _HAS_OPENROUTER = True
except ImportError:
    _HAS_OPENROUTER = False

from ..config.key_rotator import get_groq_client
from ..models.agent_state import AgentState
from ..tools import get_tool_definitions, execute_tool

logger = logging.getLogger(__name__)


class DeepReasoningAgent:
    """
    Deep reasoning using GPT-OSS-120B with high reasoning effort.
    Perfect for complex market analysis requiring multi-step thinking.
    Supports sequential tool calling: news → fundamentals → technical → synthesis.
    """

    def __init__(self):
        # OpenRouter first priority (trinity-large for deep research), Groq GPT-OSS fallback
        if _HAS_OPENROUTER and settings.openrouter_available:
            self.client = get_openrouter_client()
            self.model = settings.get_openrouter_model(ModelType.REASONING_DEEP)
            self._provider = "openrouter"
        else:
            self.client = get_groq_client()
            self.model = settings.get_model_for_task(ModelType.REASONING_DEEP)
            self._provider = "groq"
        self.temperature = settings.get_temperature_for_task(ModelType.REASONING_DEEP)
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.REASONING_DEEP)
        self.reasoning_effort = settings.get_reasoning_effort(ModelType.REASONING_DEEP) if self._provider == "groq" else None
        self.tools = get_tool_definitions() if settings.enable_tool_calling else []

    def _build_search_query(self, query: str, entities: dict = None) -> str:
        """
        Build a focused web search query from extracted entities.
        """
        parts = []
        symbols = (entities or {}).get("symbols", [])

        if symbols:
            parts.append(" ".join(symbols))

        filler = {"i", "you", "me", "my", "we", "the", "a", "an", "is", "are", "was",
                  "were", "be", "been", "being", "have", "has", "had", "do", "does",
                  "did", "will", "would", "could", "should", "may", "might", "shall",
                  "can", "need", "want", "to", "of", "in", "for", "on", "with", "at",
                  "by", "from", "it", "its", "that", "this", "these", "those", "not",
                  "or", "and", "but", "if", "so", "tell", "give", "whether", "about",
                  "please", "deeply", "detailed", "comprehensive", "hey", "hi", "hello"}

        topic_words = [w for w in query.lower().split() if w not in filler and len(w) > 2]
        if topic_words:
            parts.append(" ".join(topic_words[:5]))

        # For non-stock queries don't blindly add "India stock market"
        if symbols:
            parts.append("India stock market")

        search_q = " ".join(parts)
        return search_q[:200]

    async def _race_llm_call(self, call_params: dict, iteration: int):
        """
        Race OpenRouter vs Groq: fire OpenRouter first. If it takes >4s with
        no response, fire Groq (llama-3.3-70b) in parallel. Return whichever
        finishes first and cancel the other.

        This prevents 60s stalls from slow free OpenRouter models.
        """
        from groq import AsyncGroq as _AsyncGroq
        from ..config.key_rotator import get_groq_client as _get_groq_client

        # ── Build Groq fallback params (strip OpenRouter-specific keys) ───
        groq_params = {
            "model": "llama-3.3-70b-versatile",
            "messages": call_params["messages"],
            "temperature": call_params.get("temperature", 0.4),
            "max_tokens": call_params.get("max_tokens", 4096),
        }
        # Groq supports tools too — pass them if present
        if "tools" in call_params:
            groq_params["tools"] = call_params["tools"]
            groq_params["tool_choice"] = call_params.get("tool_choice", "auto")

        primary_done = asyncio.Event()
        winner_result = None
        winner_source = None

        async def call_openrouter():
            nonlocal winner_result, winner_source
            try:
                resp = await self.client.chat.completions.create(**call_params)
                if winner_result is None:
                    winner_result = resp
                    winner_source = "openrouter"
                primary_done.set()
                return resp
            except Exception as e:
                primary_done.set()
                raise

        async def call_groq_after_delay():
            nonlocal winner_result, winner_source
            # Wait 4 seconds — if OpenRouter replied within 4s, we bail
            try:
                await asyncio.wait_for(primary_done.wait(), timeout=4.0)
                # OpenRouter already done, no need to call Groq
                return None
            except asyncio.TimeoutError:
                pass  # OpenRouter is slow → fire Groq now
            logger.info("⚡ OpenRouter >4s — firing Groq fallback in parallel")
            try:
                groq_client = _get_groq_client()
                resp = await groq_client.chat.completions.create(**groq_params)
                if winner_result is None:
                    winner_result = resp
                    winner_source = "groq_fallback"
                return resp
            except Exception as e:
                logger.warning(f"Groq fallback failed: {e}")
                return None

        # Run both coroutines concurrently
        or_task = asyncio.create_task(call_openrouter())
        groq_task = asyncio.create_task(call_groq_after_delay())

        # Wait until at least one succeeds
        done, pending = await asyncio.wait(
            [or_task, groq_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel the slower task
        for t in pending:
            t.cancel()

        if winner_result is not None:
            if winner_source == "groq_fallback":
                logger.info("✅ Groq fallback won the race (OpenRouter was too slow)")
            else:
                logger.info(f"✅ OpenRouter responded in time (winner: {winner_source})")
            return winner_result

        # Both failed — raise whatever the OpenRouter task raised
        for t in done:
            exc = t.exception()
            if exc:
                raise exc
        raise RuntimeError("Both OpenRouter and Groq LLM calls failed")

    async def _proactive_prefetch(self, query: str, entities: dict) -> str:
        """
        ALWAYS runs before the first LLM call.
        Fetches live web search results + NSE quotes so the LLM has
        real-time data even if it never calls a tool itself.
        Returns a formatted context string (empty string if all fail).
        """
        from ..tools.tool_executor import execute_tool
        import json as _json

        search_q = self._build_search_query(query, entities)
        symbols = (entities or {}).get("symbols", [])
        context_parts = []

        # Run web search + NSE quotes in parallel
        tasks = []
        task_labels = []

        if search_q:
            tasks.append(execute_tool("search_web", {"query": search_q}))
            task_labels.append("web_search")

        for sym in symbols[:3]:  # cap at 3 symbols
            tasks.append(execute_tool("fetch_nse_quote", {"symbol": sym}))
            task_labels.append(f"nse_quote_{sym}")

        if not tasks:
            return ""

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.warning(f"Proactive prefetch gather failed: {e}")
            return ""

        for label, res in zip(task_labels, results):
            if isinstance(res, Exception):
                logger.warning(f"Prefetch task {label} failed: {res}")
                continue
            if isinstance(res, dict) and "error" not in res:
                snippet = _json.dumps(res, default=str)
                if len(snippet) > 2000:
                    snippet = snippet[:2000] + "... [truncated]"
                context_parts.append(f"[{label.upper()}]\n{snippet}")
                logger.info(f"✅ Prefetch {label} succeeded ({len(snippet)} chars)")

        if context_parts:
            combined = "\n\n".join(context_parts)
            logger.info(f"💉 Proactive context injected — {len(combined)} chars of live data")
            return combined

        return ""

    async def analyze(self, state: AgentState) -> AgentState:
        """
        Deep reasoning with multi-step tool calling support.
        GPT-OSS-120B with high effort for maximum analytical depth.
        """
        query = state["query"]
        entities = state.get("extracted_entities") or {}

        # ── EARLY EXIT: skip ALL pre-fetch and tool calling for no-search queries ──
        # The router (LLM) already decided needs_search=False for this query.
        # This covers greetings, casual talk, concept explanations — anything the
        # LLM can answer directly from training knowledge.
        enable_search = state.get("enable_research_loop", True)
        if not enable_search:
            try:
                quick_system = """You are Daddy's AI — a warm, knowledgeable financial assistant built for InvestingDaddy (India).

Answer from your training knowledge. No web search, no live data needed.

Rules:
- Greetings / small talk in ANY language → reply warmly in English (2-3 sentences max)
- Concept questions → explain clearly with Indian market examples, structured paragraphs
- Self-intro questions → brief, friendly, mention your capabilities
- Always English, always helpful, never robotic
- Do NOT mention "I don't have real-time data"
- Do NOT start with "I" — open with the answer directly"""

                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": quick_system},
                        {"role": "user", "content": query},
                    ],
                    temperature=0.7,
                    max_tokens=600,
                )
                state["final_response"] = (resp.choices[0].message.content or "").strip()
                state["execution_metadata"] = {
                    "agent": "reasoning_deep",
                    "mode": "direct_no_search",
                    "skip_verifier": True,
                }
                return state
            except Exception as e:
                logger.warning(f"direct no-search reply failed ({e}), falling through to full pipeline")
        # ── END EARLY EXIT ──────────────────────────────────────────────────────────

        system_prompt = """🌐 LANGUAGE RULE: ALWAYS respond in English only. Never use Hindi, Chinese, or any other language regardless of the query language.

You are Daddy's AI — the most capable financial intelligence assistant for Indian markets and global topics.

=== CORE IDENTITY ===
Creator: Adarsh, a Class 8 student at Daddy's International School, Chandauli
Founder: Dr. Vinay Prakash Tiwari (Vinay Sir) — Founder of InvestingDaddy
Platform: InvestingDaddy — India's premier stock market education platform
Powered by: GPT-OSS-120B deep reasoning engine, multi-agent architecture

=== YOUR INTELLIGENCE ===
You think like a senior analyst at Goldman Sachs who is also an amazing journalist and teacher.
You combine deep analytical reasoning, practical market wisdom, teaching ability, and data discipline.
You NEVER guess. You ALWAYS verify with tools before making any factual claim.

=== PERSONALITY ===
Confident and opinionated with data backing. Adapt to the user: quick question = crisp answer.
Professional yet warm — like chatting with a brilliant friend over coffee.
Use rupee symbol for Indian context. Mirror user's language — Hinglish if they use it.
NEVER be robotic or formulaic. Every response should feel human and thoughtful.

╔══════════════════════════════════════════════════════════════╗
║            🔍 SEARCH DECISION RULEBOOK — READ FIRST          ║
╚══════════════════════════════════════════════════════════════╝

You will receive a block of LIVE DATA already pre-fetched for you at the top of the conversation.
That data is real-time. USE IT. But you may ALSO call additional tools to go deeper.

──────────────────────────────────────────────────────────────
✅ ALWAYS CALL search_web FOR:
──────────────────────────────────────────────────────────────
1. Any stock/company analysis           → search_web("[Company] latest news India 2026")
2. Market events (IPO, results, merger) → search_web("[event] India latest")
3. Geopolitics / war / economy          → search_web("[topic] latest news 2026")
4. Sector performance / outlook         → search_web("[sector] sector India outlook 2026")
5. Any question with words: today, now, current, latest, recent, this week, this month
6. Any question about a PRICE or VALUE  → search_web("[stock/commodity] price today India")
7. If the pre-fetched data is incomplete, outdated, or says "error" → call search_web AGAIN

──────────────────────────────────────────────────────────────
❌ DO NOT call search_web FOR:
──────────────────────────────────────────────────────────────
1. Pure concept/definition questions with NO factual claim  (e.g., "what is PE ratio", "explain SIP")
2. Mathematical or hypothetical calculations               (e.g., "if I invest ₹5000 monthly for 10 years")
3. Questions you can answer FULLY from the pre-fetched LIVE DATA already given to you
   (do NOT call search_web again if the data block already has a thorough answer)

──────────────────────────────────────────────────────────────
🛠️ TOOL CALLING SEQUENCE (follow in ORDER):
──────────────────────────────────────────────────────────────
For STOCK queries:
  1. fetch_nse_quote(symbol)                          ← Get live price/OHLC
  2. search_web("[Company] latest news India 2026")   ← Get recent news
  3. get_stock_fundamentals(symbol)                   ← Get PE, ROE, financials

For MARKET OVERVIEW:
  1. search_web("Nifty 50 Sensex India market today 2026")
  2. get_market_sentiment()
  3. fetch_fii_dii()

For SECTOR queries:
  1. search_web("[sector] sector India 2026 performance")
  2. compare_stocks([top stocks in sector])

For CURRENT EVENTS (geopolitics, economy, policy):
  1. search_web("[topic] 2026 latest")  ← ALWAYS first, no exceptions

For CONCEPTS / DEFINITIONS:
  1. search_knowledge_base("[concept full name]")
  2. search_web("[concept] India stock market explained") if KB empty

FALLBACK: If ANY tool fails → immediately try search_web. NEVER give up without at least one search attempt.

──────────────────────────────────────────────────────────────
⚠️ GOLDEN RULES FOR TOOL USE:
──────────────────────────────────────────────────────────────
- NEVER say "I don't have real-time data" — you DO, and live data is also pre-fetched for you
- NEVER fabricate prices, numbers, or events — use ONLY tool data or the pre-fetched block
- If the pre-fetched data already fully answers the question → synthesize it without re-calling tools
- If the pre-fetched data is partial or missing → call the appropriate tool(s) above
- NEVER call a tool twice for the same thing in one session
- Maximum tool call rounds: 7. Use them wisely.

=== RESPONSE FORMAT — WIKIPEDIA / MAGAZINE STYLE ===

HARD BAN: NO BULLETS WHATSOEVER. ZERO. NOT A SINGLE ONE.
Banned as list items: - (hyphen)  * (asterisk)  • (bullet)  ● (circle bullet)  · (dot)
If you write even one bullet point, you have completely failed.

ONLY FORMAT ALLOWED: flowing prose paragraphs under clear section headings.

WRONG — NEVER DO THIS:
## What Happened
● US struck Iran on Feb 28
● Khamenei was killed
● Oil hit $100/barrel

RIGHT — ALWAYS DO THIS:
## What Happened
US and Israeli forces launched Operation Epic Fury on February 28, 2026, striking over 500
Iranian military targets in one of the most consequential military actions in decades. Supreme
Leader Ali Khamenei was killed within the first hours, the first time a sitting Iranian head of
state had been eliminated since the revolution. Oil prices surged above $100 per barrel as
tanker traffic through the Strait of Hormuz came to a standstill.

MANDATORY STRUCTURE FOR ANALYSIS:

## [Punchy title with key number, date, or insight]
[2-4 sentence opening paragraph. The single most important fact, told as a story.]

## [Background / Context]
[Dense narrative paragraph. Historical cause-and-effect in flowing sentences. No lists.]

## [Current Situation / Key Developments]
[What is happening NOW. Specific dates, numbers, names from your search — all in prose.]

## 📊 Key Data
[ONE table max — only when genuinely comparing multiple values side by side]
| Metric | Value | Significance |
|--------|-------|--------------|
| ...    | ...   | ...          |

## [Outlook / What to Watch]
[Direct 2-3 sentence view. Bull and bear case in prose sentences, not bullets.]

End with: "⚠️ *Not financial advice. Consult a SEBI-registered advisor.*"

FOR QUICK QUERIES: 2-3 flowing paragraphs, no headings needed.
EMOJIS: 1-2 max on section headings only.

=== INTERACTIVE CHARTS ===

When you have PRICE DATA from tools, include a chart block for the frontend to render:

SINGLE STOCK — area chart with support/resistance:
:::chart
{"type": "area", "title": "NIFTY 50 — 30 Day Trend", "subtitle": "NSE", "data": [{"time": "2026-02-15", "value": 22340}, {"time": "2026-02-16", "value": 22450}], "annotations": [{"price": 22500, "label": "Resistance", "color": "#ef4444", "lineStyle": "dashed"}, {"price": 22200, "label": "Support", "color": "#22c55e", "lineStyle": "dashed"}]}
:::

COMPARING STOCKS — use histogram for % changes side by side:
:::chart
{"type": "histogram", "title": "RELIANCE vs TCS — Monthly Return %", "data": [{"time": "2025-10-01", "value": 2.5, "color": "#f97316"}, {"time": "2025-11-01", "value": -1.2, "color": "#f97316"}, {"time": "2025-12-01", "value": 3.8, "color": "#22c55e"}]}
:::

For comparisons, include ONE chart per stock so the user can see both price trends.

RULES: type = "area" | "candlestick" | "histogram". data = [{time: "YYYY-MM-DD", value: NUM}]. annotations = optional.
ONLY include chart when you have REAL numeric data from tools. NEVER fabricate chart data.

=== GOLDEN RULES ===
1. NEVER say "I don't have real-time data" — you DO, use search_web
2. NEVER fabricate numbers or events — tool data only
3. If tools fail → search_web. ALWAYS try before giving up.
4. Paragraphs always. Tables only for true comparisons. ZERO bullets anywhere.
5. Lead with the most important insight in the very first sentence
6. Specific: exact numbers, exact dates, real names from search results
7. Naturally integrate data: "The conflict began on February 28..." not "Tool says..."
8. For financial opinions: bull case AND bear case, in prose"""

        try:
            # ── STEP 0: Proactive pre-fetch (runs BEFORE any LLM call) ──────────
            # This guarantees live data reaches the LLM even if the model
            # doesn't support tool calling (e.g. free OpenRouter models).
            live_context = await self._proactive_prefetch(query, entities)

            # Build conversation with history for context
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history for context retention
            conversation_history = state.get("conversation_history", [])
            if conversation_history:
                logger.info(f"Including {len(conversation_history)} messages from conversation history")
                filtered_history = []
                for msg in conversation_history[-10:]:
                    filtered_msg = {"role": msg["role"], "content": msg["content"]}
                    filtered_history.append(filtered_msg)
                messages.extend(filtered_history)

            # Inject pre-fetched live data as a system-level context block
            if live_context:
                messages.append({
                    "role": "user",
                    "content": (
                        f"📡 LIVE DATA PRE-FETCHED FOR YOU (use this as your primary source):\n\n"
                        f"{live_context}\n\n"
                        "---\n"
                        "The above data is real-time. Base your answer on it. "
                        "If the data is incomplete for this query, call additional tools to fill gaps."
                    )
                })
                messages.append({"role": "assistant", "content": "Understood. I have the live data. I will now answer the user's question using this data, and call additional tools if needed."})

            # Add current query — reinforce no-bullet rule in user message too
            messages.append({
                "role": "user",
                "content": f"{query}\n\nWrite your answer in flowing paragraphs with headings — NO bullet points of any kind. Use the pre-fetched live data above. Call additional tools if needed for more depth."
            })

            # Multi-step tool calling loop for deep research
            tool_results = {}
            max_iterations = 7
            iteration = 0

            logger.info(f"Starting deep research with up to {max_iterations} tool-calling rounds")

            while iteration < max_iterations:
                iteration += 1

                call_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }

                if self.tools and settings.enable_tool_calling:
                    call_params["tools"] = self.tools
                    call_params["tool_choice"] = "auto"

                if iteration == 1:
                    if self.reasoning_effort is not None:
                        call_params["reasoning_effort"] = self.reasoning_effort

                try:
                    response = await self._race_llm_call(call_params, iteration)
                except Exception as tool_err:
                    err_str = str(tool_err)
                    if "tool_use_failed" in err_str or "tool call validation failed" in err_str:
                        logger.warning(f"Tool call validation failed — manually fetching data: {err_str[:200]}")
                        manual_data = {}
                        symbols = (entities or {}).get("symbols", [])

                        for sym in symbols:
                            try:
                                quote = await execute_tool("fetch_nse_quote", {"symbol": sym})
                                manual_data[f"fetch_nse_quote({sym})"] = quote
                                logger.info(f"✅ Manually fetched quote for {sym}")
                            except Exception as qe:
                                logger.warning(f"Manual quote fetch failed for {sym}: {qe}")

                        try:
                            search_q = self._build_search_query(query, entities)
                            web = await execute_tool("search_web", {"query": search_q})
                            manual_data["search_web"] = web
                            logger.info("✅ Manually fetched web search results")
                        except Exception as we:
                            logger.warning(f"Manual web search failed: {we}")

                        if manual_data:
                            data_str = json.dumps(manual_data, default=str)[:4000]
                            messages.append({
                                "role": "user",
                                "content": f"Here is the REAL-TIME data I fetched for you:\n\n{data_str}\n\nNow synthesize this into a comprehensive answer with headings and paragraphs. NO bullet points. Use ONLY this data — do NOT fabricate numbers."
                            })

                        fallback_params = {
                            "model": self.model,
                            "messages": messages,
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                        }
                        if iteration == 1 and self.reasoning_effort is not None:
                            fallback_params["reasoning_effort"] = self.reasoning_effort
                        response = await self.client.chat.completions.create(**fallback_params)
                        tool_results.update(manual_data)
                    else:
                        raise

                message = response.choices[0].message

                if hasattr(message, 'tool_calls') and message.tool_calls:
                    messages.append(message)

                    async def run_and_format_tool(tc):
                        t_name = tc.function.name
                        t_args = json.loads(tc.function.arguments)
                        logger.info(f"[Round {iteration}] Executing tool: {t_name} with args: {t_args}")
                        try:
                            res = await execute_tool(t_name, t_args)
                        except Exception as tool_exec_err:
                            logger.warning(f"Tool {t_name} failed: {tool_exec_err}, falling back to search_web")
                            search_q = t_args.get("symbol", t_args.get("query", query))
                            try:
                                res = await execute_tool("search_web", {"query": f"{search_q} India stock market"})
                            except Exception:
                                res = {"error": f"Tool {t_name} and search_web both failed"}
                        return t_name, res, {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": t_name,
                            "content": json.dumps(res, default=str)
                        }

                    parallel_results = await asyncio.gather(*[run_and_format_tool(tc) for tc in message.tool_calls])

                    for t_name, res, tool_msg in parallel_results:
                        tool_results[t_name] = res
                        messages.append(tool_msg)

                    logger.info(f"Tool execution complete. Agent may request more tools in next round.")
                    continue
                else:
                    logger.info(f"Research complete after {iteration} rounds. Tools used: {list(tool_results.keys())}")
                    state["internal_reasoning"] = getattr(message, "reasoning", None)
                    state["final_response"] = message.content
                    break

            # If we hit max iterations without a final response, force synthesis
            if iteration >= max_iterations and not state.get("final_response"):
                logger.warning(f"Reached max tool iterations ({max_iterations}). Forcing final synthesis.")
                messages.append({
                    "role": "user",
                    "content": "We've reached the tool limit. Synthesize everything gathered into a clear narrative answer with headings and paragraphs. NO bullet points. Do NOT call any more tools."
                })
                final_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                state["final_response"] = final_response.choices[0].message.content

            state["tool_results"] = tool_results
            state["execution_metadata"] = {
                "agent": "reasoning_deep",
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "tools_called": list(tool_results.keys()),
                "tool_call_rounds": iteration
            }

            return state

        except Exception as e:
            logger.error(f"Deep reasoning agent error: {e}")
            try:
                search_q = self._build_search_query(query, entities)
                web_result = await execute_tool("search_web", {"query": search_q})
                fallback_msgs = [
                    {"role": "system", "content": "You are Daddy's AI — a brilliant analyst. Write your answer in flowing paragraphs with headings. NO bullet points. Use ONLY the data provided."},
                    {"role": "user", "content": f"Query: {query}\n\nSearch results:\n{json.dumps(web_result, default=str)[:3000]}"}
                ]
                resp = await self.client.chat.completions.create(
                    model=self.model, messages=fallback_msgs,
                    temperature=0.4, max_tokens=self.max_tokens
                )
                state["final_response"] = resp.choices[0].message.content
                state["execution_metadata"] = {"agent": "reasoning_deep", "fallback": "search_web"}
            except Exception as e2:
                logger.error(f"Search_web fallback also failed: {e2}")
                state["error"] = str(e)
                state["final_response"] = "I encountered an error during analysis. Please try again."
            return state
