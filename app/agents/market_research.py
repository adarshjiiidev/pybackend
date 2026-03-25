"""
Market Research Agent — Deep fundamental analysis with multi-round autonomous tool calling.
Uses GPT-OSS-120B for deep thinking. Falls back to search_web on tool failure.
"""

import logging
import asyncio
import json
from typing import Dict, Any
from datetime import datetime

from groq import AsyncGroq
from ..config import settings, ModelType
try:
    from ..config.openrouter_client import get_openrouter_client
    _HAS_OPENROUTER = True
except ImportError:
    _HAS_OPENROUTER = False

from ..config.key_rotator import get_groq_client
from ..models.agent_state import AgentState, AgentMode
from ..tools.tool_definitions import FINANCIAL_TOOLS

logger = logging.getLogger(__name__)


class MarketResearchAgent:
    """Deep market research with multi-round autonomous tool calling."""

    def __init__(self):
        # OpenRouter first priority, Groq fallback
        if _HAS_OPENROUTER and settings.openrouter_available:
            from ..config.openrouter_client import get_openrouter_client as _get_or
            self.client = _get_or()
            self._provider = "openrouter"
        else:
            self.client = get_groq_client()
            self._provider = "groq"
        if hasattr(self, '_provider') and self._provider == "openrouter":
            self.model = settings.get_openrouter_model(ModelType.REASONING_DEEP)
        else:
            self.model = settings.get_model_for_task(ModelType.REASONING_DEEP)
        self.temperature = settings.get_temperature_for_task(ModelType.REASONING_DEEP)
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.REASONING_DEEP)

    async def analyze(self, state: AgentState) -> AgentState:
        query = state["query"]
        entities = state.get("extracted_entities") or {}
        symbols = entities.get("symbols", [])

        system_prompt = """You are Daddy's AI — a senior research analyst and world-class explainer.

=== CORE IDENTITY ===
You combine deep expertise with the ability to explain complex ideas simply and compellingly.
You write like a brilliant journalist meets a sharp analyst: story-first, data-backed, opinionated.
Your training data is STALE — you MUST use tools for any factual claim about current events or prices.

=== PERSONALITY ===
Lead with INSIGHT, not data dumps. Tell the story behind the numbers.
Be opinionated with data backing — analysts have views, share yours.
Use ₹, lakhs, crores for Indian context. Use $ for global/geopolitical topics.
Match the user's language — Hinglish if they use it.
Professional yet warm — like chatting with a brilliant friend over coffee.

=== AUTONOMOUS TOOL PROTOCOL ===

You MUST use tools before making ANY factual claim. Your sequence:

For stock analysis:
1. fetch_nse_quote(symbol) — current price
2. get_stock_fundamentals(symbol) — PE, ROE, debt, financials
3. search_web("[company] latest news results India") — recent developments
4. search_financial_news("[company]") — earnings, announcements
5. get_technical_indicators(symbol) — RSI, MACD, support/resistance

For comparisons:
1. compare_stocks(symbols) — side-by-side metrics
2. fetch_nse_quote for each — live prices
3. search_web for each — recent news

For sectors or global topics (war, geopolitics, economy, current events):
1. search_web("[topic] latest news [year]") — always search first
2. search_web("[topic] impact India market") — if relevant

CRITICAL FALLBACK:
If ANY tool fails → use search_web("[query]") immediately.
search_web is your UNIVERSAL SAFETY NET. NEVER give up without trying it.

=== RESPONSE FORMAT === WIKIPEDIA STYLE ===

⛔ BULLETS ARE COMPLETELY BANNED. NOT A SINGLE HYPHEN (-), ASTERISK (*), OR BULLET (•) ANYWHERE IN YOUR RESPONSE.
⛔ If you write a bullet point list, you have failed. Everything must be prose.

✅ WRITE LIKE WIKIPEDIA — dense, informative paragraphs under clear section headings.

WRONG (banned):
## Key Developments
- Iran struck oil facilities
- US deployed carrier group
- Oil hit $100/barrel

RIGHT (required):
## Key Developments
US forces struck over 500 Iranian military targets in the opening salvo, with Iran retaliating
within hours by launching over 300 drones and ballistic missiles toward US bases across the Gulf.
Brent crude surpassed $100 per barrel for the first time since 2022 as tanker operations in the
Strait of Hormuz were disrupted.

STRUCTURE:
## [Title with key number or date] [optional emoji]
[2-4 sentence opening paragraph — the #1 most important insight, told as a story]

## [Background / Context section]
[Dense narrative paragraph. Connect events chronologically in prose. No lists.]

## [Current Situation / What's Happening Now]
[Paragraph on current developments. Real dates, specific numbers, real names from your search.]

## 📊 Key Data
[ONE table max — side-by-side comparisons only, when it genuinely helps]
| Column A | Column B | Column C |
|----------|----------|----------|
| ...      | ...      | ...      |

## [Verdict / Outlook / What to Watch]
[Direct 2-3 sentence conclusion. Your honest forward-looking view in prose.]

⚠️ *Not financial advice. Consult a SEBI-registered advisor.*

FOR QUICK SIMPLE QUERIES: 2-3 flowing paragraphs, no headings needed.
EMOJIS: 1-2 max, only on headings when they truly add clarity.

=== INTERACTIVE CHARTS ===

When you have PRICE DATA from tools, include a chart block for the frontend to render:

:::chart
{"type": "area", "title": "RELIANCE — 30 Day Price", "subtitle": "NSE", "data": [{"time": "2026-02-15", "value": 2450.50}], "annotations": [{"price": 2500, "label": "Resistance", "color": "#ef4444", "lineStyle": "dashed"}]}
:::

RULES: type = "area" | "candlestick" | "histogram". data = [{time: "YYYY-MM-DD", value: NUM}]. annotations = optional support/resistance lines.
ONLY include chart when you have REAL numeric data from tools. NEVER fabricate chart data.

=== GOLDEN RULES ===
1. NEVER answer without calling at least ONE tool — your training data is stale
2. NEVER fabricate numbers or events — use tool data only
3. If primary tools fail, search_web ALWAYS works — use it
4. Be specific: use exact numbers, dates, names from your search results
5. For stock opinions: include bull AND bear case — in prose, not bullets
6. Cite data naturally: "The US launched strikes on February 28..." not "According to tool output..." """

        messages = [{"role": "system", "content": system_prompt}]

        # Conversation history
        conversation_history = state.get("conversation_history", [])
        for msg in conversation_history[-8:]:
            m = dict(msg)
            m.pop("images", None)
            messages.append(m)

        # Current query
        context_hint = ""
        if symbols:
            context_hint = f"\n\nExtracted symbols: {', '.join(symbols)}"

        messages.append({
            "role": "user",
            "content": f"{query}{context_hint}\n\nResearch this thoroughly using tools. Write your answer in flowing paragraphs, NO bullet points. Do NOT answer from memory."
        })

        tool_results = {}
        max_rounds = 6
        iteration = 0

        try:
            while iteration < max_rounds:
                iteration += 1

                call_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "tools": FINANCIAL_TOOLS,
                    "tool_choice": "auto",
                }

                # Reasoning effort for first round (Groq GPT-OSS only)
                if iteration == 1 and self._provider == "groq":
                    reasoning_effort = settings.get_reasoning_effort(ModelType.REASONING_DEEP)
                    if reasoning_effort:
                        call_params["reasoning_effort"] = reasoning_effort

                try:
                    response = await self.client.chat.completions.create(**call_params)
                except Exception as tool_err:
                    err_str = str(tool_err)
                    if "tool_use_failed" in err_str or "tool call validation" in err_str:
                        logger.warning(f"Tool call validation failed — manually fetching data")
                        from ..tools.tool_executor import execute_tool
                        manual_data = {}

                        for sym in symbols:
                            try:
                                quote = await execute_tool("fetch_nse_quote", {"symbol": sym})
                                manual_data[f"quote_{sym}"] = quote
                            except Exception:
                                pass

                        try:
                            web = await execute_tool("search_web", {"query": f"{query} India market"})
                            manual_data["search_web"] = web
                        except Exception:
                            pass

                        if manual_data:
                            data_str = json.dumps(manual_data, default=str)[:4000]
                            messages.append({
                                "role": "user",
                                "content": f"Here is REAL-TIME data I fetched:\n\n{data_str}\n\nSynthesize into a comprehensive narrative answer with headings and paragraphs. NO bullet points. Use ONLY this data — do NOT fabricate."
                            })

                        fallback_params = {
                            "model": self.model,
                            "messages": messages,
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                        }
                        response = await self.client.chat.completions.create(**fallback_params)
                        tool_results.update(manual_data)
                    else:
                        raise

                message = response.choices[0].message

                if hasattr(message, "tool_calls") and message.tool_calls:
                    messages.append(message)
                    from ..tools.tool_executor import execute_tool

                    async def run_tool(tc):
                        name = tc.function.name
                        args = json.loads(tc.function.arguments)
                        logger.info(f"[Round {iteration}] {name}({args})")
                        try:
                            res = await execute_tool(name, args)
                        except Exception as e:
                            logger.warning(f"Tool {name} failed: {e}, searching web instead")
                            search_q = args.get("symbol", args.get("query", query))
                            res = await execute_tool("search_web", {"query": f"{search_q} India stock market"})
                        return name, res, {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": json.dumps(res, default=str)
                        }

                    results = await asyncio.gather(*[run_tool(tc) for tc in message.tool_calls])
                    for t_name, res, tool_msg in results:
                        tool_results[t_name] = res
                        messages.append(tool_msg)
                    continue
                else:
                    state["internal_reasoning"] = getattr(message, "reasoning", None)
                    state["final_response"] = message.content
                    break

            if not state.get("final_response"):
                messages.append({
                    "role": "user",
                    "content": "Synthesize all gathered data into a comprehensive analysis with headings and paragraphs. NO bullet points. No more tools."
                })
                final = await self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=self.temperature, max_tokens=self.max_tokens
                )
                state["final_response"] = final.choices[0].message.content

            state["tool_results"] = tool_results
            state["execution_metadata"] = {
                "agent": "market_research",
                "model": self.model,
                "symbols_analyzed": symbols,
                "tools_used": list(tool_results.keys()),
                "tool_call_rounds": iteration,
            }
            logger.info(f"✅ Market research: {iteration} rounds, tools={list(tool_results.keys())}")
            return state

        except Exception as e:
            logger.error(f"Market research error: {e}")
            try:
                from ..tools.tool_executor import execute_tool
                web_result = await execute_tool("search_web", {"query": f"{query} India stock market analysis"})
                fallback_msgs = [
                    {"role": "system", "content": "You are Daddy's AI — a financial analyst. Write your answer in flowing paragraphs with headings. No bullet points."},
                    {"role": "user", "content": f"Query: {query}\n\nSearch results:\n{json.dumps(web_result, default=str)[:3000]}"}
                ]
                resp = await self.client.chat.completions.create(
                    model=self.model, messages=fallback_msgs,
                    temperature=0.4, max_tokens=self.max_tokens
                )
                state["final_response"] = resp.choices[0].message.content
                state["execution_metadata"] = {"agent": "market_research", "fallback": "search_web"}
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                state["error"] = str(e)
                state["final_response"] = "I encountered an error during market research. Please try again."
            return state
