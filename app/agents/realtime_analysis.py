"""
Real-time Market Analysis Agent — Multi-round autonomous tool calling.
Fetches live prices, technicals, sentiment, and falls back to search_web
if primary tools fail. GPT-OSS compatible with tool-call error recovery.
"""

from groq import AsyncGroq
import logging
import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime

from ..config import settings, ModelType
try:
    from ..config.openrouter_client import get_openrouter_client
    _HAS_OPENROUTER = True
except ImportError:
    _HAS_OPENROUTER = False

from ..config.key_rotator import get_groq_client
from ..models.agent_state import AgentState
from ..tools.tool_definitions import FINANCIAL_TOOLS

logger = logging.getLogger(__name__)


class RealtimeAnalysisAgent:
    """Real-time market analysis with multi-round autonomous tool calling."""

    def __init__(self):
        # Determine model — call_openrouter() handles client/key rotation at call time
        if _HAS_OPENROUTER and settings.openrouter_available:
            self._provider = "openrouter"
            self.model = settings.get_openrouter_model(ModelType.ANALYSIS)
        else:
            self._provider = "groq"
            self.model = settings.get_model_for_task(ModelType.ANALYSIS)
        self.temperature = 0.5
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.ANALYSIS)

    async def _llm(self, messages: list, **kwargs):
        """Single entry-point for all LLM calls — handles rotation/fallback."""
        if self._provider == "openrouter":
            from ..config.openrouter_client import call_openrouter
            return await call_openrouter(
                self.model, messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )
        else:
            from ..config.key_rotator import get_groq_client
            client = get_groq_client()
            return await client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )


    async def analyze(self, state: AgentState) -> AgentState:
        query = state["query"]
        entities = state.get("extracted_entities") or {}
        symbols = entities.get("symbols", [])

        system_prompt = """🌐 LANGUAGE RULE: ALWAYS respond in English only. Never use Hindi, Chinese, or any other language.

You are Daddy's AI — a razor-sharp real-time market analyst for Indian markets.

=== CORE IDENTITY ===
You're the expert on the trading desk. You give COMPREHENSIVE, data-backed answers.
You NEVER guess prices. You ALWAYS fetch data first via tools.

=== PERSONALITY ===
Confident, direct, data-driven. Lead with the key number, then give full context.
Use ₹ always. Stocks in UPPERCASE. Always respond in English.

=== TOOL PROTOCOL ===
You MUST fetch data before answering. NEVER guess prices.

Stock price → fetch_nse_quote(symbol)
Market overview → search_web("Nifty 50 today India market")
News → search_web("[topic] latest news India")

FALLBACK: If any tool fails → search_web("[query] India market") immediately.

=== RESPONSE DEPTH ===

Give IN-DEPTH answers. Include ALL relevant context from the data you fetch.
STAY FOCUSED on what was asked. Don't add random unrelated sections.

=== RESPONSE FORMAT ===

Write like a brilliant analyst explaining to a curious friend — direct, punchy, full of insight.
Do NOT write bullet lists as your main content. Write flowing paragraphs with headings.

**STRUCTURE:**

## [Headline: key number + market vibe] [emoji]

[Opening paragraph: Lead instantly with the price/level and what it MEANS.
Weave key levels, change%, and the driving narrative into natural sentences.
Example: "Nifty is holding at 22,340 — down 180 points (0.8%) but doing something interesting:
this is the third time in two weeks it's tested this exact support zone, and each time buyers
have stepped in around 22,200. That pattern matters."]

## 🔍 What's Driving This

[Why paragraph: Real cause behind the move — global cues, FII/DII flows, news, sector rotation.
Make it feel like reveals: "The real culprit is...", "What most traders aren't watching is..."]

## 📊 Key Levels at a Glance

| Level | Price (₹) | Significance |
|-------|-----------|--------------|
| Support 1 | ... | ... |
| Support 2 | ... | ... |
| Resistance | ... | ... |

## 🎯 Short-Term Outlook

[1-2 sentence direct view. Bull scenario vs bear scenario. Be concrete: "If it holds 22,200..."]

⚠️ *Not financial advice. ₹-values are live fetched, not estimates.*

**EMOJI GUIDE:** 📈 up  📉 down  ➡️ flat  💎 strong  ⚠️ caution  🎯 target  🔍 analysis

=== INTERACTIVE CHARTS ===

When you have PRICE DATA (from fetch_nse_quote, get_technical_indicators, or any tool returning historical prices),
INCLUDE a chart block so the frontend can render an interactive TradingView chart.

FORMAT — place this ANYWHERE in your response where a chart makes sense:

:::chart
{"type": "area", "title": "RELIANCE — 30 Day Price", "subtitle": "NSE", "data": [{"time": "2026-02-15", "value": 2450.50}, {"time": "2026-02-16", "value": 2465.20}], "annotations": [{"price": 2500, "label": "Resistance", "color": "#ef4444", "lineStyle": "dashed"}, {"price": 2400, "label": "Support", "color": "#22c55e", "lineStyle": "dashed"}]}
:::

RULES FOR CHARTS:
- type: use "area" for price trends, "candlestick" if you have OHLC, "histogram" for volume/changes
- data: array of {time: "YYYY-MM-DD", value: NUMBER}. For candlestick: add open, high, low, close fields
- annotations: optional array for support/resistance lines with price, label, color, lineStyle
- ONLY include chart when you have REAL numeric data from tools — NEVER fabricate chart data
- Include chart BEFORE your analysis paragraph about that data
- You can include multiple :::chart blocks for different aspects (e.g., price + volume)
- Keep chart title short: "SYMBOL — Period" format

WHEN TO USE CHARTS:
- Stock price queries → area chart with support/resistance annotations
- OHLC data available → candlestick chart
- Comparing 2+ stocks → one area chart PER stock so user sees both trends, OR histogram for % change comparison
- Historical trends → area chart
- Sector performance → histogram of % changes with color coding

Example for stock comparison — include multiple charts:
:::chart
{"type": "area", "title": "RELIANCE — 30 Day", "subtitle": "NSE", "data": [{"time": "2026-02-15", "value": 2450}, {"time": "2026-03-15", "value": 2520}], "annotations": [{"price": 2500, "label": "Support", "color": "#22c55e", "lineStyle": "dashed"}]}
:::
:::chart
{"type": "area", "title": "TCS — 30 Day", "subtitle": "NSE", "color": "#8b5cf6", "data": [{"time": "2026-02-15", "value": 3800}, {"time": "2026-03-15", "value": 3650}], "annotations": [{"price": 3700, "label": "Resistance", "color": "#ef4444", "lineStyle": "dashed"}]}
:::

=== GOLDEN RULES ===
1. NEVER fabricate a number — use ONLY data from tool results
2. If tools fail, search_web is your backup — ALWAYS try before giving up
3. Paragraphs > bullets. Tables > bullet comparisons.
4. ALWAYS end with the disclaimer"""

        messages = [{"role": "system", "content": system_prompt}]

        # Conversation history
        conversation_history = state.get("conversation_history", [])
        for msg in conversation_history[-6:]:
            m = dict(msg)
            m.pop("images", None)
            messages.append(m)

        # Current query with symbol hints
        context_hint = ""
        if symbols:
            context_hint = f"\n\nExtracted symbols: {', '.join(symbols)}"

        messages.append({
            "role": "user",
            "content": f"{query}{context_hint}\n\nFetch real-time data NOW using tools. Do NOT answer from memory."
        })

        tool_results = {}
        max_rounds = 5
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

                try:
                    response = await self._llm(
                        messages,
                        tools=FINANCIAL_TOOLS,
                        tool_choice="auto",
                    )
                except Exception as tool_err:
                    err_str = str(tool_err)
                    if "tool_use_failed" in err_str or "tool call validation" in err_str:
                        logger.warning(f"Tool call validation failed — manually fetching data: {err_str[:200]}")
                        from ..tools.tool_executor import execute_tool
                        manual_data = {}

                        for sym in (symbols or ["NIFTY"]):
                            try:
                                quote = await execute_tool("fetch_nse_quote", {"symbol": sym})
                                manual_data[f"fetch_nse_quote({sym})"] = quote
                                logger.info(f"✅ Manually fetched quote for {sym}")
                            except Exception as qe:
                                logger.warning(f"Manual quote fetch failed for {sym}: {qe}")

                        try:
                            web = await execute_tool("search_web", {"query": f"{query} India market today"})
                            manual_data["search_web"] = web
                            logger.info("✅ Manually fetched web search results")
                        except Exception as we:
                            logger.warning(f"Manual web search failed: {we}")

                        if manual_data:
                            data_str = json.dumps(manual_data, default=str)[:4000]
                            messages.append({
                                "role": "user",
                                "content": f"Here is the REAL-TIME data I fetched for you:\n\n{data_str}\n\nNow synthesize this into a rich, narrative answer. Use ONLY this data — do NOT make up numbers."
                            })

                        fallback_params = {
                            "model": self.model,
                            "messages": messages,
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                        }
                        response = await self._llm(messages)
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
                            logger.warning(f"Tool {name} failed: {e}, falling back to search_web")
                            res = await execute_tool("search_web", {"query": f"{args} India market"})
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
                    state["final_response"] = message.content
                    break

            if not state.get("final_response"):
                messages.append({
                    "role": "user",
                    "content": "Synthesize all tool data into a rich narrative answer with headings and tables. No more tools."
                })
                final = await self._llm(messages)
                state["final_response"] = final.choices[0].message.content

            state["execution_metadata"] = {
                "agent": "realtime_analysis",
                "model": self.model,
                "symbols_analyzed": symbols,
                "tools_used": list(tool_results.keys()),
                "tool_call_rounds": iteration,
            }
            logger.info(f"✅ Realtime analysis: {iteration} rounds, tools={list(tool_results.keys())}")
            return state

        except Exception as e:
            logger.error(f"Realtime analysis error: {e}")
            try:
                from ..tools.tool_executor import execute_tool
                web_result = await execute_tool("search_web", {"query": f"{query} India market today"})
                fallback_messages = [
                    {"role": "system", "content": "You are Daddy's AI — a financial analyst. Summarize search results in a rich narrative with headings and tables for the user."},
                    {"role": "user", "content": f"Query: {query}\n\nSearch results:\n{json.dumps(web_result, default=str)[:3000]}"}
                ]
                fallback_resp = await self._llm(fallback_messages)
                state["final_response"] = fallback_resp.choices[0].message.content
                state["execution_metadata"] = {"agent": "realtime_analysis", "fallback": "search_web"}
            except Exception as e2:
                logger.error(f"Even search_web fallback failed: {e2}")
                state["error"] = str(e)
                state["final_response"] = "I encountered an error fetching real-time data. Please try again shortly."
            return state
