"""
ReportAgent — Final Report Generator (Asset Sub-Agent)
=======================================================
Phase 2 · Swarm Agent

The ReportAgent is always the LAST agent spawned in an orchestration run.
It acts as the "asset sub-agent" from the diagram — it reads all prior
agent findings and synthesises them into a beautifully formatted report.

Capabilities:
  • Executive Summary   — one-paragraph verdict with signal + confidence
  • Technical Analysis  — key indicators, patterns, S/R levels
  • OI / Derivatives    — PCR, Max Pain, key walls
  • Global Cues         — how world markets affect India today
  • Prediction          — ML model outlook with price range
  • Fundamentals        — valuation metrics if available
  • Sentiment           — news + social sentiment
  • Risk Warnings       — shock detection alerts, disclaimers
  • Action Plan         — concrete levels to watch (entry, stop, target)

Output formats:
  • report_md     — Full markdown (for chat display)
  • final_response — Conversational answer (for plain text)
  • sections      — Structured dict of each section

Input payload keys:
  query           (str)    Original user query
  intent          (str)    Detected intent from planner
  entities        (dict)   Extracted entities (symbols, sector, etc.)
  analysis        (dict)   Aggregated analysis from orchestrator
  agent_results   (list)   Results from all preceding agents
  key_findings    (list)   Top findings from analysis phase
  warnings        (list)   Risk warnings
  plan_reasoning  (str)    Why this plan was chosen
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..base_agent import AgentResult, AgentStatus, BaseSwarmAgent, SwarmMessage

logger = logging.getLogger(__name__)

AGENT_TYPE = "report"


# ---------------------------------------------------------------------------
# Section builders (rule-based, used as fallback if LLM fails)
# ---------------------------------------------------------------------------


def _signal_emoji(signal: str) -> str:
    return {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(signal.lower(), "⚪")


def _conf_bar(confidence: float) -> str:
    """Visual confidence bar: ████░░░░ 72%"""
    filled = int(confidence * 10)
    empty = 10 - filled
    return f"{'█' * filled}{'░' * empty} {confidence * 100:.0f}%"


def _build_exec_summary(
    query: str,
    signal: str,
    confidence: float,
    key_findings: List[str],
    warnings: List[str],
) -> str:
    emoji = _signal_emoji(signal)
    lines = [
        f"## {emoji} Executive Summary",
        "",
        f"**Overall Signal:** {signal.upper()} | **Confidence:** {_conf_bar(confidence)}",
        "",
    ]
    if key_findings:
        lines.append("**Key Findings:**")
        for f in key_findings[:5]:
            lines.append(f"- {f}")
        lines.append("")
    if warnings:
        lines.append("**⚠️ Warnings:**")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    return "\n".join(lines)


def _build_ta_section(ta_result: Optional[Dict[str, Any]]) -> str:
    if not ta_result or ta_result.get("status") != "done":
        return ""
    data = ta_result.get("data", {})
    symbol = data.get("symbol", "")
    price = data.get("current_price", 0)
    signal = ta_result.get("signal", "neutral")
    ts = data.get("trading_signal", "NEUTRAL")
    conf = ta_result.get("confidence", 0.5)
    emoji = _signal_emoji(signal)

    ind = data.get("indicators", {})
    sr = data.get("support_resistance", {})
    patterns = data.get("patterns", {})
    price_bars = data.get("price_bars", [])

    lines = [
        f"## 📊 Technical Analysis — {symbol}",
        "",
        f"**Signal:** {emoji} {ts} | **Confidence:** {_conf_bar(conf)}",
        f"**Current Price:** ₹{price:,.2f}",
        "",
    ]

    # Key indicators
    ind_lines = []
    if ind.get("rsi"):
        rsi_v = ind["rsi"].get("value_14", "—")
        rsi_s = ind["rsi"].get("interpretation", "")
        ind_lines.append(f"| RSI(14) | {rsi_v} | {rsi_s} |")
    if ind.get("macd"):
        macd_s = ind["macd"].get("interpretation", "—")
        ind_lines.append(f"| MACD | — | {macd_s} |")
    if ind.get("supertrend"):
        st_s = ind["supertrend"].get("interpretation", "—")
        ind_lines.append(
            f"| Supertrend | ₹{ind['supertrend'].get('value', '—')} | {st_s} |"
        )
    if ind.get("ema"):
        ema_s = ind["ema"].get("interpretation", "—")
        ind_lines.append(f"| EMA Cross | — | {ema_s} |")
    if ind.get("bollinger_bands"):
        bb_s = ind["bollinger_bands"].get("interpretation", "—")
        ind_lines.append(f"| Bollinger | — | {bb_s} |")
    if ind.get("adx"):
        adx_v = ind["adx"].get("adx", "—")
        adx_s = ind["adx"].get("trend_strength", "—")
        ind_lines.append(f"| ADX | {adx_v} | {adx_s} |")

    if ind_lines:
        lines += [
            "| Indicator | Value | Signal |",
            "|-----------|-------|--------|",
        ]
        lines += ind_lines
        lines.append("")

    # Support / Resistance
    if sr:
        imm_res = sr.get("immediate_resistance")
        imm_sup = sr.get("immediate_support")
        high52 = sr.get("52w_high")
        low52 = sr.get("52w_low")
        lines.append("**Key Levels:**")
        if imm_res:
            lines.append(f"- 🔴 Resistance: ₹{imm_res:,.2f}")
        if imm_sup:
            lines.append(f"- 🟢 Support: ₹{imm_sup:,.2f}")
        if high52:
            lines.append(f"- 📈 52-Week High: ₹{high52:,.2f}")
        if low52:
            lines.append(f"- 📉 52-Week Low: ₹{low52:,.2f}")
        lines.append("")

    # Top pattern
    if patterns:
        pat_summary = patterns.get("summary", {})
        top_pat = pat_summary.get("top_pattern")
        if top_pat:
            pat_sig = pat_summary.get("signal", "neutral")
            pat_conf = pat_summary.get("confidence", 0.5)
            lines.append(
                f"**Top Pattern:** {top_pat} — {pat_sig.upper()} "
                f"(confidence {pat_conf:.0%})"
            )
            lines.append("")

    # ── Embed interactive candlestick chart if we have OHLCV bars ──────────
    if price_bars and len(price_bars) >= 10:
        # Build annotations for key S/R levels
        annotations = []
        if sr:
            imm_res = sr.get("immediate_resistance")
            imm_sup = sr.get("immediate_support")
            if imm_res:
                annotations.append({
                    "price": imm_res,
                    "label": f"Resistance ₹{imm_res:,.0f}",
                    "color": "#ef4444",  # red
                    "lineStyle": "dashed",
                })
            if imm_sup:
                annotations.append({
                    "price": imm_sup,
                    "label": f"Support ₹{imm_sup:,.0f}",
                    "color": "#22c55e",  # green
                    "lineStyle": "dashed",
                })

        trend_badge = "▲ BULLISH" if signal == "bullish" else ("▼ BEARISH" if signal == "bearish" else "● NEUTRAL")
        chart_config = {
            "type": "candlestick",
            "title": f"{symbol} — Price Chart (90 Days)",
            "subtitle": f"Signal: {trend_badge} | Confidence: {conf:.0%}",
            "data": price_bars,
            "annotations": annotations,
            "showVolume": any("volume" in b for b in price_bars[:5]),
            "currency": "INR",
        }

        import json as _json
        chart_json = _json.dumps(chart_config, separators=(",", ":"))
        lines.append(":::chart")
        lines.append(chart_json)
        lines.append(":::")
        lines.append("")

    return "\n".join(lines)



def _build_oi_section(oi_result: Optional[Dict[str, Any]]) -> str:
    if not oi_result or oi_result.get("status") != "done":
        return ""
    data = oi_result.get("data", {})
    symbol = data.get("symbol", "")
    spot = data.get("spot_price", 0)
    ts = data.get("trading_signal", "NEUTRAL")
    signal = oi_result.get("signal", "neutral")
    emoji = _signal_emoji(signal)

    pcr = data.get("pcr", {})
    mp = data.get("max_pain", {})
    walls = data.get("oi_walls", {})
    iv_skew = data.get("iv_skew", {})
    oi_chg = data.get("oi_changes", {})

    lines = [
        f"## 🎯 Options & Open Interest — {symbol}",
        "",
        f"**OI Signal:** {emoji} {ts} | **Spot:** ₹{spot:,.0f}",
        f"**Expiry:** {data.get('expiry', '—')}",
        "",
    ]

    # OI table
    oi_rows = []
    if pcr.get("oi_pcr"):
        oi_rows.append(
            f"| PCR (OI) | {pcr['oi_pcr']:.3f} | {pcr.get('interpretation', '—')[:60]} |"
        )
    if pcr.get("volume_pcr"):
        oi_rows.append(
            f"| PCR (Vol) | {pcr['volume_pcr']:.3f} | Volume-based put/call ratio |"
        )
    if mp.get("max_pain"):
        oi_rows.append(
            f"| Max Pain | ₹{mp['max_pain']:,.0f} | {mp.get('interpretation', '—')[:60]} |"
        )
    if iv_skew.get("skew") is not None:
        oi_rows.append(
            f"| IV Skew | {iv_skew['skew']:+.1f}% | {iv_skew.get('interpretation', '—')[:60]} |"
        )

    if oi_rows:
        lines += [
            "| Metric | Value | Interpretation |",
            "|--------|-------|----------------|",
        ]
        lines += oi_rows
        lines.append("")

    # Walls
    key_res = walls.get("key_resistance")
    key_sup = walls.get("key_support")
    if key_res or key_sup:
        lines.append("**OI Walls (Key Levels):**")
        if key_res:
            lines.append(f"- 🔴 Call Wall (Resistance): ₹{key_res:,.0f}")
        if key_sup:
            lines.append(f"- 🟢 Put Wall (Support): ₹{key_sup:,.0f}")
        lines.append("")

    # OI change interpretation
    oi_chg_interp = oi_chg.get("interpretation", "")
    if oi_chg_interp:
        lines.append(f"**OI Change:** {oi_chg_interp}")
        lines.append("")

    return "\n".join(lines)


def _build_global_section(global_result: Optional[Dict[str, Any]]) -> str:
    if not global_result or global_result.get("status") != "done":
        return ""
    data = global_result.get("data", {})
    signal = global_result.get("signal", "neutral")
    emoji = _signal_emoji(signal)

    impact = data.get("india_impact", {})
    regime = data.get("risk_regime", {})
    fii_pred = data.get("fii_flow_prediction", {})
    alerts = data.get("key_alerts", [])
    snap = data.get("global_snapshot", {})

    lines = [
        "## 🌍 Global Market Cues",
        "",
        f"**Global Signal for India:** {emoji} {signal.upper()}",
        f"**India Impact Score:** {impact.get('score', 0):+.1f}/100",
        f"**Risk Regime:** {regime.get('regime', '—')}",
        "",
    ]

    # Key global prices
    indices = snap.get("indices", {})
    commodities = snap.get("commodities", {})
    forex = snap.get("forex", {})
    bonds = snap.get("bonds", {})

    global_rows = []
    for key in ["SP500", "NASDAQ", "NIKKEI", "HANGSENG"]:
        info = indices.get(key, {})
        if info.get("price"):
            global_rows.append(
                f"| {info.get('name', key)} | {info['price']:,.2f} | {info.get('region', '').upper()} |"
            )
    for key in ["CRUDEWTI", "GOLD", "BRENT"]:
        info = commodities.get(key, {})
        if info.get("price"):
            global_rows.append(
                f"| {info.get('name', key)} | ${info['price']:,.2f} | {info.get('unit', '')} |"
            )
    for key in ["DXY", "USDINR"]:
        info = forex.get(key, {})
        if info.get("price"):
            global_rows.append(
                f"| {info.get('name', key)} | {info['price']:,.4f} | {info.get('pair', '')} |"
            )
    us10y = bonds.get("US10Y", {})
    if us10y.get("yield_pct"):
        global_rows.append(f"| US 10Y Yield | {us10y['yield_pct']:.3f}% | Bond Yield |")

    if global_rows:
        lines += [
            "| Instrument | Price | Category |",
            "|------------|-------|----------|",
        ]
        lines += global_rows
        lines.append("")

    # FII prediction
    if fii_pred:
        fii_dir = fii_pred.get("direction", "neutral")
        fii_interp = fii_pred.get("interpretation", "")
        if fii_interp:
            lines.append(
                f"**FII Flow Prediction:** {fii_dir.upper()} — {fii_interp[:200]}"
            )
            lines.append("")

    # Key alerts
    if alerts:
        lines.append("**⚠️ Global Alerts:**")
        for alert in alerts[:3]:
            lines.append(f"- {alert}")
        lines.append("")

    return "\n".join(lines)


def _build_prediction_section(pred_result: Optional[Dict[str, Any]]) -> str:
    if not pred_result or pred_result.get("status") != "done":
        return ""
    data = pred_result.get("data", {})
    symbol = data.get("symbol", "")
    signal = pred_result.get("signal", "neutral")
    emoji = _signal_emoji(signal)
    label = data.get("signal_label", signal.upper())
    price = data.get("current_price", 0)

    predictions = data.get("predictions", {})
    price_range = data.get("price_range_5d", {})
    regime = data.get("market_regime", {})

    lines = [
        f"## 🤖 ML Prediction — {symbol}",
        "",
        f"**ML Signal:** {emoji} {label}",
        f"**Current Price:** ₹{price:,.2f}",
        f"**Market Regime:** {regime.get('regime', '—')} "
        f"({regime.get('ann_volatility_pct', 0):.1f}% ann. vol)",
        "",
    ]

    # Prediction table per horizon
    pred_rows = []
    for h_key in ["1d", "5d", "20d"]:
        pred = predictions.get(h_key)
        if pred and "error" not in pred:
            sig = pred.get("signal", "neutral")
            conf = pred.get("confidence", 0.5)
            proba = pred.get("probabilities", {})
            bull_p = proba.get("bullish", 0) * 100
            bear_p = proba.get("bearish", 0) * 100
            pred_rows.append(
                f"| {h_key} | {_signal_emoji(sig)} {sig.upper()} | {conf:.0%} | "
                f"Bull {bull_p:.0f}% / Bear {bear_p:.0f}% |"
            )

    if pred_rows:
        lines += [
            "| Horizon | Signal | Confidence | Probabilities |",
            "|---------|--------|------------|---------------|",
        ]
        lines += pred_rows
        lines.append("")

    # Price range
    if price_range.get("predicted_close"):
        lines += [
            "**5-Day Price Range Estimate:**",
            f"- 🎯 Target Close: ₹{price_range.get('predicted_close', 0):,.2f}",
            f"- 📈 Range High: ₹{price_range.get('predicted_high', 0):,.2f}",
            f"- 📉 Range Low: ₹{price_range.get('predicted_low', 0):,.2f}",
            f"- 95% CI: ₹{price_range.get('ci_95_lower', 0):,.2f} – ₹{price_range.get('ci_95_upper', 0):,.2f}",
            f"- Expected Move: ₹{price_range.get('expected_move_pts', 0):,.2f} "
            f"({price_range.get('expected_move_pct', 0):.2f}%)",
            "",
        ]

    lines.append("⚠️ *ML predictions are probabilistic estimates, not guarantees.*")
    lines.append("")
    return "\n".join(lines)


def _build_fundamentals_section(fund_result: Optional[Dict[str, Any]]) -> str:
    if not fund_result or fund_result.get("status") != "done":
        return ""
    data = fund_result.get("data", {})
    fund = data.get("fundamentals", {})
    symbol = data.get("symbol", "")
    if not fund:
        return ""

    lines = [
        f"## 💼 Fundamentals — {symbol}",
        "",
        f"**Company:** {fund.get('company_name', symbol)}",
        f"**Sector:** {fund.get('sector', '—')} | **Industry:** {fund.get('industry', '—')}",
        "",
    ]

    val_rows = []
    metric_map = [
        ("Market Cap", f"₹{fund.get('market_cap_cr', 0):,.0f} Cr"),
        ("PE Ratio", f"{fund.get('pe_ratio', '—')}x"),
        ("PB Ratio", f"{fund.get('pb_ratio', '—')}x"),
        ("EPS (TTM)", f"₹{fund.get('eps', '—')}"),
        ("ROE", f"{(fund.get('roe', 0) or 0) * 100:.1f}%" if fund.get("roe") else "—"),
        ("ROA", f"{(fund.get('roa', 0) or 0) * 100:.1f}%" if fund.get("roa") else "—"),
        ("Debt/Equity", str(fund.get("debt_to_equity", "—"))),
        ("Dividend Yield", f"{(fund.get('dividend_yield', 0) or 0) * 100:.2f}%"),
        ("Beta", str(fund.get("beta", "—"))),
        ("Profit Margin", f"{(fund.get('profit_margins', 0) or 0) * 100:.1f}%"),
    ]

    for name, val in metric_map:
        if val and val not in ("—", "None", "0%", "₹0 Cr", "0.0%"):
            val_rows.append(f"| {name} | {val} |")

    if val_rows:
        lines += ["| Metric | Value |", "|--------|-------|"]
        lines += val_rows
        lines.append("")

    return "\n".join(lines)


def _build_sentiment_section(sent_result: Optional[Dict[str, Any]]) -> str:
    if not sent_result or sent_result.get("status") != "done":
        return ""
    data = sent_result.get("data", {})
    signal = sent_result.get("signal", "neutral")
    emoji = _signal_emoji(signal)
    score = data.get("sentiment_score", 0.0)
    interp = data.get("interpretation", "")
    headlines = data.get("top_headlines", [])
    topic = data.get("query", "")

    lines = [
        f"## 📰 News & Sentiment — {topic}",
        "",
        f"**Sentiment Signal:** {emoji} {signal.upper()} | Score: {score:+.2f}",
    ]
    if interp:
        lines.append(f"*{interp}*")
    lines.append("")

    if headlines:
        lines.append("**Top Headlines:**")
        for h in headlines[:5]:
            sentiment_tag = h.get("sentiment", "neutral")
            headline_emoji = _signal_emoji(sentiment_tag)
            lines.append(f"- {headline_emoji} {h.get('headline', '')[:120]}")
        lines.append("")

    return "\n".join(lines)


def _build_web_research_section(web_result: Optional[Dict[str, Any]]) -> str:
    if not web_result or web_result.get("status") != "done":
        return ""
    data = web_result.get("data", {})
    report_md = data.get("report_md", "")
    key_facts = data.get("synthesis", [])
    sources = data.get("sources", [])
    total_findings = data.get("total_findings", 0)
    rounds = data.get("research_rounds", 0)

    if not report_md and not key_facts:
        return ""

    lines = [f"## 🔍 Research Findings ({total_findings} sources, {rounds} research rounds)", ""]

    if report_md and len(report_md.strip()) > 100:
        # Strip the outer ## header the WebResearchAgent adds to its own report_md
        # to avoid double-nesting like:  ## 🔍 Research Findings → ## 🔍 Research Report
        cleaned_lines = []
        for line in report_md.strip().splitlines():
            # Downgrade ## to ### so they become sub-sections, not duplicate top headers
            if line.startswith("## "):
                cleaned_lines.append("### " + line[3:])
            else:
                cleaned_lines.append(line)
        lines.append("\n".join(cleaned_lines))
        lines.append("")
    elif key_facts:
        for fact in key_facts[:8]:
            lines.append(f"- {fact}")
        lines.append("")

    if sources:
        lines.append("**Sources consulted:**")
        for src in sources[:5]:
            name = src.get("name", "")
            url = src.get("url", "")
            cred = src.get("credibility", 0)
            if name:
                link = f"[{name}]({url})" if url else name
                lines.append(f"- {link} (credibility: {cred:.0%})")
        lines.append("")

    return "\n".join(lines)


def _build_shock_section(shock_result: Optional[Dict[str, Any]]) -> str:
    if not shock_result or shock_result.get("status") != "done":
        return ""
    data = shock_result.get("data", {})
    signal = shock_result.get("signal", "neutral")
    if signal != "bearish":
        return ""  # Only show shock section if anomalies detected

    lines = [
        "## ⚠️ Risk & Shock Detection",
        "",
        "**Anomalies Detected — Exercise Extra Caution**",
        "",
    ]

    alerts = data.get("alerts", [])
    for alert in alerts[:5]:
        lines.append(f"- ⚠️ {alert}")
    lines.append("")

    drawdown = data.get("drawdown", {})
    if drawdown.get("current_drawdown_pct") is not None:
        lines.append(
            f"**Current Drawdown:** {drawdown.get('current_drawdown_pct', 0):.2f}% "
            f"from peak ₹{drawdown.get('peak_price', 0):,.2f}"
        )
        lines.append("")

    return "\n".join(lines)


def _build_action_plan(
    signal: str,
    ta_result: Optional[Dict[str, Any]],
    oi_result: Optional[Dict[str, Any]],
    pred_result: Optional[Dict[str, Any]],
    symbol: str,
) -> str:
    """Build a concrete action plan with entry, stop, target levels."""
    sr = (ta_result or {}).get("data", {}).get("support_resistance", {})
    price_range = (pred_result or {}).get("data", {}).get("price_range_5d", {})
    oi_walls = (oi_result or {}).get("data", {}).get("oi_walls", {})

    current = (ta_result or {}).get("data", {}).get("current_price")
    if not current:
        current = price_range.get("current_price", 0)

    lines = [f"## 🎯 Action Plan — {symbol}", ""]

    if not current:
        lines.append("*Insufficient data for action plan.*")
        return "\n".join(lines)

    if signal == "bullish":
        entry = current
        stop = sr.get("immediate_support") or (current * 0.97)
        target1 = sr.get("immediate_resistance") or (current * 1.03)
        target2 = price_range.get("predicted_high") or (current * 1.06)
        lines += [
            f"**Bullish Setup:**",
            f"- 🟢 Entry Zone: ₹{entry:,.2f} (current) or dips to ₹{stop * 1.01:,.2f}",
            f"- 🔴 Stop Loss: ₹{stop:,.2f} (below support)",
            f"- 🎯 Target 1: ₹{target1:,.2f}",
            f"- 🎯 Target 2: ₹{target2:,.2f} (ML predicted high)",
            f"- 📐 Risk/Reward: 1:{((target1 - entry) / max(entry - stop, 1)):.1f}",
        ]
    elif signal == "bearish":
        entry = current
        stop = sr.get("immediate_resistance") or (current * 1.03)
        target1 = sr.get("immediate_support") or (current * 0.97)
        target2 = price_range.get("predicted_low") or (current * 0.94)
        lines += [
            f"**Bearish Setup:**",
            f"- 🔴 Short Entry: ₹{entry:,.2f} or bounce to ₹{stop * 0.99:,.2f}",
            f"- 🟢 Stop Loss: ₹{stop:,.2f} (above resistance)",
            f"- 🎯 Target 1: ₹{target1:,.2f}",
            f"- 🎯 Target 2: ₹{target2:,.2f} (ML predicted low)",
            f"- 📐 Risk/Reward: 1:{((entry - target1) / max(stop - entry, 1)):.1f}",
        ]
    else:
        lines += [
            f"**Neutral — Wait for Breakout:**",
            f"- 👁️ Watch for break above ₹{sr.get('immediate_resistance', current * 1.02):,.2f} (bullish)",
            f"- 👁️ Or break below ₹{sr.get('immediate_support', current * 0.98):,.2f} (bearish)",
            f"- ⏳ Patience — no clear edge currently.",
        ]

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main ReportAgent
# ---------------------------------------------------------------------------


class ReportAgent(BaseSwarmAgent):
    """
    Asset sub-agent that generates the final structured report.

    Reads all prior agent results, builds rule-based sections,
    then uses LLM to produce a cohesive narrative tying everything together.
    The LLM synthesis is optional — the rule-based sections are always built
    first as fallback.
    """

    AGENT_TYPE = AGENT_TYPE
    DEFAULT_TIMEOUT_S = 120.0

    async def execute(self, message: SwarmMessage) -> AgentResult:
        payload = message.payload
        query: str = str(payload.get("query", ""))
        intent: str = str(payload.get("intent", "general"))
        entities: Dict[str, Any] = dict(payload.get("entities", {}))
        analysis: Dict[str, Any] = dict(payload.get("analysis", {}))
        agent_results: List[Dict[str, Any]] = list(payload.get("agent_results", []))
        key_findings: List[str] = list(payload.get("key_findings", []))
        warnings: List[str] = list(payload.get("warnings", []))

        self._log.info(
            f"ReportAgent: query={query[:60]!r} | "
            f"intent={intent} | agents={len(agent_results)}"
        )

        overall_signal = analysis.get("signal", "neutral")
        overall_conf = float(analysis.get("confidence", 0.5))

        # ── Extract typed agent results ───────────────────────────────────
        def _get(atype: str) -> Optional[Dict[str, Any]]:
            for r in agent_results:
                if r.get("agent_type") == atype and r.get("status") == "done":
                    return r
            return None

        ta_result = _get("technical_analysis")
        oi_result = _get("oi_analysis")
        global_result = _get("global_market")
        pred_result = _get("prediction")
        fund_result = _get("fundamentals")
        sent_result = _get("sentiment")
        web_result = _get("web_research")
        shock_result = _get("shock_detection")

        # ── Determine primary symbol ──────────────────────────────────────
        symbols = entities.get("symbols", [])
        primary_symbol = (
            symbols[0]
            if symbols
            else (ta_result or {}).get("data", {}).get("symbol", "")
            or (pred_result or {}).get("data", {}).get("symbol", "")
            or "NIFTY"
        )

        # ── Build rule-based sections ─────────────────────────────────────
        sections: Dict[str, str] = {}

        sections["exec_summary"] = _build_exec_summary(
            query, overall_signal, overall_conf, key_findings, warnings
        )
        sections["technical"] = _build_ta_section(ta_result)
        sections["options_oi"] = _build_oi_section(oi_result)
        sections["global"] = _build_global_section(global_result)
        sections["prediction"] = _build_prediction_section(pred_result)
        sections["fundamentals"] = _build_fundamentals_section(fund_result)
        sections["sentiment"] = _build_sentiment_section(sent_result)
        sections["research"] = _build_web_research_section(web_result)
        sections["shock"] = _build_shock_section(shock_result)
        sections["action_plan"] = _build_action_plan(
            overall_signal, ta_result, oi_result, pred_result, primary_symbol
        )

        # ── Try LLM synthesis for executive narrative ─────────────────────
        llm_narrative = await self._llm_narrative(
            query=query,
            intent=intent,
            symbol=primary_symbol,
            signal=overall_signal,
            confidence=overall_conf,
            key_findings=key_findings,
            warnings=warnings,
            sections=sections,
            agent_results=agent_results,
        )

        # ── Assemble final report ─────────────────────────────────────────
        report_md = self._assemble_report(
            query=query,
            symbol=primary_symbol,
            signal=overall_signal,
            confidence=overall_conf,
            llm_narrative=llm_narrative,
            sections=sections,
            agent_results=agent_results,
        )

        # ── Conversational final response ─────────────────────────────────
        final_response = llm_narrative if llm_narrative else report_md[:800]

        agents_summary = f"Total sub-agents used: {len(agent_results)} research + 1 report = {len(agent_results) + 1} total"

        return self._ok(
            data={
                "report_md": report_md,
                "final_response": final_response,
                "sections": {k: v for k, v in sections.items() if v},
                "symbol": primary_symbol,
                "signal": overall_signal,
                "confidence": overall_conf,
                "agents_used": len(agent_results) + 1,
                "agents_summary": agents_summary,
                "key_findings": key_findings,
                "warnings": warnings,
            },
            summary=final_response[:500],
            signal=overall_signal,
            confidence=overall_conf,
            metadata={
                "agent": AGENT_TYPE,
                "symbol": primary_symbol,
                "intent": intent,
                "sections_built": [k for k, v in sections.items() if v],
            },
        )

    # ────────────────────────────────────────────────────────────────────────
    # LLM narrative synthesis
    # ────────────────────────────────────────────────────────────────────────

    async def _llm_narrative(
        self,
        query: str,
        intent: str,
        symbol: str,
        signal: str,
        confidence: float,
        key_findings: List[str],
        warnings: List[str],
        sections: Dict[str, str],
        agent_results: List[Dict[str, Any]],
    ) -> str:
        """
        Use LLM to write a cohesive narrative that ties all sections together.
        Returns the conversational answer (not the full markdown report).
        """
        try:
            model = self.tools.get_model("reasoning")

            findings_text = "\n".join(f"• {f}" for f in key_findings[:6])
            warnings_text = "\n".join(f"⚠️ {w}" for w in warnings[:3])

            # Build rich agent context — more chars per agent = better synthesis
            agent_context = "\n\n".join(
                f"**{r.get('task_name', r.get('agent_type', 'Agent'))}** "
                f"[Signal: {r.get('signal', 'neutral').upper()} | Conf: {r.get('confidence', 0.5):.0%}]:\n{r.get('summary', '')[:1500]}"
                for r in agent_results
                if r.get("status") == "done" and r.get("summary")
            )

            system = """You are Daddy's AI — the world's most thorough financial research analyst. You have just deployed a full swarm of specialised AI agents and collected comprehensive data on this company/topic. Your job is to write an EXHAUSTIVE, EDUCATIONAL deep-research report.

## MANDATORY REQUIREMENTS:

**Length:** MINIMUM 5000 words. More is better. This is a premium research report — do not truncate.

**Audience:** Write as if explaining to someone intelligent but new to investing. Define every term you use. Don't just say "RSI is oversold" — explain what RSI is, why it matters, what oversold means for the stock price, and what a smart investor would do about it.

**Structure (use ## headers for each):**

## 1. Quick Verdict
Open with one clear sentence: BUY / SELL / HOLD at ₹X. Then a 2-sentence explanation of WHY in plain English.

## 2. What Does This Company Actually Do?
Explain the company's business model from scratch. What does it sell? Who are its customers? How does it make money? Why does this business exist? History, current market position, competitors.

## 3. The Numbers Tell a Story — Current Price & Technical Picture
Explain EVERY technical indicator you have data for:
- RSI: what it is (a momentum gauge from 0-100), what the current reading means, historical context
- MACD: what it is (moving average convergence/divergence), what the signal means for traders  
- Supertrend: what it is, what the current reading tells us
- Bollinger Bands: what they are, what it means to be above/below/at the middle band
- Support/Resistance: explain like drawing a floor and ceiling on a graph — WHY are these levels important?
- Volume: is buying or selling intensity rising or falling?
- EMA crossovers: what's a golden cross vs death cross?
For each: state the value → explain what it means → explain what it predicts → explain what an investor should watch for.

## 4. The News — What's Happening Right Now?
Summarise every piece of recent news. For each story:
- What happened?
- Why does it matter to the company's business?
- Is it good, bad, or neutral for the stock price, and WHY?
- What's the market's likely reaction?

## 5. Fundamentals Deep Dive
If available: explain PE ratio (what a fair PE is for this sector), Price-to-Book, Return on Equity, debt levels, revenue growth. If not available, explain what you'd look for and why.

## 6. Risks — What Could Go WRONG?
Be specific. List at least 5 concrete risks with:
- What is the risk?
- How likely is it (low/medium/high)?
- What would happen to the stock if it materialises?
- At what price level should an investor cut losses?

## 7. The Bull Case — Why It Could Go UP
At least 400 words explaining every reason a bullish investor would buy this. Specific price targets with reasoning.

## 8. The Bear Case — Why It Could Go DOWN
At least 400 words explaining every reason a skeptical investor would avoid or short this.

## 9. Analyst Verdict & Action Plan
Clear, specific:
- Signal: BUY / HOLD / SELL
- Entry zone: ₹X – ₹Y  
- Stop loss: ₹Z (explain why this level)
- Target 1: ₹A (explain the reasoning — what needs to happen for this target?)
- Target 2: ₹B (longer-term — what's the multi-year bull scenario?)
- Time horizon: weeks / months / years?

## 10. Key Levels to Watch
A table of critical price levels and what each means.

**Formatting rules:**
- Use ₹ for all Indian prices
- Bold **key numbers** and **key terms** on first use
- Cite data sources: "According to NSE data..." or "As reported by Economic Times..."
- Tables for comparing data points
- Never use vague phrases like "mixed signals" or "market volatility" without quantifying them
- Every paragraph must contain at least one specific number

⚠️ End with: *This report is for informational purposes only. Not financial advice. Consult a SEBI-registered investment advisor before investing.*"""

            user_prompt = f"""User asked: "{query}"
Intent: {intent} | Primary Symbol: {symbol}
Overall Multi-Agent Signal: {signal.upper()} | Aggregate Confidence: {confidence:.0%}

KEY FINDINGS FROM ALL AGENTS:
{findings_text or "(Technical analysis data below)"}

{f'RISK WARNINGS:{chr(10)}{warnings_text}' if warnings_text else ''}

DETAILED AGENT RESEARCH DATA (use ALL of this in your report):
{agent_context or '(No agent data — use your training knowledge for a comprehensive analysis)'}

Write the COMPLETE research report following ALL 10 sections above.
Minimum 5000 words. Define every term. Be specific with every number.
Explain the WHY behind every data point as if teaching someone who has never invested before.
Total agents deployed: {len(agent_results)}"""

            response = await self.tools.call_openrouter(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=8000,
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            self._log.warning(f"LLM narrative failed: {exc}")
            return ""

    # ────────────────────────────────────────────────────────────────────────
    # Report assembly
    # ────────────────────────────────────────────────────────────────────────

    def _assemble_report(
        self,
        query: str,
        symbol: str,
        signal: str,
        confidence: float,
        llm_narrative: str,
        sections: Dict[str, str],
        agent_results: List[Dict[str, Any]],
    ) -> str:
        """
        Assemble the full markdown report from all sections.
        """
        emoji = _signal_emoji(signal)
        timestamp = datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")
        n_agents = len(agent_results) + 1  # +1 for this report agent

        header = f"""# {emoji} DaddysAI Research Report — {symbol}
*Generated: {timestamp} | Agents deployed: {n_agents} | Signal: {signal.upper()} {_conf_bar(confidence)}*

---

**Query:** *{query}*

---
"""

        parts = [header]

        # LLM narrative goes first (if available)
        if llm_narrative:
            parts.append("## 💡 Analysis\n")
            parts.append(llm_narrative)
            parts.append("\n---\n")

        # Then all rule-based sections
        section_order = [
            "exec_summary",
            "technical",
            "options_oi",
            "prediction",
            "global",
            "fundamentals",
            "sentiment",
            "research",
            "shock",
            "action_plan",
        ]

        for key in section_order:
            section_content = sections.get(key, "")
            if section_content and section_content.strip():
                parts.append(section_content)
                parts.append("\n---\n")

        # Footer
        footer = f"""*⚠️ This report is generated by DaddysAI's autonomous multi-agent system for informational purposes only.*
*It does NOT constitute financial advice. Always consult a SEBI-registered investment advisor before making investment decisions.*
*Past performance does not guarantee future results. Markets are subject to risk.*

*Report generated by {n_agents} AI agents | DaddysAI Phase 2 Swarm Intelligence*"""

        parts.append(footer)

        return "\n".join(parts)
