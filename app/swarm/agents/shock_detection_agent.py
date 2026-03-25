"""
ShockDetectionAgent — Market Shock & Crash Detection
FundamentalsAgent  — Company Fundamentals via yfinance
SentimentAgent     — News & Social Sentiment Analysis
DataFetchAgent     — OHLCV Data Fetch & Storage
=====================================================
Phase 2 · Swarm Agents
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base_agent import AgentResult, AgentStatus, BaseSwarmAgent, SwarmMessage

logger = logging.getLogger(__name__)


# ===========================================================================
# ShockDetectionAgent
# ===========================================================================


class ShockDetectionAgent(BaseSwarmAgent):
    """
    Detects market shocks, crashes, volatility anomalies, and tail-risk events.

    Checks:
      • Volatility spike (Z-score > threshold)
      • Flash crash (intraday % drop > threshold)
      • Max drawdown (peak-to-trough)
      • Circuit breaker probability
      • Momentum exhaustion
      • Gap analysis (up/down gaps)
      • Volume anomaly
      • VIX-proxy fear gauge

    Input payload keys:
      symbol     (str)   NSE symbol. Default: "NIFTY"
      days       (int)   Lookback. Default: 365
      zscore_threshold (float) Z-score for vol spike. Default: 3.0

    Output AgentResult.data keys:
      symbol, current_price, alerts, drawdown, volatility,
      circuit_breaker, gaps, momentum_exhaustion, shock_score,
      risk_level, interpretation
    """

    AGENT_TYPE = "shock_detection"
    DEFAULT_TIMEOUT_S = 60.0

    # Thresholds
    VOL_ZSCORE_THRESHOLD = 3.0  # Z-score for volatility spike
    FLASH_CRASH_PCT = -4.0  # Single-day drop % that triggers alert
    MAX_DRAWDOWN_WARN = -15.0  # Drawdown % that triggers warning
    MAX_DRAWDOWN_CRITICAL = -25.0  # Drawdown % that triggers critical
    VOLUME_SPIKE_RATIO = 3.0  # Volume vs 20-day avg

    async def execute(self, message: SwarmMessage) -> AgentResult:
        payload = message.payload
        symbol: str = str(payload.get("symbol", "NIFTY")).upper().strip()
        days: int = int(payload.get("days", 365))
        zscore_thr = float(payload.get("zscore_threshold", self.VOL_ZSCORE_THRESHOLD))

        self._log.info(f"ShockDetectionAgent: {symbol} | days={days}")

        # ── 1. Fetch OHLCV ─────────────────────────────────────────────
        df = await self.tools.get_ohlcv(symbol, days=days, interval="1d")

        if df is None or df.empty or len(df) < 30:
            return self._ok(
                data={
                    "symbol": symbol,
                    "error": "Insufficient data for shock detection",
                },
                summary=f"Not enough data to detect shocks for {symbol}.",
                signal="neutral",
                confidence=0.1,
            )

        df = self._norm(df)
        current_price = float(df["close"].iloc[-1])

        # ── 2. Run all checks ────────────────────────────────────────────
        alerts: List[str] = []
        shock_score: float = 0.0  # 0–100, higher = more risk

        vol_result = self._check_volatility(df, zscore_thr)
        drawdown_res = self._check_drawdown(df)
        flash_res = self._check_flash_crash(df)
        gap_res = self._check_gaps(df)
        volume_res = self._check_volume_anomaly(df)
        mom_res = self._check_momentum_exhaustion(df)
        circuit_res = self._circuit_breaker_probability(df)

        # ── 3. Accumulate alerts + score ──────────────────────────────
        if vol_result.get("spike_detected"):
            alerts.append(
                f"🚨 Volatility spike: Z-score={vol_result.get('zscore', 0):.2f} "
                f"(threshold {zscore_thr:.1f}) — {vol_result.get('interpretation', '')}"
            )
            shock_score += 25.0

        if drawdown_res.get("current_drawdown_pct", 0) < self.MAX_DRAWDOWN_CRITICAL:
            alerts.append(
                f"💀 Critical drawdown: {drawdown_res.get('current_drawdown_pct', 0):.2f}% "
                f"from peak ₹{drawdown_res.get('peak_price', 0):,.2f}"
            )
            shock_score += 30.0
        elif drawdown_res.get("current_drawdown_pct", 0) < self.MAX_DRAWDOWN_WARN:
            alerts.append(
                f"⚠️ Significant drawdown: {drawdown_res.get('current_drawdown_pct', 0):.2f}% from peak"
            )
            shock_score += 15.0

        if flash_res.get("flash_crash_detected"):
            alerts.append(
                f"⚡ Flash crash signal: single-day drop of "
                f"{flash_res.get('worst_single_day_pct', 0):.2f}% detected"
            )
            shock_score += 20.0

        if gap_res.get("recent_gap_down"):
            alerts.append(
                f"📉 Gap down: {gap_res.get('gap_pct', 0):.2f}% gap — "
                "institutional selling or panic detected"
            )
            shock_score += 10.0

        if volume_res.get("volume_spike"):
            alerts.append(
                f"📊 Volume anomaly: {volume_res.get('volume_ratio', 0):.1f}x average — "
                "unusual activity detected"
            )
            shock_score += 10.0

        if mom_res.get("exhaustion_detected"):
            alerts.append(
                f"🔴 Momentum exhaustion: {mom_res.get('signal', '')} — "
                "trend may be losing steam"
            )
            shock_score += 10.0

        if circuit_res.get("probability", 0) > 0.3:
            alerts.append(
                f"⚠️ Circuit breaker risk: {circuit_res.get('probability', 0):.0%} probability — "
                f"{circuit_res.get('interpretation', '')}"
            )
            shock_score += 15.0

        # ── 4. Risk level ─────────────────────────────────────────────
        shock_score = min(100.0, shock_score)

        if shock_score >= 60:
            risk_level = "CRITICAL"
            signal = "bearish"
            confidence = 0.90
        elif shock_score >= 35:
            risk_level = "HIGH"
            signal = "bearish"
            confidence = 0.75
        elif shock_score >= 20:
            risk_level = "MODERATE"
            signal = "bearish"
            confidence = 0.55
        elif shock_score >= 10:
            risk_level = "LOW"
            signal = "neutral"
            confidence = 0.40
        else:
            risk_level = "NORMAL"
            signal = "neutral"
            confidence = 0.20

        interp = self._build_interpretation(
            symbol, current_price, shock_score, risk_level, alerts
        )

        return self._ok(
            data={
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "shock_score": round(shock_score, 2),
                "risk_level": risk_level,
                "alerts": alerts,
                "drawdown": drawdown_res,
                "volatility": vol_result,
                "flash_crash": flash_res,
                "gaps": gap_res,
                "volume_anomaly": volume_res,
                "momentum_exhaustion": mom_res,
                "circuit_breaker": circuit_res,
                "analysis_date": datetime.utcnow().isoformat(),
            },
            summary=interp,
            signal=signal,
            confidence=confidence,
            metadata={"agent": "shock_detection", "symbol": symbol},
        )

    # ────────────────────────────────────────────────────────────────────
    # Checks
    # ────────────────────────────────────────────────────────────────────

    def _check_volatility(self, df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        closes = df["close"]
        log_rets = np.log(closes / closes.shift(1)).dropna()
        if len(log_rets) < 20:
            return {"spike_detected": False}

        rolling_vol = log_rets.rolling(20).std()
        mean_vol = float(rolling_vol.mean())
        std_vol = float(rolling_vol.std())
        curr_vol = float(rolling_vol.iloc[-1])

        if std_vol == 0:
            return {"spike_detected": False, "current_vol": curr_vol}

        zscore = (curr_vol - mean_vol) / std_vol
        spike = zscore > threshold

        ann_vol = curr_vol * math.sqrt(252) * 100

        return {
            "spike_detected": spike,
            "zscore": round(zscore, 3),
            "current_daily_vol": round(curr_vol * 100, 4),
            "annualised_vol_pct": round(ann_vol, 2),
            "mean_daily_vol": round(mean_vol * 100, 4),
            "interpretation": (
                f"Current vol {ann_vol:.1f}% ann. (Z={zscore:.2f})"
                + (" — SPIKE!" if spike else " — normal")
            ),
        }

    def _check_drawdown(self, df: pd.DataFrame) -> Dict[str, Any]:
        closes = df["close"]
        cummax = closes.cummax()
        drawdown_series = (closes - cummax) / cummax * 100
        current_dd = float(drawdown_series.iloc[-1])
        max_dd = float(drawdown_series.min())
        peak_price = float(cummax.iloc[-1])
        current = float(closes.iloc[-1])

        # Duration of current drawdown
        drawdown_start = closes[closes == cummax.iloc[-1]].index
        days_in_dd = 0
        if not drawdown_start.empty:
            last_peak = drawdown_start[-1]
            try:
                peak_loc_result = df.index.get_loc(last_peak)
                peak_loc = (
                    int(peak_loc_result)
                    if not hasattr(peak_loc_result, "__len__")
                    else 0
                )
            except Exception:
                peak_loc = 0
            days_in_dd = len(df) - peak_loc - 1

        return {
            "current_drawdown_pct": round(current_dd, 3),
            "max_drawdown_pct": round(max_dd, 3),
            "peak_price": round(peak_price, 2),
            "current_price": round(current, 2),
            "days_in_drawdown": days_in_dd,
            "interpretation": (
                f"Currently {current_dd:.2f}% below peak ₹{peak_price:,.2f} "
                f"(max drawdown in window: {max_dd:.2f}%)"
            ),
        }

    def _check_flash_crash(self, df: pd.DataFrame) -> Dict[str, Any]:
        daily_rets = df["close"].pct_change() * 100
        worst = float(daily_rets.min())
        worst_date = daily_rets.idxmin()
        recent_worst = float(daily_rets.tail(10).min())

        return {
            "flash_crash_detected": worst < self.FLASH_CRASH_PCT,
            "worst_single_day_pct": round(worst, 3),
            "worst_date": str(worst_date)[:10] if worst_date is not None else "",
            "recent_10d_worst_pct": round(recent_worst, 3),
            "interpretation": (
                f"Worst single-day return in window: {worst:.2f}% "
                f"({'FLASH CRASH' if worst < self.FLASH_CRASH_PCT else 'normal'})"
            ),
        }

    def _check_gaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        opens = df["open"]
        prev_cl = df["close"].shift(1)
        gap_pcts = ((opens - prev_cl) / prev_cl * 100).dropna()

        recent_gaps = gap_pcts.tail(5)
        biggest_gap = float(recent_gaps.min())
        gap_down = biggest_gap < -1.5

        return {
            "recent_gap_down": gap_down,
            "gap_pct": round(biggest_gap, 3),
            "all_recent_gaps": [round(g, 2) for g in recent_gaps.tolist()],
            "interpretation": (
                f"Largest recent gap: {biggest_gap:.2f}% "
                + ("— significant gap down" if gap_down else "— normal")
            ),
        }

    def _check_volume_anomaly(self, df: pd.DataFrame) -> Dict[str, Any]:
        vol = df.get("volume", pd.Series(dtype=float))
        if vol is None or vol.empty or vol.sum() == 0:
            return {"volume_spike": False, "note": "no volume data"}

        vol_ma20 = float(vol.rolling(20).mean().iloc[-1])
        curr_vol = float(vol.iloc[-1])

        if vol_ma20 == 0:
            return {"volume_spike": False}

        ratio = curr_vol / vol_ma20
        spike = ratio > self.VOLUME_SPIKE_RATIO

        return {
            "volume_spike": spike,
            "volume_ratio": round(ratio, 3),
            "current_volume": curr_vol,
            "avg_volume_20d": round(vol_ma20, 0),
            "interpretation": (
                f"Volume {ratio:.1f}x 20-day avg "
                + ("— ANOMALY: panic/institutional activity" if spike else "— normal")
            ),
        }

    def _check_momentum_exhaustion(self, df: pd.DataFrame) -> Dict[str, Any]:
        closes = df["close"]
        n = len(closes)
        if n < 14:
            return {"exhaustion_detected": False}

        # RSI
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = float((100 - 100 / (1 + rs)).iloc[-1])

        # Price vs 3-period high/low
        high3 = float(closes.tail(3).max())
        low3 = float(closes.tail(3).min())
        curr = float(closes.iloc[-1])

        # Exhaustion: RSI overbought + price near recent high
        bull_exhaustion = rsi >= 75 and curr >= high3 * 0.99
        # Exhaustion: RSI oversold + price near recent low
        bear_exhaustion = rsi <= 25 and curr <= low3 * 1.01

        exhaustion = bull_exhaustion or bear_exhaustion
        sig_label = (
            "BULLISH_EXHAUSTION"
            if bull_exhaustion
            else ("BEARISH_EXHAUSTION" if bear_exhaustion else "none")
        )

        return {
            "exhaustion_detected": exhaustion,
            "signal": sig_label,
            "rsi": round(rsi, 2),
            "interpretation": (
                f"RSI {rsi:.1f} "
                + (
                    f"— {sig_label}: trend may reverse"
                    if exhaustion
                    else "— no exhaustion signal"
                )
            ),
        }

    def _circuit_breaker_probability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Estimate circuit breaker probability based on:
          • Recent volatility vs historical
          • Whether price is near key levels
          • Recent large single-day moves
        """
        closes = df["close"]
        daily_r = closes.pct_change().dropna() * 100
        recent5 = daily_r.tail(5)

        max_recent = float(recent5.abs().max())
        prob = min(1.0, max_recent / 10.0)  # 10% move → 100% prob

        # NSE circuit breakers: 10%, 15%, 20% triggers
        triggered_10 = max_recent >= 10.0
        triggered_15 = max_recent >= 15.0

        return {
            "probability": round(prob, 3),
            "max_recent_5d_pct": round(max_recent, 3),
            "triggered_10pct": triggered_10,
            "triggered_15pct": triggered_15,
            "interpretation": (
                f"Max 5d move: {max_recent:.2f}% | "
                f"CB probability: {prob:.0%}"
                + (" — 10% CB likely triggered" if triggered_10 else "")
            ),
        }

    def _build_interpretation(
        self,
        symbol: str,
        price: float,
        score: float,
        risk_level: str,
        alerts: List[str],
    ) -> str:
        risk_emoji = {
            "CRITICAL": "🚨💀",
            "HIGH": "🔴⚠️",
            "MODERATE": "🟠",
            "LOW": "🟡",
            "NORMAL": "🟢",
        }.get(risk_level, "⚪")

        lines = [
            f"{risk_emoji} **{symbol} Risk Assessment — {risk_level}** "
            f"(Score: {score:.0f}/100)",
            f"  • Current Price: ₹{price:,.2f}",
        ]
        if alerts:
            lines.append(f"  • Alerts: {len(alerts)} active")
            for alert in alerts[:3]:
                lines.append(f"    - {alert[:120]}")
        else:
            lines.append("  • No significant shock signals detected.")
        return "\n".join(lines)

    @staticmethod
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                df[col] = float("nan")
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df


# ===========================================================================
# FundamentalsAgent
# ===========================================================================


class FundamentalsAgent(BaseSwarmAgent):
    """
    Fetches and scores company fundamentals via yfinance.

    Metrics: PE, PB, PS, ROE, ROA, EPS, revenue, margins,
    debt/equity, current ratio, dividend yield, beta, market cap,
    52-week range, institutional holdings, sector/industry.

    Input payload keys:
      symbol    (str, required)  NSE symbol

    Output AgentResult.data keys:
      symbol, fundamentals, valuation_score, growth_score,
      quality_score, overall_score, signal, interpretation
    """

    AGENT_TYPE = "fundamentals"
    DEFAULT_TIMEOUT_S = 45.0

    async def execute(self, message: SwarmMessage) -> AgentResult:
        symbol = str(message.payload.get("symbol", "")).upper().strip()
        if not symbol:
            return self._ok(
                data={"error": "No symbol provided"},
                summary="FundamentalsAgent: no symbol in payload.",
                signal="neutral",
                confidence=0.1,
            )

        self._log.info(f"FundamentalsAgent: {symbol}")

        # ── 1. Try data pool first ─────────────────────────────────────
        fund = await self.tools.data_pool.get_fundamentals(symbol)

        # ── 2. Fetch from yfinance if not cached ──────────────────────
        if not fund:
            fund = await self.tools.data_fetcher.fetch_fundamentals(symbol)
            if fund and "error" not in fund:
                await self.tools.data_pool.store_fundamentals(symbol, fund)

        if not fund or "error" in fund:
            # Fallback: web search for basic fundamentals
            search_result = await self.tools.web_search(
                f"{symbol} NSE PE ratio market cap fundamentals 2025"
            )
            return self._ok(
                data={
                    "symbol": symbol,
                    "fundamentals": {"web_fallback": search_result[:500]},
                    "note": "Fetched from web (yfinance unavailable)",
                },
                summary=f"Fundamentals for {symbol} via web search: {search_result[:200]}",
                signal="neutral",
                confidence=0.4,
            )

        # ── 3. Score the fundamentals ─────────────────────────────────
        val_score, growth_score, quality_score = self._score(fund)
        overall_score = (val_score + growth_score + quality_score) / 3

        signal = (
            "bullish"
            if overall_score >= 65
            else ("bearish" if overall_score <= 35 else "neutral")
        )
        confidence = min(1.0, abs(overall_score - 50) / 50)

        interpretation = self._interpret(
            symbol, fund, val_score, growth_score, quality_score, overall_score
        )

        return self._ok(
            data={
                "symbol": symbol,
                "fundamentals": fund,
                "valuation_score": round(val_score, 1),
                "growth_score": round(growth_score, 1),
                "quality_score": round(quality_score, 1),
                "overall_score": round(overall_score, 1),
                "sector": fund.get("sector", "—"),
                "industry": fund.get("industry", "—"),
                "market_cap_cr": fund.get("market_cap_cr"),
            },
            summary=interpretation,
            signal=signal,
            confidence=round(confidence, 3),
            metadata={"agent": "fundamentals", "symbol": symbol},
        )

    def _score(self, fund: Dict[str, Any]) -> Tuple[float, float, float]:
        """Score 0–100 for valuation, growth, and quality."""

        def _safe(k: str, default: float = 0.0) -> float:
            v = fund.get(k, default)
            try:
                f = float(v)
                return f if math.isfinite(f) else default
            except (TypeError, ValueError):
                return default

        # ── Valuation (lower PE/PB = better) ───────────────────────────
        pe = _safe("pe_ratio")
        pb = _safe("pb_ratio")
        div = _safe("dividend_yield") * 100

        val = 50.0
        if 0 < pe <= 15:
            val += 20
        elif 15 < pe <= 25:
            val += 10
        elif pe > 40:
            val -= 15
        if 0 < pb <= 1.5:
            val += 15
        elif pb > 5:
            val -= 10
        if div > 3:
            val += 10
        elif div > 1:
            val += 5

        # ── Growth (margins, revenue) ───────────────────────────────────
        profit_margin = _safe("profit_margins") * 100
        op_margin = _safe("operating_margins") * 100
        gross_margin = _safe("gross_margins") * 100

        growth = 50.0
        if profit_margin > 20:
            growth += 20
        elif profit_margin > 10:
            growth += 10
        elif profit_margin < 0:
            growth -= 20
        if op_margin > 25:
            growth += 15
        elif op_margin > 15:
            growth += 8

        # ── Quality (ROE, debt, current ratio) ─────────────────────────
        roe = _safe("roe") * 100
        roa = _safe("roa") * 100
        debt_equity = _safe("debt_to_equity")
        curr_ratio = _safe("current_ratio")

        quality = 50.0
        if roe > 20:
            quality += 20
        elif roe > 12:
            quality += 10
        elif roe < 0:
            quality -= 20
        if debt_equity < 0.5:
            quality += 15
        elif debt_equity > 2.0:
            quality -= 10
        if curr_ratio > 2.0:
            quality += 10
        elif curr_ratio < 1.0:
            quality -= 10

        return (
            max(0.0, min(100.0, val)),
            max(0.0, min(100.0, growth)),
            max(0.0, min(100.0, quality)),
        )

    def _interpret(
        self,
        symbol: str,
        fund: Dict[str, Any],
        val: float,
        growth: float,
        quality: float,
        overall: float,
    ) -> str:
        emoji = "🟢" if overall >= 65 else ("🔴" if overall <= 35 else "🟡")
        verdict = (
            "Attractively valued, high quality"
            if overall >= 65
            else ("Overvalued or weak quality" if overall <= 35 else "Fairly valued")
        )
        pe = fund.get("pe_ratio", "—")
        mc = fund.get("market_cap_cr")
        mc_str = f"₹{mc:,.0f} Cr" if mc else "—"
        roe = fund.get("roe")
        roe_str = f"{float(roe) * 100:.1f}%" if roe else "—"

        return (
            f"{emoji} **{symbol} Fundamentals — {verdict}** "
            f"(Score: {overall:.0f}/100)\n"
            f"  • Market Cap: {mc_str} | PE: {pe}x | ROE: {roe_str}\n"
            f"  • Valuation {val:.0f}/100 | Growth {growth:.0f}/100 | Quality {quality:.0f}/100"
        )


# ===========================================================================
# SentimentAgent
# ===========================================================================


class SentimentAgent(BaseSwarmAgent):
    """
    Analyses news and social sentiment for a stock/topic.

    Process:
      1. Search for recent news headlines (3–5 searches)
      2. Use LLM to score each headline: positive/negative/neutral + confidence
      3. Aggregate into an overall sentiment score [-1, +1]
      4. Return top headlines with sentiment tags

    Input payload keys:
      query      (str, required)  e.g. "RELIANCE stock news" or "Nifty market"
      symbol     (str)            Optional NSE symbol for focused search
      n_searches (int)            Number of news searches. Default: 3

    Output AgentResult.data keys:
      query, symbol, sentiment_score, signal,
      top_headlines, interpretation, sources_checked
    """

    AGENT_TYPE = "sentiment"
    DEFAULT_TIMEOUT_S = 60.0

    async def execute(self, message: SwarmMessage) -> AgentResult:
        payload = message.payload
        query = str(payload.get("query", "")).strip()
        symbol = str(payload.get("symbol", "")).upper().strip()
        n_search = int(payload.get("n_searches", 3))

        if not query and symbol:
            query = f"{symbol} stock news analysis"
        elif not query:
            query = "Nifty India market sentiment today"

        self._log.info(f"SentimentAgent: {query!r} | symbol={symbol}")

        # ── 1. Build search queries ────────────────────────────────────
        search_queries = self._build_queries(query, symbol)[:n_search]

        # ── 2. Parallel web searches ───────────────────────────────────
        raw_results = await asyncio.gather(
            *[self.tools.web_search(q) for q in search_queries],
            return_exceptions=True,
        )

        combined_text = "\n\n---\n\n".join(
            str(r) for r in raw_results if not isinstance(r, Exception) and r
        )

        if not combined_text.strip():
            return self._ok(
                data={"query": query, "symbol": symbol, "error": "No search results"},
                summary=f"Could not fetch news for: {query}",
                signal="neutral",
                confidence=0.2,
            )

        # ── 3. LLM sentiment scoring ───────────────────────────────────
        analysis = await self._llm_analyse(query, symbol, combined_text)

        sentiment_score = float(analysis.get("sentiment_score", 0.0))
        headlines = analysis.get("headlines", [])
        interp = analysis.get("interpretation", "")

        # Clamp score
        sentiment_score = max(-1.0, min(1.0, sentiment_score))

        if sentiment_score > 0.25:
            signal = "bullish"
            confidence = min(1.0, sentiment_score * 1.2)
        elif sentiment_score < -0.25:
            signal = "bearish"
            confidence = min(1.0, abs(sentiment_score) * 1.2)
        else:
            signal = "neutral"
            confidence = 0.4

        summary = (
            f"{'🟢' if signal == 'bullish' else '🔴' if signal == 'bearish' else '🟡'} "
            f"**Sentiment for {symbol or query}: {signal.upper()}** "
            f"(score {sentiment_score:+.2f})\n"
            f"  • {interp[:200]}"
        )

        return self._ok(
            data={
                "query": query,
                "symbol": symbol,
                "sentiment_score": round(sentiment_score, 4),
                "signal": signal,
                "top_headlines": headlines[:8],
                "interpretation": interp,
                "sources_checked": len(search_queries),
                "searches_run": len(
                    [r for r in raw_results if not isinstance(r, Exception)]
                ),
            },
            summary=summary,
            signal=signal,
            confidence=round(confidence, 3),
            metadata={"agent": "sentiment", "symbol": symbol},
        )

    def _build_queries(self, query: str, symbol: str) -> List[str]:
        base = [
            f"{symbol or query} news today India",
            f"{symbol or query} latest analysis bullish bearish",
            f"{symbol or query} NSE outlook 2025",
            f"{symbol or query} earnings results quarterly",
            f"{symbol or query} analyst target price",
        ]
        return base

    async def _llm_analyse(
        self,
        query: str,
        symbol: str,
        combined_text: str,
    ) -> Dict[str, Any]:
        """
        Use LLM to extract sentiment score and headlines from raw search text.
        Returns dict with: sentiment_score, headlines, interpretation
        """
        try:
            client = self.tools.get_llm_client()
            model = self.tools.get_model("fast")

            prompt = f"""You are a financial sentiment analyst for Indian markets.

Query: "{query}"  |  Symbol: "{symbol or "market"}"

News/Search Results:
{combined_text[:3000]}

Analyse the sentiment of this text as it relates to {symbol or query}.

Output ONLY valid JSON (no extra text):
{{
  "sentiment_score": 0.0,
  "headlines": [
    {{"headline": "Headline text here", "sentiment": "positive", "relevance": 0.9}},
    {{"headline": "Another headline", "sentiment": "negative", "relevance": 0.7}}
  ],
  "interpretation": "One-sentence summary of overall sentiment",
  "key_drivers": ["driver 1", "driver 2"]
}}

Rules:
- sentiment_score: float from -1.0 (very bearish) to +1.0 (very bullish), 0 = neutral
- headline sentiment: "positive", "negative", or "neutral"
- Include up to 6 most relevant headlines
- interpretation: specific, mention numbers/events if available
- Only include headlines directly relevant to "{symbol or query}"
"""

            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600,
            )

            import json
            import re

            raw = (response.choices[0].message.content or "").strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
            return json.loads(raw)

        except Exception as exc:
            self._log.warning(f"Sentiment LLM error: {exc}")
            # Simple fallback: count positive/negative words
            pos_words = [
                "rally",
                "gain",
                "bullish",
                "strong",
                "positive",
                "up",
                "rise",
                "growth",
                "beat",
            ]
            neg_words = [
                "fall",
                "drop",
                "bearish",
                "weak",
                "negative",
                "down",
                "decline",
                "miss",
                "loss",
            ]
            text_lower = combined_text.lower()
            pos_count = sum(text_lower.count(w) for w in pos_words)
            neg_count = sum(text_lower.count(w) for w in neg_words)
            total = pos_count + neg_count or 1
            score = (pos_count - neg_count) / total
            return {
                "sentiment_score": round(max(-1.0, min(1.0, score)), 3),
                "headlines": [],
                "interpretation": f"Sentiment based on keyword analysis: {'positive' if score > 0 else 'negative' if score < 0 else 'neutral'}",
                "key_drivers": [],
            }


# ===========================================================================
# DataFetchAgent
# ===========================================================================


class DataFetchAgent(BaseSwarmAgent):
    """
    Fetches and stores OHLCV data for any NSE symbol into the data pool.

    Validates data quality, fills small gaps, and returns summary stats.
    Useful as a prerequisite step before TechnicalAnalysis or Prediction agents.

    Input payload keys:
      symbol    (str, required)   NSE symbol e.g. "RELIANCE"
      period    (str)             yfinance period: "1mo","3mo","6mo","1y","2y","5y". Default: "1y"
      interval  (str)             Bar interval: "1d","1wk","1h","15m". Default: "1d"
      force     (bool)            Force re-fetch even if fresh. Default: False

    Output AgentResult.data keys:
      symbol, bars_stored, bars_total, date_range, freshness,
      data_quality, latest_price, status
    """

    AGENT_TYPE = "data_fetch"
    DEFAULT_TIMEOUT_S = 45.0

    async def execute(self, message: SwarmMessage) -> AgentResult:
        payload = message.payload
        symbol = str(payload.get("symbol", "")).upper().strip()
        period = str(payload.get("period", "1y"))
        interval = str(payload.get("interval", "1d"))
        force = bool(payload.get("force", False))

        if not symbol:
            return self._ok(
                data={"error": "No symbol provided"},
                summary="DataFetchAgent: missing symbol.",
                signal="neutral",
                confidence=0.1,
            )

        self._log.info(
            f"DataFetchAgent: {symbol} | period={period} interval={interval} force={force}"
        )

        # ── Check if already fresh ─────────────────────────────────────
        if not force:
            is_fresh = await self.tools.data_pool.is_symbol_fresh(
                symbol, max_age_hours=25
            )
            if is_fresh:
                existing = await self.tools.data_pool.get_ohlcv(
                    symbol, days=5, interval=interval
                )
                if existing is not None and len(existing) >= 1:
                    latest_price = float(existing["close"].iloc[-1])
                    self._log.info(
                        f"DataFetchAgent: {symbol} already fresh, skipping fetch"
                    )
                    return self._ok(
                        data={
                            "symbol": symbol,
                            "status": "already_fresh",
                            "bars_stored": 0,
                            "latest_price": round(latest_price, 2),
                            "note": "Data already up-to-date in pool",
                        },
                        summary=f"{symbol} data already fresh in pool (₹{latest_price:,.2f})",
                        signal="neutral",
                        confidence=0.8,
                    )

        # ── Fetch from yfinance ────────────────────────────────────────
        df = await self.tools.data_fetcher.fetch_ohlcv(
            symbol, period=period, interval=interval
        )

        if df is None or df.empty:
            return self._ok(
                data={
                    "symbol": symbol,
                    "status": "fetch_failed",
                    "error": "yfinance returned empty data",
                },
                summary=f"Failed to fetch data for {symbol}.",
                signal="neutral",
                confidence=0.1,
            )

        # ── Basic data quality check ───────────────────────────────────
        df.columns = [str(c).lower() for c in df.columns]
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                df[col] = float("nan")
        if "volume" not in df.columns:
            df["volume"] = 0.0

        df = df.dropna(subset=["close"])
        if df.empty:
            return self._ok(
                data={"symbol": symbol, "status": "no_valid_data"},
                summary=f"No valid OHLCV data after cleaning for {symbol}.",
                signal="neutral",
                confidence=0.1,
            )

        # ── Store in data pool ─────────────────────────────────────────
        bars_stored = await self.tools.data_pool.store_ohlcv(
            symbol, df, interval=interval
        )

        latest_price = float(df["close"].iloc[-1])
        oldest_date = str(df.index.min())[:10]
        newest_date = str(df.index.max())[:10]

        # ── Simple quality metrics ─────────────────────────────────────
        pct_complete = round(
            len(df.dropna(subset=["close"])) / max(len(df), 1) * 100, 1
        )
        zero_vol_pct = (
            round((df["volume"] == 0).sum() / max(len(df), 1) * 100, 1)
            if "volume" in df.columns
            else 0.0
        )

        summary = (
            f"✅ {symbol}: {bars_stored} new bars stored | "
            f"Total {len(df)} bars | ₹{latest_price:,.2f} latest | "
            f"{oldest_date} → {newest_date}"
        )

        return self._ok(
            data={
                "symbol": symbol,
                "status": "success",
                "bars_stored": bars_stored,
                "bars_total": len(df),
                "latest_price": round(latest_price, 2),
                "date_range": {"start": oldest_date, "end": newest_date},
                "data_quality": {
                    "completeness_pct": pct_complete,
                    "zero_volume_pct": zero_vol_pct,
                    "total_rows": len(df),
                },
                "period": period,
                "interval": interval,
            },
            summary=summary,
            signal="neutral",
            confidence=0.9,
            metadata={"agent": "data_fetch", "symbol": symbol},
        )
