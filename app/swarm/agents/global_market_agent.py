"""
GlobalMarketAgent — Global Financial Markets Monitor
====================================================
Phase 2 · Swarm Agent

Autonomously monitors and analyses all major global markets:
  • US Indices      — S&P 500, Nasdaq, Dow Jones, Russell 2000
  • European        — FTSE 100, DAX, CAC 40, Euro Stoxx 50
  • Asian           — Nikkei 225, Hang Seng, Shanghai Composite, SGX Nifty
  • Indian          — Nifty 50, Sensex, Bank Nifty, India VIX
  • Commodities     — Crude Oil (WTI/Brent), Gold, Silver, Natural Gas
  • Forex           — DXY, USD/INR, EUR/USD, GBP/USD, JPY/USD
  • Bonds           — US 10Y Yield, US 2Y Yield, India 10Y G-Sec
  • Crypto          — Bitcoin, Ethereum (as risk sentiment proxies)
  • Macro           — Fed stance, RBI policy, global PMIs, inflation

Also computes:
  • India market impact score — how global conditions affect Nifty
  • Global risk-on / risk-off regime
  • FII flow prediction from global cues
  • Correlation matrix snippet (key pairs)

Input payload keys:
  focus           (str)   "all" | "us" | "asia" | "commodities" | "india_impact"
  include_macro   (bool)  Fetch macro news. Default: True
  depth           (str)   "quick" | "full". Default: "full"

Output AgentResult.data keys:
  global_snapshot, india_impact, risk_regime, fii_flow_prediction,
  key_alerts, correlations, macro_summary, report_md
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..base_agent import AgentResult, AgentStatus, BaseSwarmAgent, SwarmMessage

logger = logging.getLogger(__name__)

AGENT_TYPE = "global_market"


# ---------------------------------------------------------------------------
# Symbol mapping for yfinance
# ---------------------------------------------------------------------------

GLOBAL_INDICES: Dict[str, Dict[str, str]] = {
    # US
    "SP500": {"ticker": "^GSPC", "name": "S&P 500", "region": "us"},
    "NASDAQ": {"ticker": "^IXIC", "name": "Nasdaq Composite", "region": "us"},
    "DOW": {"ticker": "^DJI", "name": "Dow Jones", "region": "us"},
    "RUSSELL": {"ticker": "^RUT", "name": "Russell 2000", "region": "us"},
    "VIX": {"ticker": "^VIX", "name": "CBOE VIX", "region": "us"},
    # European
    "FTSE": {"ticker": "^FTSE", "name": "FTSE 100", "region": "europe"},
    "DAX": {"ticker": "^GDAXI", "name": "DAX", "region": "europe"},
    "CAC40": {"ticker": "^FCHI", "name": "CAC 40", "region": "europe"},
    # Asian
    "NIKKEI": {"ticker": "^N225", "name": "Nikkei 225", "region": "asia"},
    "HANGSENG": {"ticker": "^HSI", "name": "Hang Seng", "region": "asia"},
    "SHANGHAI": {"ticker": "000001.SS", "name": "Shanghai Composite", "region": "asia"},
    "SGXNIFTY": {"ticker": "^NSEBANK", "name": "SGX Nifty (proxy)", "region": "asia"},
    # Indian
    "NIFTY50": {"ticker": "^NSEI", "name": "Nifty 50", "region": "india"},
    "SENSEX": {"ticker": "^BSESN", "name": "Sensex", "region": "india"},
    "INDIAVIX": {"ticker": "^NSEBANK", "name": "India VIX (proxy)", "region": "india"},
}

COMMODITIES: Dict[str, Dict[str, str]] = {
    "GOLD": {"ticker": "GC=F", "name": "Gold Futures", "unit": "USD/oz"},
    "SILVER": {"ticker": "SI=F", "name": "Silver Futures", "unit": "USD/oz"},
    "CRUDEWTI": {"ticker": "CL=F", "name": "Crude Oil WTI", "unit": "USD/bbl"},
    "BRENT": {"ticker": "BZ=F", "name": "Brent Crude", "unit": "USD/bbl"},
    "NATGAS": {"ticker": "NG=F", "name": "Natural Gas", "unit": "USD/mmBtu"},
    "COPPER": {"ticker": "HG=F", "name": "Copper", "unit": "USD/lb"},
}

FOREX: Dict[str, Dict[str, str]] = {
    "DXY": {"ticker": "DX-Y.NYB", "name": "US Dollar Index", "pair": "DXY"},
    "USDINR": {"ticker": "USDINR=X", "name": "USD/INR", "pair": "USD/INR"},
    "EURUSD": {"ticker": "EURUSD=X", "name": "EUR/USD", "pair": "EUR/USD"},
    "GBPUSD": {"ticker": "GBPUSD=X", "name": "GBP/USD", "pair": "GBP/USD"},
    "JPYUSD": {"ticker": "JPYUSD=X", "name": "JPY/USD", "pair": "JPY/USD"},
}

BONDS: Dict[str, Dict[str, str]] = {
    "US10Y": {"ticker": "^TNX", "name": "US 10Y Treasury Yield", "unit": "%"},
    "US2Y": {"ticker": "^IRX", "name": "US 2Y Treasury Yield", "unit": "%"},
    "US30Y": {"ticker": "^TYX", "name": "US 30Y Treasury Yield", "unit": "%"},
}

CRYPTO: Dict[str, Dict[str, str]] = {
    "BTC": {"ticker": "BTC-USD", "name": "Bitcoin", "unit": "USD"},
    "ETH": {"ticker": "ETH-USD", "name": "Ethereum", "unit": "USD"},
}

# Search query templates for global market news
NEWS_QUERIES = {
    "us": "US stock market today S&P 500 Federal Reserve",
    "asia": "Asian markets today Nikkei Hang Seng China stocks",
    "commodities": "Gold crude oil price today commodity markets",
    "forex": "USD INR dollar rupee exchange rate today",
    "india_fii": "FII DII India equity flows today Nifty",
    "macro": "Global macro outlook Fed RBI rate cut interest rates 2025",
    "geopolitical": "geopolitical risk global markets today",
}


# ---------------------------------------------------------------------------
# Helper: quick yfinance price fetch
# ---------------------------------------------------------------------------


async def _fetch_yf_prices(
    tickers: List[str],
    max_concurrent: int = 6,
) -> Dict[str, Optional[float]]:
    """
    Fetch latest close prices for a list of yfinance tickers concurrently.
    Returns dict: ticker → latest_close (or None on failure).
    """
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)
    results: Dict[str, Optional[float]] = {}

    async def _fetch_one(ticker: str) -> None:
        async with semaphore:
            try:
                import yfinance as yf

                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None,
                    lambda: yf.Ticker(ticker).history(period="5d", interval="1d"),
                )
                if data is not None and not data.empty:
                    close = data["Close"].dropna()
                    if not close.empty:
                        results[ticker] = float(close.iloc[-1])
                        return
                results[ticker] = None
            except Exception as exc:
                logger.debug(f"yf fetch failed for {ticker}: {exc}")
                results[ticker] = None

    await asyncio.gather(*[_fetch_one(t) for t in tickers], return_exceptions=True)
    return results


def _pct_change(current: Optional[float], prev: Optional[float]) -> Optional[float]:
    """Safe percentage change calculation."""
    if current is None or prev is None or prev == 0:
        return None
    return round((current - prev) / abs(prev) * 100, 2)


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------


class GlobalMarketAgent(BaseSwarmAgent):
    """
    Autonomous Global Market Monitor.

    On each run:
      1. Fetches prices for all tracked global instruments via yfinance
      2. Searches web for latest market news + macro developments
      3. Computes India impact score from global cues
      4. Classifies global risk regime (risk-on / risk-off / mixed)
      5. Predicts FII flow direction based on global signals
      6. Generates a comprehensive global market report
    """

    AGENT_TYPE = AGENT_TYPE
    DEFAULT_TIMEOUT_S = 120.0

    async def execute(self, message: SwarmMessage) -> AgentResult:
        payload = message.payload
        focus: str = str(payload.get("focus", "all")).lower()
        include_macro: bool = bool(payload.get("include_macro", True))
        depth: str = str(payload.get("depth", "full")).lower()

        self._log.info(f"🌍 GlobalMarketAgent starting | focus={focus} depth={depth}")

        # ── 1. Fetch prices concurrently ─────────────────────────────────
        all_tickers = self._get_tickers_for_focus(focus)
        self._log.info(f"Fetching {len(all_tickers)} global tickers…")

        prices = await _fetch_yf_prices(all_tickers)
        self._log.info(
            f"Price fetch done: {sum(1 for v in prices.values() if v)} / {len(all_tickers)} succeeded"
        )

        # ── 2. Build global snapshot ──────────────────────────────────────
        snapshot = self._build_snapshot(prices)

        # ── 3. Fetch market news (parallel) ───────────────────────────────
        news_data: Dict[str, str] = {}
        if include_macro or depth == "full":
            news_queries = self._get_news_queries(focus)
            news_results = await asyncio.gather(
                *[self.tools.web_search(q) for q in news_queries],
                return_exceptions=True,
            )
            for i, result in enumerate(news_results):
                if not isinstance(result, Exception) and result:
                    key = (
                        list(news_queries.keys())[i]
                        if i < len(news_queries)
                        else f"news_{i}"
                    )
                    news_data[key] = str(result)[:1000]

        # ── 4. Compute India impact ───────────────────────────────────────
        india_impact = self._compute_india_impact(snapshot, news_data)

        # ── 5. Classify global risk regime ───────────────────────────────
        risk_regime = self._classify_risk_regime(snapshot)

        # ── 6. Predict FII flow direction ────────────────────────────────
        fii_prediction = self._predict_fii_flows(snapshot, india_impact, risk_regime)

        # ── 7. Identify key alerts ────────────────────────────────────────
        alerts = self._identify_alerts(snapshot, india_impact)

        # ── 8. Compute key correlations ───────────────────────────────────
        correlations = self._build_correlation_notes(snapshot)

        # ── 9. LLM macro synthesis ────────────────────────────────────────
        macro_summary = ""
        report_md = ""
        if include_macro and news_data:
            macro_summary, report_md = await self._llm_synthesis(
                snapshot, india_impact, risk_regime, fii_prediction, news_data, alerts
            )

        # If LLM synthesis failed, build rule-based report
        if not report_md:
            report_md = self._build_report_md(
                snapshot, india_impact, risk_regime, fii_prediction, alerts
            )

        # ── 10. Overall signal ────────────────────────────────────────────
        overall_signal = india_impact.get("signal", "neutral")
        confidence = india_impact.get("confidence", 0.5)

        return self._ok(
            data={
                "global_snapshot": snapshot,
                "india_impact": india_impact,
                "risk_regime": risk_regime,
                "fii_flow_prediction": fii_prediction,
                "key_alerts": alerts,
                "correlations": correlations,
                "macro_summary": macro_summary,
                "report_md": report_md,
                "fetch_timestamp": datetime.utcnow().isoformat(),
                "tickers_fetched": len([v for v in prices.values() if v]),
            },
            summary=report_md[:600]
            if report_md
            else f"Global markets: {risk_regime.get('regime', 'unknown')} regime | India impact: {india_impact.get('score', 0):+.1f}",
            signal=overall_signal,
            confidence=confidence,
            metadata={
                "agent": AGENT_TYPE,
                "focus": focus,
                "tickers": len(all_tickers),
                "news_sources": len(news_data),
            },
        )

    # ────────────────────────────────────────────────────────────────────────
    # Ticker selection
    # ────────────────────────────────────────────────────────────────────────

    def _get_tickers_for_focus(self, focus: str) -> List[str]:
        """Return the relevant ticker list based on focus."""
        if focus == "us":
            groups = [GLOBAL_INDICES]
            filter_regions = {"us"}
        elif focus == "asia":
            groups = [GLOBAL_INDICES]
            filter_regions = {"asia", "india"}
        elif focus == "commodities":
            groups = [COMMODITIES, FOREX]
            filter_regions = None
        elif focus == "india_impact":
            groups = [GLOBAL_INDICES, COMMODITIES, FOREX, BONDS]
            filter_regions = None
        else:  # "all"
            groups = [GLOBAL_INDICES, COMMODITIES, FOREX, BONDS, CRYPTO]
            filter_regions = None

        tickers: List[str] = []
        for group in groups:
            for key, meta in group.items():
                if filter_regions is not None:
                    region = meta.get("region", "")
                    if region not in filter_regions:
                        continue
                tickers.append(meta["ticker"])

        return list(set(tickers))  # deduplicate

    def _get_news_queries(self, focus: str) -> Dict[str, str]:
        """Return relevant news queries for the given focus."""
        if focus == "us":
            return {k: v for k, v in NEWS_QUERIES.items() if k in ("us", "macro")}
        elif focus == "asia":
            return {k: v for k, v in NEWS_QUERIES.items() if k in ("asia", "india_fii")}
        elif focus == "commodities":
            return {
                k: v for k, v in NEWS_QUERIES.items() if k in ("commodities", "forex")
            }
        else:
            return NEWS_QUERIES  # all

    # ────────────────────────────────────────────────────────────────────────
    # Snapshot builder
    # ────────────────────────────────────────────────────────────────────────

    def _build_snapshot(self, prices: Dict[str, Optional[float]]) -> Dict[str, Any]:
        """
        Organise raw prices into a structured global snapshot.
        Each instrument: {name, ticker, price, change_pct, region/unit}
        """
        snapshot: Dict[str, Any] = {
            "indices": {},
            "commodities": {},
            "forex": {},
            "bonds": {},
            "crypto": {},
        }

        # Indices
        for key, meta in GLOBAL_INDICES.items():
            ticker = meta["ticker"]
            price = prices.get(ticker)
            snapshot["indices"][key] = {
                "name": meta["name"],
                "ticker": ticker,
                "price": round(price, 2) if price else None,
                "region": meta["region"],
            }

        # Commodities
        for key, meta in COMMODITIES.items():
            ticker = meta["ticker"]
            price = prices.get(ticker)
            snapshot["commodities"][key] = {
                "name": meta["name"],
                "ticker": ticker,
                "price": round(price, 2) if price else None,
                "unit": meta["unit"],
            }

        # Forex
        for key, meta in FOREX.items():
            ticker = meta["ticker"]
            price = prices.get(ticker)
            snapshot["forex"][key] = {
                "name": meta["name"],
                "ticker": ticker,
                "price": round(price, 4) if price else None,
                "pair": meta["pair"],
            }

        # Bonds
        for key, meta in BONDS.items():
            ticker = meta["ticker"]
            price = prices.get(ticker)
            snapshot["bonds"][key] = {
                "name": meta["name"],
                "ticker": ticker,
                "yield_pct": round(price, 3) if price else None,
            }

        # Crypto
        for key, meta in CRYPTO.items():
            ticker = meta["ticker"]
            price = prices.get(ticker)
            snapshot["crypto"][key] = {
                "name": meta["name"],
                "ticker": ticker,
                "price": round(price, 2) if price else None,
                "unit": meta["unit"],
            }

        return snapshot

    # ────────────────────────────────────────────────────────────────────────
    # India impact score
    # ────────────────────────────────────────────────────────────────────────

    def _compute_india_impact(
        self,
        snapshot: Dict[str, Any],
        news_data: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Compute an India market impact score from global cues.

        Score: -100 (extreme negative for India) to +100 (extreme positive)
        Weights:
          US markets     35%  — strongest correlator
          Asian markets  20%  — overnight + same-day cues
          DXY            15%  — inverse correlation with FII flows
          Crude oil      15%  — negative for India (oil importer)
          US bonds       10%  — higher yields → FII outflows
          Risk regime     5%  — VIX-based adjustment

        Returns dict with: score, signal, confidence, factors, interpretation
        """
        factors: Dict[str, Dict[str, Any]] = {}
        score = 0.0
        total_weight = 0.0

        indices = snapshot.get("indices", {})
        commodities = snapshot.get("commodities", {})
        forex = snapshot.get("forex", {})
        bonds = snapshot.get("bonds", {})

        # ── US markets (35% weight) ────────────────────────────────────────
        sp500_price = indices.get("SP500", {}).get("price")
        nasdaq_price = indices.get("NASDAQ", {}).get("price")

        # We don't have yesterday's prices here, so use VIX and level as proxy
        vix = indices.get("VIX", {}).get("price")
        us_signal = 0.0
        us_interp = ""
        if vix is not None:
            if vix < 15:
                us_signal = 1.0
                us_interp = f"VIX {vix:.1f} — Low fear, risk-on for US markets."
            elif vix < 20:
                us_signal = 0.5
                us_interp = f"VIX {vix:.1f} — Moderate, cautiously positive."
            elif vix < 25:
                us_signal = -0.3
                us_interp = f"VIX {vix:.1f} — Elevated anxiety, mild caution."
            elif vix < 35:
                us_signal = -0.7
                us_interp = f"VIX {vix:.1f} — High fear, risk-off signal."
            else:
                us_signal = -1.0
                us_interp = f"VIX {vix:.1f} — Extreme fear. Negative for India."

        weight = 0.35
        score += us_signal * weight * 100
        total_weight += weight
        factors["us_markets"] = {
            "signal": us_signal,
            "weight": weight,
            "contribution": round(us_signal * weight * 100, 2),
            "interpretation": us_interp,
            "vix": vix,
            "sp500": sp500_price,
        }

        # ── Asian markets (20% weight) ─────────────────────────────────────
        nikkei = indices.get("NIKKEI", {}).get("price")
        hangseng = indices.get("HANGSENG", {}).get("price")
        asia_signal = 0.0
        asia_interp = "Asian market data unavailable."

        # Use Nikkei as proxy (generally risk-on indicator)
        if nikkei is not None:
            # Historical reference: Nikkei > 35000 = bullish
            if nikkei > 38000:
                asia_signal = 0.8
                asia_interp = f"Nikkei ₹{nikkei:,.0f} — Strong. Positive for Asia-India sentiment."
            elif nikkei > 33000:
                asia_signal = 0.4
                asia_interp = f"Nikkei ₹{nikkei:,.0f} — Solid. Mild positive."
            elif nikkei > 28000:
                asia_signal = 0.0
                asia_interp = f"Nikkei ₹{nikkei:,.0f} — Neutral zone."
            else:
                asia_signal = -0.5
                asia_interp = f"Nikkei ₹{nikkei:,.0f} — Weak. Negative regional cue."

        weight = 0.20
        score += asia_signal * weight * 100
        total_weight += weight
        factors["asian_markets"] = {
            "signal": asia_signal,
            "weight": weight,
            "contribution": round(asia_signal * weight * 100, 2),
            "interpretation": asia_interp,
            "nikkei": nikkei,
            "hangseng": hangseng,
        }

        # ── DXY (15% weight) — inverse correlation ─────────────────────────
        dxy = forex.get("DXY", {}).get("price")
        dxy_signal = 0.0
        dxy_interp = "DXY data unavailable."
        if dxy is not None:
            # DXY > 105 = strong dollar = FII outflows from EM including India
            if dxy > 107:
                dxy_signal = -1.0
                dxy_interp = f"DXY {dxy:.2f} — Very strong dollar. Heavy FII outflow risk for India."
            elif dxy > 104:
                dxy_signal = -0.5
                dxy_interp = f"DXY {dxy:.2f} — Strong dollar. Moderate FII headwind."
            elif dxy > 101:
                dxy_signal = -0.2
                dxy_interp = (
                    f"DXY {dxy:.2f} — Slightly elevated dollar. Mild FII headwind."
                )
            elif dxy > 98:
                dxy_signal = 0.3
                dxy_interp = f"DXY {dxy:.2f} — Neutral dollar. Benign for FII flows."
            else:
                dxy_signal = 0.8
                dxy_interp = (
                    f"DXY {dxy:.2f} — Weak dollar. Positive for FII inflows into India."
                )

        weight = 0.15
        score += dxy_signal * weight * 100
        total_weight += weight
        factors["dxy"] = {
            "signal": dxy_signal,
            "weight": weight,
            "contribution": round(dxy_signal * weight * 100, 2),
            "interpretation": dxy_interp,
            "dxy": dxy,
        }

        # ── Crude oil (15% weight) — negative for India ────────────────────
        crude = commodities.get("CRUDEWTI", {}).get("price")
        brent = commodities.get("BRENT", {}).get("price")
        crude_signal = 0.0
        crude_interp = "Crude oil data unavailable."
        crude_ref = brent or crude  # prefer Brent
        if crude_ref is not None:
            # India imports ~85% of crude. Higher oil = higher CAD = negative
            if crude_ref > 95:
                crude_signal = -1.0
                crude_interp = f"Brent ₹{crude_ref:.1f}/bbl — Very high crude. CAD pressure, negative for India."
            elif crude_ref > 85:
                crude_signal = -0.5
                crude_interp = (
                    f"Brent ₹{crude_ref:.1f}/bbl — Elevated crude. Modest CAD concern."
                )
            elif crude_ref > 70:
                crude_signal = 0.0
                crude_interp = (
                    f"Brent ₹{crude_ref:.1f}/bbl — Moderate crude. Neutral for India."
                )
            elif crude_ref > 55:
                crude_signal = 0.5
                crude_interp = f"Brent ₹{crude_ref:.1f}/bbl — Low crude. Positive for India's import bill."
            else:
                crude_signal = 0.8
                crude_interp = f"Brent ₹{crude_ref:.1f}/bbl — Very low crude. Strongly positive for India."

        weight = 0.15
        score += crude_signal * weight * 100
        total_weight += weight
        factors["crude_oil"] = {
            "signal": crude_signal,
            "weight": weight,
            "contribution": round(crude_signal * weight * 100, 2),
            "interpretation": crude_interp,
            "wti": crude,
            "brent": brent,
        }

        # ── US Bond Yields (10% weight) ─────────────────────────────────────
        us10y = bonds.get("US10Y", {}).get("yield_pct")
        us2y = bonds.get("US2Y", {}).get("yield_pct")
        bond_signal = 0.0
        bond_interp = "Bond yield data unavailable."
        if us10y is not None:
            # High US yields attract capital away from EM (India)
            if us10y > 4.8:
                bond_signal = -0.8
                bond_interp = f"US 10Y {us10y:.2f}% — Very high yield. Strong FII outflow pressure."
            elif us10y > 4.2:
                bond_signal = -0.4
                bond_interp = (
                    f"US 10Y {us10y:.2f}% — Elevated yield. Moderate FII headwind."
                )
            elif us10y > 3.5:
                bond_signal = -0.1
                bond_interp = f"US 10Y {us10y:.2f}% — Moderate yield. Mild headwind."
            elif us10y > 2.5:
                bond_signal = 0.3
                bond_interp = (
                    f"US 10Y {us10y:.2f}% — Low yield. Supports FII flows into India."
                )
            else:
                bond_signal = 0.6
                bond_interp = f"US 10Y {us10y:.2f}% — Very low yield. Strong FII inflow potential."

            # Yield curve (2Y vs 10Y) — inverted = recession risk
            if us2y is not None and us2y > us10y + 0.5:
                bond_interp += " | Yield curve inverted — US recession risk elevated."
                bond_signal -= 0.2

        weight = 0.10
        score += bond_signal * weight * 100
        total_weight += weight
        factors["us_bonds"] = {
            "signal": bond_signal,
            "weight": weight,
            "contribution": round(bond_signal * weight * 100, 2),
            "interpretation": bond_interp,
            "us_10y": us10y,
            "us_2y": us2y,
            "yield_curve_inverted": (us2y and us10y and us2y > us10y),
        }

        # ── Gold (5% weight) — flight-to-safety indicator ─────────────────
        gold = commodities.get("GOLD", {}).get("price")
        gold_signal = 0.0
        gold_interp = "Gold data unavailable."
        if gold is not None:
            # Gold rallying = risk-off = neutral-to-negative for equity
            if gold > 3000:
                gold_signal = -0.3
                gold_interp = f"Gold ${gold:,.0f}/oz — Very high. Risk-off signal."
            elif gold > 2500:
                gold_signal = -0.1
                gold_interp = (
                    f"Gold ${gold:,.0f}/oz — Elevated, mild safe-haven demand."
                )
            elif gold > 1900:
                gold_signal = 0.1
                gold_interp = f"Gold ${gold:,.0f}/oz — Normal range, neutral."
            else:
                gold_signal = 0.2
                gold_interp = f"Gold ${gold:,.0f}/oz — Low, risk-on preference."

        weight = 0.05
        score += gold_signal * weight * 100
        total_weight += weight
        factors["gold"] = {
            "signal": gold_signal,
            "weight": weight,
            "contribution": round(gold_signal * weight * 100, 2),
            "interpretation": gold_interp,
            "price_usd": gold,
        }

        # ── Final score ───────────────────────────────────────────────────
        final_score = round(score, 2)

        if final_score >= 15:
            signal = "bullish"
            interpretation = f"Global cues POSITIVE for India ({final_score:+.1f}/100). Expect gap-up / FII buying."
        elif final_score >= 5:
            signal = "bullish"
            interpretation = f"Mild positive global cues ({final_score:+.1f}/100). Cautiously bullish open expected."
        elif final_score >= -5:
            signal = "neutral"
            interpretation = (
                f"Mixed global cues ({final_score:+.1f}/100). Flat to rangebound open."
            )
        elif final_score >= -15:
            signal = "bearish"
            interpretation = f"Mild negative global cues ({final_score:+.1f}/100). Cautiously bearish open expected."
        else:
            signal = "bearish"
            interpretation = f"Global cues NEGATIVE for India ({final_score:+.1f}/100). Expect gap-down / FII selling."

        return {
            "score": round(final_score, 2),
            "signal": signal,
            "confidence": round(min(1.0, abs(final_score) / 30.0), 3),
            "interpretation": interpretation,
            "factors": factors,
        }

    # ────────────────────────────────────────────────────────────────────────
    # Risk regime classifier
    # ────────────────────────────────────────────────────────────────────────

    def _classify_risk_regime(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify the global risk regime as:
          RISK_ON    — equities up, VIX low, USD weak, gold flat
          RISK_OFF   — equities down, VIX high, USD strong, gold up
          MIXED      — conflicting signals
          UNCERTAIN  — insufficient data
        """
        indices = snapshot.get("indices", {})
        commodities = snapshot.get("commodities", {})
        forex = snapshot.get("forex", {})

        vix = indices.get("VIX", {}).get("price")
        gold = commodities.get("GOLD", {}).get("price")
        dxy = forex.get("DXY", {}).get("price")
        sp500 = indices.get("SP500", {}).get("price")

        risk_on_signals = 0
        risk_off_signals = 0
        total_signals = 0

        if vix is not None:
            total_signals += 1
            if vix < 18:
                risk_on_signals += 1
            elif vix > 25:
                risk_off_signals += 1

        if dxy is not None:
            total_signals += 1
            if dxy < 101:
                risk_on_signals += 1
            elif dxy > 105:
                risk_off_signals += 1

        if gold is not None:
            total_signals += 1
            if gold < 2000:
                risk_on_signals += 1
            elif gold > 2500:
                risk_off_signals += 1

        if total_signals == 0:
            return {
                "regime": "UNCERTAIN",
                "confidence": 0.2,
                "interpretation": "Insufficient data.",
            }

        if risk_on_signals > risk_off_signals:
            regime = "RISK_ON"
            conf = risk_on_signals / total_signals
            interp = f"Risk-on environment ({risk_on_signals}/{total_signals} signals). Positive for Indian equities / FII inflows."
        elif risk_off_signals > risk_on_signals:
            regime = "RISK_OFF"
            conf = risk_off_signals / total_signals
            interp = f"Risk-off environment ({risk_off_signals}/{total_signals} signals). Caution — FII outflows likely."
        else:
            regime = "MIXED"
            conf = 0.4
            interp = "Mixed signals — no clear risk-on or risk-off stance."

        return {
            "regime": regime,
            "confidence": round(conf, 3),
            "risk_on_signals": risk_on_signals,
            "risk_off_signals": risk_off_signals,
            "total_signals": total_signals,
            "interpretation": interp,
        }

    # ────────────────────────────────────────────────────────────────────────
    # FII flow prediction
    # ────────────────────────────────────────────────────────────────────────

    def _predict_fii_flows(
        self,
        snapshot: Dict[str, Any],
        india_impact: Dict[str, Any],
        risk_regime: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Predict FII flow direction based on global cues.
        Returns: direction ('inflow'|'outflow'|'neutral'), magnitude ('high'|'medium'|'low')
        """
        score = india_impact.get("score", 0.0)
        regime = risk_regime.get("regime", "MIXED")

        if score >= 15 and regime == "RISK_ON":
            direction = "inflow"
            magnitude = "high"
            interp = "Strong FII inflows likely. Risk-on + positive global cues favour EM allocation."
        elif score >= 5:
            direction = "inflow"
            magnitude = "medium"
            interp = (
                "Mild FII inflows expected. Positive but not overwhelming global cues."
            )
        elif score <= -15 and regime == "RISK_OFF":
            direction = "outflow"
            magnitude = "high"
            interp = "Strong FII outflows likely. Risk-off + negative global cues → EM selling."
        elif score <= -5:
            direction = "outflow"
            magnitude = "medium"
            interp = (
                "Mild FII outflows expected. Negative global cues weighing on India."
            )
        else:
            direction = "neutral"
            magnitude = "low"
            interp = "FII flows likely neutral. Mixed global cues, no strong directional bias."

        return {
            "direction": direction,
            "magnitude": magnitude,
            "interpretation": interp,
            "india_impact_score": score,
            "risk_regime": regime,
        }

    # ────────────────────────────────────────────────────────────────────────
    # Key alerts
    # ────────────────────────────────────────────────────────────────────────

    def _identify_alerts(
        self,
        snapshot: Dict[str, Any],
        india_impact: Dict[str, Any],
    ) -> List[str]:
        """Identify key market alerts from global data."""
        alerts: List[str] = []
        indices = snapshot.get("indices", {})
        commodities = snapshot.get("commodities", {})
        bonds = snapshot.get("bonds", {})
        forex = snapshot.get("forex", {})

        vix = indices.get("VIX", {}).get("price")
        if vix and vix > 30:
            alerts.append(
                f"⚠️ VIX at {vix:.1f} — Extreme fear in US markets. High volatility regime."
            )
        elif vix and vix > 20:
            alerts.append(f"VIX elevated at {vix:.1f} — Caution warranted.")

        us10y = bonds.get("US10Y", {}).get("yield_pct")
        if us10y and us10y > 4.8:
            alerts.append(
                f"US 10Y yield at {us10y:.2f}% — High bond yields pressuring EM equities."
            )

        crude = commodities.get("BRENT", {}).get("price") or commodities.get(
            "CRUDEWTI", {}
        ).get("price")
        if crude and crude > 95:
            alerts.append(
                f"Crude oil at ${crude:.1f}/bbl — High oil prices negative for India's current account."
            )

        usdinr = forex.get("USDINR", {}).get("price")
        if usdinr and usdinr > 87:
            alerts.append(
                f"USD/INR at {usdinr:.2f} — Weak rupee increases import costs and inflation pressure."
            )

        score = india_impact.get("score", 0)
        if score <= -20:
            alerts.append(
                f"Global cues score {score:+.1f}/100 — Strongly negative for Indian markets today."
            )

        return alerts

    # ────────────────────────────────────────────────────────────────────────
    # Correlation notes
    # ────────────────────────────────────────────────────────────────────────

    def _build_correlation_notes(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Build key correlation observations."""
        notes = []
        indices = snapshot.get("indices", {})
        forex = snapshot.get("forex", {})
        commodities = snapshot.get("commodities", {})

        sp500 = indices.get("SP500", {}).get("price")
        nifty = indices.get("NIFTY50", {}).get("price")
        dxy = forex.get("DXY", {}).get("price")
        gold = commodities.get("GOLD", {}).get("price")
        crude = commodities.get("CRUDEWTI", {}).get("price")

        notes.append(
            {
                "pair": "Nifty–S&P500",
                "note": "Historically 0.65+ correlation. S&P500 direction often sets Nifty's gap-up/down.",
                "sp500": sp500,
                "nifty": nifty,
            }
        )
        notes.append(
            {
                "pair": "DXY–FII",
                "note": "Strong dollar (DXY>105) = FII outflows from India. Watch closely.",
                "dxy": dxy,
            }
        )
        notes.append(
            {
                "pair": "Crude–India",
                "note": "India imports ~85% crude. $10 rise in Brent ≈ 40bps GDP impact.",
                "crude_wti": crude,
            }
        )

        return {"key_correlations": notes}

    # ────────────────────────────────────────────────────────────────────────
    # LLM synthesis
    # ────────────────────────────────────────────────────────────────────────

    async def _llm_synthesis(
        self,
        snapshot: Dict[str, Any],
        india_impact: Dict[str, Any],
        risk_regime: Dict[str, Any],
        fii_prediction: Dict[str, Any],
        news_data: Dict[str, str],
        alerts: List[str],
    ) -> tuple:
        """Use LLM to synthesise global data into a market narrative. Returns (macro_summary, report_md)."""
        try:
            client = self.tools.get_llm_client()
            model = self.tools.get_model("fast")

            score = india_impact.get("score", 0)
            signal = india_impact.get("signal", "neutral")
            regime = risk_regime.get("regime", "MIXED")
            fii_dir = fii_prediction.get("direction", "neutral")

            indices = snapshot.get("indices", {})
            commodities = snapshot.get("commodities", {})
            forex = snapshot.get("forex", {})
            bonds = snapshot.get("bonds", {})

            data_summary = []
            for key in ["SP500", "NASDAQ", "NIKKEI", "HANGSENG"]:
                p = indices.get(key, {}).get("price")
                if p:
                    data_summary.append(f"{key}: {p:,.2f}")
            for key in ["CRUDEWTI", "BRENT", "GOLD"]:
                p = commodities.get(key, {}).get("price")
                if p:
                    data_summary.append(f"{key}: ${p:,.2f}")
            for key in ["DXY", "USDINR"]:
                p = forex.get(key, {}).get("price")
                if p:
                    data_summary.append(f"{key}: {p:.4f}")
            us10y = bonds.get("US10Y", {}).get("yield_pct")
            if us10y:
                data_summary.append(f"US10Y: {us10y:.3f}%")

            news_summary = "\n".join(
                f"[{k}]: {v[:300]}" for k, v in list(news_data.items())[:4]
            )

            alerts_text = (
                "\n".join(f"- {a}" for a in alerts) if alerts else "No major alerts."
            )

            prompt = f"""You are a global macro strategist at India's top brokerage.

Global Market Snapshot:
{chr(10).join(data_summary)}

News:
{news_summary}

Alerts:
{alerts_text}

Analysis:
- India Impact Score: {score:+.1f}/100
- Global Signal: {signal.upper()}
- Risk Regime: {regime}
- FII Flow Prediction: {fii_dir.upper()}

Write a concise global markets report (3-4 paragraphs) explaining:
1. What global markets did today and why
2. How this affects Indian markets specifically
3. Key levels to watch (Nifty range, USD/INR, crude)
4. FII flow expectation and sectors to watch

Use ₹ for Indian prices. Be specific with numbers. Format as markdown.
End with: ⚠️ *Not financial advice.*"""

            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
            )
            report_md = (response.choices[0].message.content or "").strip()
            macro_summary = report_md[:400]
            return macro_summary, report_md

        except Exception as exc:
            self._log.warning(f"Global market LLM synthesis error: {exc}")
            return "", ""

    # ────────────────────────────────────────────────────────────────────────
    # Rule-based report builder (fallback)
    # ────────────────────────────────────────────────────────────────────────

    def _build_report_md(
        self,
        snapshot: Dict[str, Any],
        india_impact: Dict[str, Any],
        risk_regime: Dict[str, Any],
        fii_prediction: Dict[str, Any],
        alerts: List[str],
    ) -> str:
        score = india_impact.get("score", 0)
        signal = india_impact.get("signal", "neutral")
        regime = risk_regime.get("regime", "MIXED")
        emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(signal, "⚪")

        lines = [
            f"## {emoji} Global Markets Report",
            f"**India Impact Score:** {score:+.1f}/100 | **Risk Regime:** {regime}",
            "",
        ]

        if alerts:
            lines.append("**Key Alerts:**")
            for alert in alerts:
                lines.append(f"- {alert}")
            lines.append("")

        factors = india_impact.get("factors", {})
        lines.append("**Factor Breakdown:**")
        for key, factor in factors.items():
            interp = factor.get("interpretation", "")
            if interp:
                lines.append(f"- {interp}")
        lines.append("")

        fii_dir = fii_prediction.get("direction", "neutral")
        fii_interp = fii_prediction.get("interpretation", "")
        lines.append(f"**FII Flow Prediction:** {fii_dir.upper()} — {fii_interp}")
        lines.append("")
        lines.append("⚠️ *Not financial advice.*")

        return "\n".join(lines)
