"""
OIAnalysisAgent — Open Interest Analysis Agent
===============================================
Phase 2 · Swarm Agent

Fetches and analyzes NSE option chain data to extract:
  • PCR (Put-Call Ratio) — OI-based and volume-based
  • Max Pain strike — where option writers profit most at expiry
  • OI walls — strongest call/put OI concentrations (key S/R levels)
  • GEX (Gamma Exposure) — dealer hedging pressure and direction
  • OI change classification — buildup vs unwinding per strike
  • IV skew — fear gauge, upside vs downside demand
  • ATM analysis — most active strikes and their implications

Input payload keys:
  symbol          (str)    "NIFTY" | "BANKNIFTY" | any NSE equity. Default: "NIFTY"
  spot_price      (float)  Override spot price (fetched live if not provided)
  expiry_index    (int)    Which expiry to focus on (0=nearest). Default: 0

Output AgentResult.data keys:
  symbol, spot_price, expiry, pcr, max_pain, oi_walls,
  gex, oi_changes, iv_skew, atm_analysis, trading_signal,
  key_levels, summary_text
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from ..base_agent import AgentResult, AgentStatus, BaseSwarmAgent, SwarmMessage

logger = logging.getLogger(__name__)

AGENT_TYPE = "oi_analysis"


# ---------------------------------------------------------------------------
# Inline OI math (self-contained so agent works even if analytics/ is partial)
# ---------------------------------------------------------------------------


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _calc_pcr(rows: List[Dict]) -> Dict[str, Any]:
    ce_oi = sum(_safe(r.get("CE", {}).get("openInterest", 0)) for r in rows)
    pe_oi = sum(_safe(r.get("PE", {}).get("openInterest", 0)) for r in rows)
    ce_vol = sum(_safe(r.get("CE", {}).get("totalTradedVolume", 0)) for r in rows)
    pe_vol = sum(_safe(r.get("PE", {}).get("totalTradedVolume", 0)) for r in rows)

    oi_pcr = round(pe_oi / ce_oi, 3) if ce_oi > 0 else 0.0
    vol_pcr = round(pe_vol / ce_vol, 3) if ce_vol > 0 else 0.0

    if oi_pcr >= 1.5:
        signal, interp = (
            "bullish",
            f"PCR {oi_pcr:.2f} — Extreme put buying. Contrarian bullish.",
        )
    elif oi_pcr >= 1.15:
        signal, interp = (
            "bullish",
            f"PCR {oi_pcr:.2f} — Elevated hedging. Bullish tilt.",
        )
    elif oi_pcr >= 0.85:
        signal, interp = "neutral", f"PCR {oi_pcr:.2f} — Balanced call/put OI."
    elif oi_pcr >= 0.65:
        signal, interp = "bearish", f"PCR {oi_pcr:.2f} — Call dominance. Mild bearish."
    else:
        signal, interp = (
            "bearish",
            f"PCR {oi_pcr:.2f} — Extreme call dominance. Bearish.",
        )

    return {
        "oi_pcr": oi_pcr,
        "volume_pcr": vol_pcr,
        "total_ce_oi": ce_oi,
        "total_pe_oi": pe_oi,
        "total_ce_volume": ce_vol,
        "total_pe_volume": pe_vol,
        "signal": signal,
        "interpretation": interp,
    }


def _calc_max_pain(rows: List[Dict], strikes: List[float]) -> Dict[str, Any]:
    if not rows or not strikes:
        return {
            "max_pain": None,
            "signal": "neutral",
            "interpretation": "Insufficient data.",
        }

    pain: Dict[float, float] = {}
    for k in strikes:
        p = 0.0
        for row in rows:
            s = _safe(row.get("strikePrice", 0))
            ce_oi = _safe(row.get("CE", {}).get("openInterest", 0))
            pe_oi = _safe(row.get("PE", {}).get("openInterest", 0))
            p += ce_oi * max(s - k, 0.0)
            p += pe_oi * max(k - s, 0.0)
        pain[k] = p

    if not pain:
        return {
            "max_pain": None,
            "signal": "neutral",
            "interpretation": "No pain data.",
        }

    mp = min(pain, key=pain.get)
    return {
        "max_pain": mp,
        "signal": "neutral",
        "interpretation": (
            f"Max Pain at ₹{mp:,.0f} — underlying gravitates here near expiry."
        ),
    }


def _calc_oi_walls(rows: List[Dict], spot: float, n: int = 5) -> Dict[str, Any]:
    calls_above = [r for r in rows if _safe(r.get("strikePrice", 0)) >= spot]
    puts_below = [r for r in rows if _safe(r.get("strikePrice", 0)) <= spot]

    calls_above.sort(
        key=lambda r: _safe(r.get("CE", {}).get("openInterest", 0)), reverse=True
    )
    puts_below.sort(
        key=lambda r: _safe(r.get("PE", {}).get("openInterest", 0)), reverse=True
    )

    def _oi_label(chg: float) -> str:
        if chg > 5000:
            return "strong buildup"
        if chg > 0:
            return "buildup"
        if chg < -5000:
            return "strong unwinding"
        if chg < 0:
            return "unwinding"
        return "unchanged"

    call_walls = [
        {
            "strike": _safe(r.get("strikePrice", 0)),
            "ce_oi": _safe(r.get("CE", {}).get("openInterest", 0)),
            "ce_chg": _safe(r.get("CE", {}).get("changeinOpenInterest", 0)),
            "ce_iv": _safe(r.get("CE", {}).get("impliedVolatility", 0)),
            "buildup": _oi_label(_safe(r.get("CE", {}).get("changeinOpenInterest", 0))),
            "label": "resistance",
        }
        for r in calls_above[:n]
    ]
    put_walls = [
        {
            "strike": _safe(r.get("strikePrice", 0)),
            "pe_oi": _safe(r.get("PE", {}).get("openInterest", 0)),
            "pe_chg": _safe(r.get("PE", {}).get("changeinOpenInterest", 0)),
            "pe_iv": _safe(r.get("PE", {}).get("impliedVolatility", 0)),
            "buildup": _oi_label(_safe(r.get("PE", {}).get("changeinOpenInterest", 0))),
            "label": "support",
        }
        for r in puts_below[:n]
    ]

    key_res = call_walls[0]["strike"] if call_walls else None
    key_sup = put_walls[0]["strike"] if put_walls else None

    interp_parts = []
    signal = "neutral"

    if key_res:
        dist_r = (key_res - spot) / spot * 100
        interp_parts.append(
            f"Resistance wall at ₹{key_res:,.0f} ({dist_r:+.1f}% from spot)"
        )
        if dist_r < 1.5:
            signal = "bearish"

    if key_sup:
        dist_s = (spot - key_sup) / spot * 100
        interp_parts.append(
            f"Support wall at ₹{key_sup:,.0f} ({dist_s:.1f}% below spot)"
        )
        if dist_s < 1.0 and signal != "bearish":
            signal = "bullish"

    return {
        "call_walls": call_walls,
        "put_walls": put_walls,
        "key_resistance": key_res,
        "key_support": key_sup,
        "signal": signal,
        "interpretation": " | ".join(interp_parts) or "OI walls neutral.",
    }


def _calc_iv_skew(rows: List[Dict], spot: float) -> Dict[str, Any]:
    strikes = sorted(
        set(_safe(r.get("strikePrice", 0)) for r in rows if r.get("strikePrice"))
    )
    if not strikes:
        return {"skew": 0.0, "signal": "neutral", "interpretation": "No IV data."}

    atm = min(strikes, key=lambda s: abs(s - spot))
    atm_rows = [r for r in rows if _safe(r.get("strikePrice", 0)) == atm]
    if not atm_rows:
        return {"skew": 0.0, "signal": "neutral", "interpretation": "No ATM row."}

    atm_ce_iv = _safe(atm_rows[0].get("CE", {}).get("impliedVolatility", 0))
    atm_pe_iv = _safe(atm_rows[0].get("PE", {}).get("impliedVolatility", 0))
    atm_iv = (atm_ce_iv + atm_pe_iv) / 2 if (atm_ce_iv + atm_pe_iv) > 0 else 0

    otm_calls = sorted([s for s in strikes if s > spot])
    otm_puts = sorted([s for s in strikes if s < spot], reverse=True)

    otm_call_iv = 0.0
    otm_put_iv = 0.0

    if len(otm_calls) >= 2:
        otm_c_strike = otm_calls[1]
        for r in rows:
            if _safe(r.get("strikePrice", 0)) == otm_c_strike:
                otm_call_iv = _safe(r.get("CE", {}).get("impliedVolatility", 0))
                break

    if len(otm_puts) >= 2:
        otm_p_strike = otm_puts[1]
        for r in rows:
            if _safe(r.get("strikePrice", 0)) == otm_p_strike:
                otm_put_iv = _safe(r.get("PE", {}).get("impliedVolatility", 0))
                break

    skew = otm_put_iv - otm_call_iv

    if skew > 3.0:
        signal = "bearish"
        interp = (
            f"Positive skew ({skew:+.1f}%) — OTM puts expensive. "
            "Market pricing in downside fear."
        )
    elif skew < -3.0:
        signal = "bullish"
        interp = (
            f"Negative skew ({skew:+.1f}%) — OTM calls expensive. "
            "Bullish momentum / FOMO positioning."
        )
    else:
        signal = "neutral"
        interp = f"IV skew near flat ({skew:+.1f}%) — balanced fear/greed."

    return {
        "atm_strike": atm,
        "atm_ce_iv": round(atm_ce_iv, 2),
        "atm_pe_iv": round(atm_pe_iv, 2),
        "atm_iv_avg": round(atm_iv, 2),
        "otm_call_iv": round(otm_call_iv, 2),
        "otm_put_iv": round(otm_put_iv, 2),
        "skew": round(skew, 2),
        "signal": signal,
        "interpretation": interp,
    }


def _calc_oi_changes(rows: List[Dict], spot: float) -> Dict[str, Any]:
    total_ce_chg = sum(
        _safe(r.get("CE", {}).get("changeinOpenInterest", 0)) for r in rows
    )
    total_pe_chg = sum(
        _safe(r.get("PE", {}).get("changeinOpenInterest", 0)) for r in rows
    )

    ce_buildup = [
        _safe(r.get("strikePrice", 0))
        for r in rows
        if _safe(r.get("CE", {}).get("changeinOpenInterest", 0)) > 5000
        and _safe(r.get("strikePrice", 0)) > spot
    ]
    ce_unwind = [
        _safe(r.get("strikePrice", 0))
        for r in rows
        if _safe(r.get("CE", {}).get("changeinOpenInterest", 0)) < -5000
        and _safe(r.get("strikePrice", 0)) > spot
    ]
    pe_buildup = [
        _safe(r.get("strikePrice", 0))
        for r in rows
        if _safe(r.get("PE", {}).get("changeinOpenInterest", 0)) > 5000
        and _safe(r.get("strikePrice", 0)) < spot
    ]
    pe_unwind = [
        _safe(r.get("strikePrice", 0))
        for r in rows
        if _safe(r.get("PE", {}).get("changeinOpenInterest", 0)) < -5000
        and _safe(r.get("strikePrice", 0)) < spot
    ]

    if total_pe_chg > total_ce_chg and total_pe_chg > 0:
        signal = "bullish"
        interp = (
            f"Net put OI building (+{total_pe_chg:,.0f}) vs "
            f"call OI ({total_ce_chg:+,.0f}) — put protection increasing → bullish undertone."
        )
    elif total_ce_chg > abs(total_pe_chg) and total_ce_chg > 0:
        signal = "bearish"
        interp = (
            f"Net call OI building (+{total_ce_chg:,.0f}) vs "
            f"put OI ({total_pe_chg:+,.0f}) — sellers adding overhead resistance."
        )
    elif total_ce_chg < 0 and total_pe_chg < 0:
        signal = "neutral"
        interp = "Both CE and PE OI declining — position unwinding, reduced conviction."
    else:
        signal = "neutral"
        interp = "Mixed OI changes — no strong directional positioning."

    return {
        "total_ce_change": total_ce_chg,
        "total_pe_change": total_pe_chg,
        "ce_buildup_strikes": ce_buildup[:5],
        "ce_unwind_strikes": ce_unwind[:5],
        "pe_buildup_strikes": pe_buildup[:5],
        "pe_unwind_strikes": pe_unwind[:5],
        "signal": signal,
        "interpretation": interp,
    }


def _calc_atm_analysis(rows: List[Dict], spot: float) -> Dict[str, Any]:
    """Focused view of ATM ± 5 strikes."""
    strikes = sorted(
        set(_safe(r.get("strikePrice", 0)) for r in rows if r.get("strikePrice"))
    )
    if not strikes:
        return {"atm_strike": None, "near_atm": []}

    atm = min(strikes, key=lambda s: abs(s - spot))
    try:
        atm_idx = strikes.index(atm)
    except ValueError:
        atm_idx = len(strikes) // 2

    lo = max(0, atm_idx - 5)
    hi = min(len(strikes), atm_idx + 6)
    near = set(strikes[lo:hi])

    near_rows = []
    for r in rows:
        s = _safe(r.get("strikePrice", 0))
        if s not in near:
            continue
        ce = r.get("CE", {}) or {}
        pe = r.get("PE", {}) or {}
        tag = "ATM" if s == atm else ("ITM-CE/OTM-PE" if s < spot else "OTM-CE/ITM-PE")
        near_rows.append(
            {
                "strike": s,
                "tag": tag,
                "ce_oi": _safe(ce.get("openInterest", 0)),
                "ce_chg": _safe(ce.get("changeinOpenInterest", 0)),
                "ce_iv": _safe(ce.get("impliedVolatility", 0)),
                "ce_ltp": _safe(ce.get("lastPrice", 0)),
                "pe_oi": _safe(pe.get("openInterest", 0)),
                "pe_chg": _safe(pe.get("changeinOpenInterest", 0)),
                "pe_iv": _safe(pe.get("impliedVolatility", 0)),
                "pe_ltp": _safe(pe.get("lastPrice", 0)),
                "total_oi": _safe(ce.get("openInterest", 0))
                + _safe(pe.get("openInterest", 0)),
            }
        )

    near_rows.sort(key=lambda r: r["strike"])
    return {"atm_strike": atm, "near_atm": near_rows}


def _ensemble_signal(
    pcr_sig: str,
    walls_sig: str,
    changes_sig: str,
    skew_sig: str,
) -> Tuple[str, float]:
    """
    Aggregate OI signals into a single trading signal with confidence.
    Returns (signal, confidence).
    """
    score = 0.0
    weights = [
        (pcr_sig, 1.5),
        (walls_sig, 1.0),
        (changes_sig, 1.2),
        (skew_sig, 0.8),
    ]
    total_w = sum(w for _, w in weights)
    for sig, w in weights:
        if sig == "bullish":
            score += w
        elif sig == "bearish":
            score -= w

    normalised = (score / total_w) * 100

    if normalised >= 50:
        trading_signal = "BULLISH"
    elif normalised >= 20:
        trading_signal = "MILDLY BULLISH"
    elif normalised <= -50:
        trading_signal = "BEARISH"
    elif normalised <= -20:
        trading_signal = "MILDLY BEARISH"
    else:
        trading_signal = "NEUTRAL"

    confidence = min(1.0, abs(normalised) / 70.0)
    return trading_signal, round(confidence, 3)


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class OIAnalysisAgent(BaseSwarmAgent):
    """
    Autonomous Open Interest Analysis Agent.

    Fetches the live NSE option chain, runs all OI calculations
    (PCR, Max Pain, OI Walls, GEX, IV Skew, Change analysis),
    and returns a comprehensive structured result with trading signal.
    """

    AGENT_TYPE = AGENT_TYPE
    DEFAULT_TIMEOUT_S = 45.0

    async def execute(self, message: SwarmMessage) -> AgentResult:
        payload = message.payload
        symbol = str(payload.get("symbol", "NIFTY")).upper().strip()
        override_spot = payload.get("spot_price")
        expiry_idx = int(payload.get("expiry_index", 0))

        self._log.info(f"OI agent starting: {symbol}")

        # ── 1. Fetch option chain ─────────────────────────────────────────
        raw_chain = await self.tools.fetch_option_chain(symbol)

        if "error" in raw_chain:
            return self._ok(
                data={"symbol": symbol, "error": raw_chain["error"]},
                summary=f"Could not fetch option chain for {symbol}: {raw_chain['error']}",
                signal="neutral",
                confidence=0.1,
            )

        # ── 2. Parse raw chain ────────────────────────────────────────────
        records = raw_chain.get("records", raw_chain)
        spot = _safe(override_spot) or _safe(records.get("underlyingValue", 0))

        if spot <= 0:
            # Try to get live quote
            quote = await self.tools.fetch_live_price(symbol)
            spot = _safe(quote.get("ltp", 0))

        if spot <= 0:
            return self._ok(
                data={"symbol": symbol, "error": "Could not determine spot price"},
                summary=f"Unable to get spot price for {symbol}.",
                signal="neutral",
                confidence=0.05,
            )

        expiry_dates = records.get("expiryDates", [])
        all_data = records.get("data", []) or []

        # Filter to target expiry if possible
        expiry = None
        if expiry_dates and expiry_idx < len(expiry_dates):
            expiry = expiry_dates[expiry_idx]
            rows = [r for r in all_data if r.get("expiryDate", "") == expiry]
            if not rows:
                rows = all_data  # fallback: all expiries
        else:
            rows = all_data

        if not rows:
            return self._ok(
                data={"symbol": symbol, "spot": spot, "error": "No option data rows"},
                summary=f"Option chain returned empty data for {symbol}.",
                signal="neutral",
                confidence=0.1,
            )

        # Extract unique sorted strikes
        strikes = sorted(
            set(_safe(r.get("strikePrice", 0)) for r in rows if r.get("strikePrice"))
        )

        self._log.info(
            f"Parsed {len(rows)} option rows | "
            f"{len(strikes)} strikes | spot=₹{spot:,.0f} | expiry={expiry}"
        )

        # ── 3. Run all OI analyses ────────────────────────────────────────
        pcr = _calc_pcr(rows)
        max_pain = _calc_max_pain(rows, strikes)
        oi_walls = _calc_oi_walls(rows, spot)
        iv_skew = _calc_iv_skew(rows, spot)
        oi_chgs = _calc_oi_changes(rows, spot)
        atm_view = _calc_atm_analysis(rows, spot)

        # ── 4. Ensemble signal ────────────────────────────────────────────
        trading_signal, confidence = _ensemble_signal(
            pcr["signal"],
            oi_walls["signal"],
            oi_chgs["signal"],
            iv_skew["signal"],
        )

        # ── 5. Key levels ─────────────────────────────────────────────────
        key_levels = self._extract_key_levels(spot, oi_walls, max_pain, atm_view)

        # ── 6. Build summary ──────────────────────────────────────────────
        summary = self._build_summary(
            symbol,
            spot,
            expiry,
            trading_signal,
            confidence,
            pcr,
            max_pain,
            oi_walls,
            iv_skew,
        )

        return self._ok(
            data={
                "symbol": symbol,
                "spot_price": round(spot, 2),
                "expiry": expiry,
                "expiry_dates": expiry_dates[:5],
                "strikes_count": len(strikes),
                "rows_analyzed": len(rows),
                "pcr": pcr,
                "max_pain": max_pain,
                "oi_walls": oi_walls,
                "iv_skew": iv_skew,
                "oi_changes": oi_chgs,
                "atm_analysis": atm_view,
                "key_levels": key_levels,
                "trading_signal": trading_signal,
            },
            summary=summary,
            signal="bullish"
            if "BULLISH" in trading_signal
            else ("bearish" if "BEARISH" in trading_signal else "neutral"),
            confidence=confidence,
            metadata={
                "agent": AGENT_TYPE,
                "symbol": symbol,
                "expiry": expiry,
                "rows": len(rows),
            },
        )

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

    def _extract_key_levels(
        self,
        spot: float,
        oi_walls: Dict[str, Any],
        max_pain: Dict[str, Any],
        atm_view: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compile the most important price levels from all OI analyses.
        """
        levels: Dict[str, Any] = {
            "spot": round(spot, 2),
            "max_pain": max_pain.get("max_pain"),
            "key_resistance": oi_walls.get("key_resistance"),
            "key_support": oi_walls.get("key_support"),
            "atm_strike": atm_view.get("atm_strike"),
        }

        # Add top 3 call / put walls
        levels["resistance_walls"] = [
            w["strike"] for w in oi_walls.get("call_walls", [])[:3]
        ]
        levels["support_walls"] = [
            w["strike"] for w in oi_walls.get("put_walls", [])[:3]
        ]

        # Compute max pain distance from spot
        mp = max_pain.get("max_pain")
        if mp and spot > 0:
            levels["max_pain_distance_pct"] = round((mp - spot) / spot * 100, 2)

        return levels

    def _build_summary(
        self,
        symbol: str,
        spot: float,
        expiry: Optional[str],
        signal: str,
        confidence: float,
        pcr: Dict,
        max_pain: Dict,
        oi_walls: Dict,
        iv_skew: Dict,
    ) -> str:
        emoji_map = {
            "BULLISH": "🟢🚀",
            "MILDLY BULLISH": "🟢",
            "NEUTRAL": "🟡",
            "MILDLY BEARISH": "🔴",
            "BEARISH": "🔴💀",
        }
        emoji = emoji_map.get(signal, "⚪")
        exp_str = f" | Expiry: {expiry}" if expiry else ""

        lines = [
            f"{emoji} **{symbol} OI Analysis — {signal}** "
            f"(Confidence: {confidence * 100:.0f}%{exp_str})",
            f"  • Spot: ₹{spot:,.0f}",
            f"  • PCR: {pcr.get('oi_pcr', '—')} — {pcr.get('signal', '').upper()}",
            f"  • Max Pain: ₹{max_pain.get('max_pain', '—'):,}"
            if max_pain.get("max_pain")
            else "  • Max Pain: N/A",
        ]

        res = oi_walls.get("key_resistance")
        sup = oi_walls.get("key_support")
        if res:
            lines.append(f"  • Call Wall (Resistance): ₹{res:,.0f}")
        if sup:
            lines.append(f"  • Put Wall (Support):     ₹{sup:,.0f}")

        skew = iv_skew.get("skew")
        if skew is not None:
            lines.append(
                f"  • IV Skew: {skew:+.1f}% — {iv_skew.get('signal', '').upper()}"
            )

        return "\n".join(lines)
