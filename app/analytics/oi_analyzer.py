"""
OI Analyzer — Open Interest Analysis Engine
============================================
Phase 2 · Feature 2 · Open Interest Intelligence

Analyzes NSE option chain data to extract:
  • PCR      — Put-Call Ratio (volume + OI based)
  • Max Pain — Strike price causing maximum buyer losses at expiry
  • OI walls — Strongest call/put OI concentrations (support/resistance)
  • GEX      — Gamma Exposure (dealer hedging pressure)
  • OI changes — Buildup vs unwinding classification
  • IV surface — Implied volatility skew and term structure

Input:  Raw option chain dict from NSE API (fetch_option_chain tool)
Output: Rich analysis dict with signals, levels, and interpretations

Usage::

    analyzer = OIAnalyzer()
    result = analyzer.analyze(option_chain_data, spot_price=22400)

    # Or convenience wrapper:
    result = analyze_oi(option_chain_data, spot_price=22400)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PCR interpretation thresholds
PCR_EXTREMELY_BEARISH = 0.50  # too many calls → bearish sentiment
PCR_BEARISH = 0.70
PCR_NEUTRAL_LOW = 0.85
PCR_NEUTRAL_HIGH = 1.15
PCR_BULLISH = 1.30
PCR_EXTREMELY_BULLISH = 1.50  # too many puts → contrarian bullish (oversold fear)

# Number of strikes around ATM to show in the focused analysis
ATM_STRIKE_WINDOW = 10  # ± strikes from ATM


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Safely convert any value to float."""
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _find_atm_strike(spot: float, strikes: List[float]) -> float:
    """Return the strike price closest to the spot price."""
    if not strikes:
        return spot
    return min(strikes, key=lambda s: abs(s - spot))


def _interpret_pcr(pcr: float) -> Tuple[str, str]:
    """
    Return (signal, interpretation) for a given PCR value.

    PCR > 1 means more put OI than call OI → options market is hedging → bullish bias.
    PCR < 1 means more call OI → complacency or bullish bets → bearish tilt.
    """
    if pcr >= PCR_EXTREMELY_BULLISH:
        return (
            "bullish",
            f"PCR {pcr:.2f} — Extreme fear/put buying. Contrarian BULLISH signal; often marks short-term bottoms.",
        )
    elif pcr >= PCR_BULLISH:
        return (
            "bullish",
            f"PCR {pcr:.2f} — Elevated put OI vs calls. Bearish hedging prevalent → underlying support likely.",
        )
    elif pcr >= PCR_NEUTRAL_HIGH:
        return "bullish", f"PCR {pcr:.2f} — Slightly put-heavy. Mild bullish tilt."
    elif pcr >= PCR_NEUTRAL_LOW:
        return (
            "neutral",
            f"PCR {pcr:.2f} — Balanced call/put OI. Market in equilibrium.",
        )
    elif pcr >= PCR_BEARISH:
        return "neutral", f"PCR {pcr:.2f} — Slightly call-heavy. Mild caution."
    elif pcr >= PCR_EXTREMELY_BEARISH:
        return (
            "bearish",
            f"PCR {pcr:.2f} — Call OI dominates. Complacency or bullish over-positioning → bearish tilt.",
        )
    else:
        return (
            "bearish",
            f"PCR {pcr:.2f} — Extreme call dominance. Very bearish signal — market top risk.",
        )


# ---------------------------------------------------------------------------
# Core parser: extract structured data from raw NSE option chain response
# ---------------------------------------------------------------------------


class OptionChainParser:
    """
    Parses the raw JSON from NSE's option chain API into structured lists.

    NSE response structure (simplified)::

        {
          "records": {
            "underlyingValue": 22400.5,
            "expiryDates": ["26-Jun-2025", ...],
            "strikePrices": [21000, 21100, ..., 24000],
            "data": [
              {
                "strikePrice": 22000,
                "expiryDate": "26-Jun-2025",
                "CE": {
                  "openInterest": 120000,
                  "changeinOpenInterest": 5000,
                  "totalTradedVolume": 45000,
                  "impliedVolatility": 14.5,
                  "lastPrice": 450.0,
                  "bidQty": 150, "bidprice": 448,
                  "askPrice": 452, "askQty": 100
                },
                "PE": { ... same structure ... }
              }, ...
            ]
          }
        }
    """

    def parse(
        self, raw: Dict[str, Any]
    ) -> Tuple[float, List[float], List[Dict[str, Any]]]:
        """
        Parse raw NSE option chain response.

        Returns:
            (spot_price, sorted_strikes, option_rows)

            Each option_row dict::
                {
                  'strike': float,
                  'expiry': str,
                  'ce_oi': float, 'ce_oi_chg': float, 'ce_vol': float,
                  'ce_iv': float, 'ce_ltp': float,
                  'pe_oi': float, 'pe_oi_chg': float, 'pe_vol': float,
                  'pe_iv': float, 'pe_ltp': float,
                  'total_oi': float, 'oi_ratio': float,  (ce/pe)
                }
        """
        records = raw.get("records", raw)  # handle both wrapped and bare

        spot = _safe_float(records.get("underlyingValue", 0))
        strikes_raw: List[float] = [
            _safe_float(s) for s in records.get("strikePrices", [])
        ]
        data_rows = records.get("data", []) or records.get("filteredData", [])

        rows: List[Dict[str, Any]] = []
        for item in data_rows:
            if not isinstance(item, dict):
                continue
            strike = _safe_float(item.get("strikePrice", 0))
            if strike <= 0:
                continue
            expiry = item.get("expiryDate", "")
            ce = item.get("CE", {}) or {}
            pe = item.get("PE", {}) or {}

            ce_oi = _safe_float(ce.get("openInterest", 0))
            ce_oi_chg = _safe_float(ce.get("changeinOpenInterest", 0))
            ce_vol = _safe_float(ce.get("totalTradedVolume", 0))
            ce_iv = _safe_float(ce.get("impliedVolatility", 0))
            ce_ltp = _safe_float(ce.get("lastPrice", 0))

            pe_oi = _safe_float(pe.get("openInterest", 0))
            pe_oi_chg = _safe_float(pe.get("changeinOpenInterest", 0))
            pe_vol = _safe_float(pe.get("totalTradedVolume", 0))
            pe_iv = _safe_float(pe.get("impliedVolatility", 0))
            pe_ltp = _safe_float(pe.get("lastPrice", 0))

            total_oi = ce_oi + pe_oi
            oi_ratio = ce_oi / pe_oi if pe_oi > 0 else float("inf")

            rows.append(
                {
                    "strike": strike,
                    "expiry": expiry,
                    "ce_oi": ce_oi,
                    "ce_oi_chg": ce_oi_chg,
                    "ce_vol": ce_vol,
                    "ce_iv": ce_iv,
                    "ce_ltp": ce_ltp,
                    "pe_oi": pe_oi,
                    "pe_oi_chg": pe_oi_chg,
                    "pe_vol": pe_vol,
                    "pe_iv": pe_iv,
                    "pe_ltp": pe_ltp,
                    "total_oi": total_oi,
                    "oi_ratio": oi_ratio,
                }
            )

        rows.sort(key=lambda r: r["strike"])
        sorted_strikes = sorted(set(r["strike"] for r in rows))
        return spot, sorted_strikes, rows


# ---------------------------------------------------------------------------
# PCR Calculator
# ---------------------------------------------------------------------------


class PCRCalculator:
    """Computes Put-Call Ratio in three flavors: OI, volume, and combined."""

    def calculate(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_ce_oi = sum(r["ce_oi"] for r in rows)
        total_pe_oi = sum(r["pe_oi"] for r in rows)
        total_ce_vol = sum(r["ce_vol"] for r in rows)
        total_pe_vol = sum(r["pe_vol"] for r in rows)

        oi_pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0.0
        vol_pcr = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0.0
        combined_pcr = (oi_pcr + vol_pcr) / 2 if (oi_pcr + vol_pcr) > 0 else 0.0

        signal, interpretation = _interpret_pcr(oi_pcr)
        _, vol_interp = _interpret_pcr(vol_pcr)

        return {
            "oi_pcr": round(oi_pcr, 3),
            "volume_pcr": round(vol_pcr, 3),
            "combined_pcr": round(combined_pcr, 3),
            "total_ce_oi": total_ce_oi,
            "total_pe_oi": total_pe_oi,
            "total_ce_volume": total_ce_vol,
            "total_pe_volume": total_pe_vol,
            "signal": signal,
            "interpretation": interpretation,
            "volume_interpretation": vol_interp,
        }


# ---------------------------------------------------------------------------
# Max Pain Calculator
# ---------------------------------------------------------------------------


class MaxPainCalculator:
    """
    Calculates the Max Pain strike.

    Definition: The strike price at which the total combined P&L of all
    outstanding option holders (at expiry, assuming underlying closes exactly
    at that strike) is minimised — i.e., option writers keep maximum premium.

    Formula:
        For each candidate strike K:
            pain = Σ_calls  CE_OI(S) × max(S - K, 0)
                 + Σ_puts   PE_OI(S) × max(K - S, 0)
        Max Pain = K that minimises total pain.
    """

    def calculate(
        self, rows: List[Dict[str, Any]], strikes: List[float]
    ) -> Dict[str, Any]:
        if not rows or not strikes:
            return {
                "max_pain": None,
                "pain_values": {},
                "signal": "neutral",
                "interpretation": "Insufficient data for Max Pain calculation.",
            }

        pain_by_strike: Dict[float, float] = {}

        for k in strikes:
            pain = 0.0
            for row in rows:
                s = row["strike"]
                # Call side: CE holders lose if spot < strike
                pain += row["ce_oi"] * max(s - k, 0.0)
                # Put side: PE holders lose if spot > strike
                pain += row["pe_oi"] * max(k - s, 0.0)
            pain_by_strike[k] = pain

        max_pain_strike = min(pain_by_strike, key=pain_by_strike.get)

        # Sort pain values for display
        sorted_pain = sorted(pain_by_strike.items(), key=lambda x: x[1])
        top_5_low_pain = [{"strike": s, "pain": round(p)} for s, p in sorted_pain[:5]]

        return {
            "max_pain": max_pain_strike,
            "pain_values": {
                str(int(k)): round(v) for k, v in sorted(pain_by_strike.items())
            },
            "top_5_low_pain_strikes": top_5_low_pain,
            "signal": "neutral",
            "interpretation": (
                f"Max Pain at ₹{max_pain_strike:,.0f} — "
                "underlying tends to gravitate toward this level near expiry "
                "as market makers neutralise their delta exposure."
            ),
        }


# ---------------------------------------------------------------------------
# OI Walls Analyzer (Support / Resistance via OI)
# ---------------------------------------------------------------------------


class OIWallsAnalyzer:
    """
    Identifies the strongest call and put OI concentrations.

    • Max Call OI strike = resistance wall (market makers will sell/hedge there)
    • Max Put OI strike  = support wall (market makers will buy/hedge there)
    """

    def analyze(
        self,
        rows: List[Dict[str, Any]],
        spot: float,
        n_walls: int = 5,
    ) -> Dict[str, Any]:

        if not rows:
            return {
                "call_walls": [],
                "put_walls": [],
                "signal": "neutral",
                "key_resistance": None,
                "key_support": None,
                "interpretation": "No data.",
            }

        # Separate above / below spot
        calls_above = [r for r in rows if r["strike"] >= spot]
        puts_below = [r for r in rows if r["strike"] <= spot]

        # Sort by OI descending
        calls_above.sort(key=lambda r: r["ce_oi"], reverse=True)
        puts_below.sort(key=lambda r: r["pe_oi"], reverse=True)

        call_walls = [
            {
                "strike": r["strike"],
                "ce_oi": r["ce_oi"],
                "ce_oi_chg": r["ce_oi_chg"],
                "ce_iv": r["ce_iv"],
                "label": "resistance",
                "buildup": _oi_change_label(r["ce_oi_chg"]),
            }
            for r in calls_above[:n_walls]
        ]
        put_walls = [
            {
                "strike": r["strike"],
                "pe_oi": r["pe_oi"],
                "pe_oi_chg": r["pe_oi_chg"],
                "pe_iv": r["pe_iv"],
                "label": "support",
                "buildup": _oi_change_label(r["pe_oi_chg"]),
            }
            for r in puts_below[:n_walls]
        ]

        key_resistance = call_walls[0]["strike"] if call_walls else None
        key_support = put_walls[0]["strike"] if put_walls else None

        # Signal based on proximity of spot to walls
        signal = "neutral"
        interp_parts = []

        if key_resistance:
            dist_r = (key_resistance - spot) / spot * 100
            interp_parts.append(
                f"Call wall (resistance) at ₹{key_resistance:,.0f} "
                f"({dist_r:+.1f}% from spot)"
            )
            if dist_r < 1.5:
                signal = "bearish"
                interp_parts.append("— very close resistance, breakout needed.")
            elif dist_r > 5.0:
                signal = "bullish"
                interp_parts.append("— ample upside room before resistance.")

        if key_support:
            dist_s = (spot - key_support) / spot * 100
            interp_parts.append(
                f"Put wall (support) at ₹{key_support:,.0f} ({dist_s:.1f}% below spot)"
            )
            if dist_s < 1.0 and signal != "bearish":
                signal = "bullish"
                interp_parts.append("— strong support directly below.")

        return {
            "call_walls": call_walls,
            "put_walls": put_walls,
            "key_resistance": key_resistance,
            "key_support": key_support,
            "signal": signal,
            "interpretation": " | ".join(interp_parts) or "OI walls neutral.",
        }


def _oi_change_label(chg: float) -> str:
    """Classify OI change as buildup / unwinding / fresh."""
    if chg > 5000:
        return "strong buildup"
    elif chg > 0:
        return "buildup"
    elif chg < -5000:
        return "strong unwinding"
    elif chg < 0:
        return "unwinding"
    else:
        return "unchanged"


# ---------------------------------------------------------------------------
# GEX (Gamma Exposure) Calculator
# ---------------------------------------------------------------------------


class GEXCalculator:
    """
    Estimates Gamma Exposure (GEX) for each strike.

    GEX = OI × Gamma × Multiplier × Spot²
        (approximated using BSM-like gamma: we use IV and time to approximate)

    Since we don't have exact time-to-expiry per strike, we use a simplified
    approach: gamma proxy = 1/IV² (higher IV → lower gamma for ATM options).
    GEX is used to identify where dealers must hedge aggressively.

    Positive GEX: dealers are long gamma → they sell rallies + buy dips → market stabilising
    Negative GEX: dealers are short gamma → they buy rallies + sell dips → market destabilising
    """

    LOT_SIZE = 50  # NIFTY lot size (approximate)

    def calculate(self, rows: List[Dict[str, Any]], spot: float) -> Dict[str, Any]:
        if not rows:
            return {
                "total_gex": 0.0,
                "gex_by_strike": {},
                "signal": "neutral",
                "interpretation": "No data for GEX.",
            }

        gex_by_strike: Dict[float, float] = {}
        total_positive_gex = 0.0
        total_negative_gex = 0.0

        for row in rows:
            strike = row["strike"]
            ce_oi = row["ce_oi"]
            pe_oi = row["pe_oi"]
            ce_iv = row["ce_iv"] / 100.0 if row["ce_iv"] > 0 else 0.15
            pe_iv = row["pe_iv"] / 100.0 if row["pe_iv"] > 0 else 0.15

            # Approximate gamma: near-ATM gamma peaks, falls off with distance
            dist_pct = abs(strike - spot) / spot

            # BSM approximate gamma = N'(d1) / (S * σ * √T)
            # Simplified: gamma_proxy ≈ exp(-dist_pct² / (2 * σ²)) / (S * σ)
            ce_gamma = (
                math.exp(-(dist_pct**2) / (2 * ce_iv**2)) / (spot * ce_iv)
                if ce_iv > 0
                else 0.0
            )
            pe_gamma = (
                math.exp(-(dist_pct**2) / (2 * pe_iv**2)) / (spot * pe_iv)
                if pe_iv > 0
                else 0.0
            )

            # Dealers are short gamma on calls (they sold calls) → negative GEX
            # Dealers are short gamma on puts → positive GEX (they sold puts)
            call_gex = -ce_oi * ce_gamma * self.LOT_SIZE * (spot**2) * 0.01
            put_gex = pe_oi * pe_gamma * self.LOT_SIZE * (spot**2) * 0.01

            net_gex = call_gex + put_gex
            gex_by_strike[strike] = round(net_gex, 2)

            if net_gex >= 0:
                total_positive_gex += net_gex
            else:
                total_negative_gex += net_gex

        total_gex = total_positive_gex + total_negative_gex

        # Signal
        if total_gex > 0:
            signal = "bullish"
            interp = (
                f"Net GEX: +{total_gex:,.0f} — Positive gamma environment. "
                "Dealers stabilise by selling strength / buying weakness → lower volatility expected."
            )
        elif total_gex < 0:
            signal = "bearish"
            interp = (
                f"Net GEX: {total_gex:,.0f} — Negative gamma environment. "
                "Dealers destabilise by selling dips / buying rallies → higher volatility expected."
            )
        else:
            signal = "neutral"
            interp = "GEX near zero — dealers balanced, no directional pressure."

        # Top GEX strikes (largest absolute values)
        top_strikes = sorted(
            gex_by_strike.items(), key=lambda x: abs(x[1]), reverse=True
        )[:5]

        return {
            "total_gex": round(total_gex, 2),
            "positive_gex": round(total_positive_gex, 2),
            "negative_gex": round(total_negative_gex, 2),
            "top_gex_strikes": [{"strike": s, "gex": g} for s, g in top_strikes],
            "gex_by_strike": {str(int(k)): v for k, v in gex_by_strike.items()},
            "signal": signal,
            "interpretation": interp,
        }


# ---------------------------------------------------------------------------
# OI Change Analyzer (Buildup / Unwinding)
# ---------------------------------------------------------------------------


class OIChangeAnalyzer:
    """
    Classifies OI changes as:
      - Long buildup  : price up + OI up   (bullish)
      - Short covering: price up + OI down (bullish)
      - Short buildup : price down + OI up (bearish)
      - Long unwinding: price down + OI down (bearish)
    """

    def analyze(
        self,
        rows: List[Dict[str, Any]],
        spot: float,
        prev_spot: Optional[float] = None,
    ) -> Dict[str, Any]:

        # CE OI changes
        ce_buildup_strikes = [
            r["strike"] for r in rows if r["ce_oi_chg"] > 5000 and r["strike"] > spot
        ]
        ce_unwinding_strikes = [
            r["strike"] for r in rows if r["ce_oi_chg"] < -5000 and r["strike"] > spot
        ]
        pe_buildup_strikes = [
            r["strike"] for r in rows if r["pe_oi_chg"] > 5000 and r["strike"] < spot
        ]
        pe_unwinding_strikes = [
            r["strike"] for r in rows if r["pe_oi_chg"] < -5000 and r["strike"] < spot
        ]

        total_ce_chg = sum(r["ce_oi_chg"] for r in rows)
        total_pe_chg = sum(r["pe_oi_chg"] for r in rows)

        # Net interpretation
        if total_pe_chg > total_ce_chg and total_pe_chg > 0:
            signal = "bullish"
            interp = (
                f"Net put OI building (+{total_pe_chg:,.0f}) vs call OI "
                f"({total_ce_chg:+,.0f}) — bearish hedging increasing, supports underlying."
            )
        elif total_ce_chg > total_pe_chg and total_ce_chg > 0:
            signal = "bearish"
            interp = (
                f"Net call OI building (+{total_ce_chg:,.0f}) vs put OI "
                f"({total_pe_chg:+,.0f}) — sellers adding upside resistance."
            )
        elif total_ce_chg < 0 and total_pe_chg < 0:
            signal = "neutral"
            interp = "Both CE and PE OI declining — position unwinding, low conviction."
        else:
            signal = "neutral"
            interp = "OI change balanced — mixed positioning."

        return {
            "total_ce_oi_change": total_ce_chg,
            "total_pe_oi_change": total_pe_chg,
            "ce_buildup_strikes": ce_buildup_strikes[:5],
            "ce_unwinding_strikes": ce_unwinding_strikes[:5],
            "pe_buildup_strikes": pe_buildup_strikes[:5],
            "pe_unwinding_strikes": pe_unwinding_strikes[:5],
            "signal": signal,
            "interpretation": interp,
        }


# ---------------------------------------------------------------------------
# IV Skew Analyzer
# ---------------------------------------------------------------------------


class IVSkewAnalyzer:
    """
    Analyzes IV skew:
    - Positive skew (put IV > call IV) = market fears downside → protective puts expensive
    - Negative skew (call IV > put IV) = market fears upside / momentum → calls expensive
    """

    def analyze(self, rows: List[Dict[str, Any]], spot: float) -> Dict[str, Any]:
        if not rows:
            return {"skew": 0.0, "signal": "neutral", "interpretation": "No IV data."}

        atm_strike = _find_atm_strike(spot, [r["strike"] for r in rows])

        # ATM IV
        atm_rows = [r for r in rows if r["strike"] == atm_strike]
        if not atm_rows:
            return {
                "skew": 0.0,
                "signal": "neutral",
                "interpretation": "No ATM row found.",
            }

        atm = atm_rows[0]
        atm_ce_iv = atm["ce_iv"]
        atm_pe_iv = atm["pe_iv"]
        atm_iv_avg = (atm_ce_iv + atm_pe_iv) / 2 if (atm_ce_iv + atm_pe_iv) > 0 else 0

        # OTM IV (2 strikes away)
        otm_strikes_call = [r["strike"] for r in rows if r["strike"] > spot]
        otm_strikes_put = [r["strike"] for r in rows if r["strike"] < spot]

        otm_call_iv = 0.0
        otm_put_iv = 0.0

        if len(otm_strikes_call) >= 2:
            otm_call_strike = sorted(otm_strikes_call)[1]
            otm_call_rows = [r for r in rows if r["strike"] == otm_call_strike]
            if otm_call_rows:
                otm_call_iv = otm_call_rows[0]["ce_iv"]

        if len(otm_strikes_put) >= 2:
            otm_put_strike = sorted(otm_strikes_put, reverse=True)[1]
            otm_put_rows = [r for r in rows if r["strike"] == otm_put_strike]
            if otm_put_rows:
                otm_put_iv = otm_put_rows[0]["pe_iv"]

        # Skew = OTM Put IV - OTM Call IV (positive = fear of downside)
        skew = otm_put_iv - otm_call_iv

        if skew > 3.0:
            signal = "bearish"
            interp = (
                f"Positive IV skew ({skew:+.1f}%) — OTM puts expensive relative to calls. "
                "Market is paying for downside protection → bearish fear present."
            )
        elif skew < -3.0:
            signal = "bullish"
            interp = (
                f"Negative IV skew ({skew:+.1f}%) — OTM calls expensive. "
                "Demand for upside calls → momentum/bullish positioning."
            )
        else:
            signal = "neutral"
            interp = (
                f"IV skew near flat ({skew:+.1f}%) — balanced fear/greed between "
                "upside and downside options."
            )

        return {
            "atm_strike": atm_strike,
            "atm_ce_iv": round(atm_ce_iv, 2),
            "atm_pe_iv": round(atm_pe_iv, 2),
            "atm_iv_avg": round(atm_iv_avg, 2),
            "otm_call_iv": round(otm_call_iv, 2),
            "otm_put_iv": round(otm_put_iv, 2),
            "skew": round(skew, 2),
            "signal": signal,
            "interpretation": interp,
        }


# ---------------------------------------------------------------------------
# ATM Analysis
# ---------------------------------------------------------------------------


class ATMAnalyzer:
    """
    Focused analysis on ATM and near-ATM strikes.
    Returns a rich view of the most actively traded options.
    """

    def analyze(
        self, rows: List[Dict[str, Any]], spot: float, window: int = 5
    ) -> Dict[str, Any]:
        if not rows:
            return {"atm_strike": None, "near_atm": [], "signal": "neutral"}

        all_strikes = sorted(set(r["strike"] for r in rows))
        atm_strike = _find_atm_strike(spot, all_strikes)

        # Find the index of ATM strike
        try:
            atm_idx = all_strikes.index(atm_strike)
        except ValueError:
            atm_idx = len(all_strikes) // 2

        lo = max(0, atm_idx - window)
        hi = min(len(all_strikes), atm_idx + window + 1)
        near_strikes = set(all_strikes[lo:hi])

        near_rows = [r for r in rows if r["strike"] in near_strikes]
        near_rows.sort(key=lambda r: r["strike"])

        # Format for display
        near_atm_display = []
        for r in near_rows:
            tag = (
                "ATM"
                if r["strike"] == atm_strike
                else ("ITM-CE / OTM-PE" if r["strike"] < spot else "OTM-CE / ITM-PE")
            )
            near_atm_display.append(
                {
                    "strike": r["strike"],
                    "tag": tag,
                    "ce_oi": r["ce_oi"],
                    "ce_oi_chg": r["ce_oi_chg"],
                    "ce_iv": r["ce_iv"],
                    "ce_ltp": r["ce_ltp"],
                    "pe_oi": r["pe_oi"],
                    "pe_oi_chg": r["pe_oi_chg"],
                    "pe_iv": r["pe_iv"],
                    "pe_ltp": r["pe_ltp"],
                    "total_oi": r["total_oi"],
                }
            )

        return {
            "atm_strike": atm_strike,
            "near_atm": near_atm_display,
            "signal": "neutral",
        }


# ---------------------------------------------------------------------------
# Master OI Analyzer — orchestrates all OI analyses
# ---------------------------------------------------------------------------


class OIAnalyzer:
    """
    Master Open Interest Analyzer.

    Parses a raw NSE option chain response and runs all OI calculations:
      • PCR (Put-Call Ratio)
      • Max Pain
      • OI Walls (support/resistance)
      • GEX (Gamma Exposure)
      • OI Change Analysis
      • IV Skew
      • ATM Analysis

    Usage::

        analyzer = OIAnalyzer()
        result = analyzer.analyze(raw_option_chain, spot_price=22400)

        # Or convenience wrapper:
        result = analyze_oi(raw_option_chain, spot_price=22400)
    """

    def __init__(self) -> None:
        self._parser = OptionChainParser()
        self._pcr = PCRCalculator()
        self._max_pain = MaxPainCalculator()
        self._oi_walls = OIWallsAnalyzer()
        self._gex = GEXCalculator()
        self._oi_chg = OIChangeAnalyzer()
        self._iv_skew = IVSkewAnalyzer()
        self._atm = ATMAnalyzer()

    def analyze(
        self,
        raw_option_chain: dict,
        spot_price: float = 0.0,
        n_walls: int = 5,
    ) -> dict:
        """
        Run all OI analyses on a raw NSE option chain response.

        Args:
            raw_option_chain: Raw JSON dict from NSE option chain API
            spot_price:       Override spot price (uses underlyingValue if 0)
            n_walls:          Number of OI walls to return per side

        Returns:
            Comprehensive OI analysis dict with all sub-analyses
        """
        # Parse raw chain
        spot, strikes, rows = self._parser.parse(raw_option_chain)

        # Use override if provided
        if spot_price > 0:
            spot = spot_price

        if not rows or spot <= 0:
            return {
                "error": "Insufficient option chain data",
                "spot": spot,
                "rows": len(rows),
            }

        # Run all analyses
        pcr = self._pcr.calculate(rows)
        max_pain = self._max_pain.calculate(rows, strikes)
        oi_walls = self._oi_walls.analyze(rows, spot, n=n_walls)
        gex = self._gex.calculate(rows, spot)
        oi_changes = self._oi_chg.analyze(rows, spot)
        iv_skew = self._iv_skew.analyze(rows, spot)
        atm = self._atm.analyze(rows, spot)

        # Ensemble signal
        signals = [
            pcr.get("signal", "neutral"),
            oi_walls.get("signal", "neutral"),
            oi_changes.get("signal", "neutral"),
            iv_skew.get("signal", "neutral"),
        ]
        bull_count = signals.count("bullish")
        bear_count = signals.count("bearish")
        if bull_count > bear_count:
            overall_signal = "bullish"
            overall_conf = round(bull_count / len(signals), 3)
        elif bear_count > bull_count:
            overall_signal = "bearish"
            overall_conf = round(bear_count / len(signals), 3)
        else:
            overall_signal = "neutral"
            overall_conf = 0.4

        return {
            "spot": round(spot, 2),
            "strikes_count": len(strikes),
            "rows_analyzed": len(rows),
            "pcr": pcr,
            "max_pain": max_pain,
            "oi_walls": oi_walls,
            "gex": gex,
            "oi_changes": oi_changes,
            "iv_skew": iv_skew,
            "atm_analysis": atm,
            "overall_signal": overall_signal,
            "overall_confidence": overall_conf,
            "signal_breakdown": {
                "pcr": pcr.get("signal"),
                "oi_walls": oi_walls.get("signal"),
                "oi_changes": oi_changes.get("signal"),
                "iv_skew": iv_skew.get("signal"),
            },
        }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def analyze_oi(
    raw_option_chain: dict,
    spot_price: float = 0.0,
    n_walls: int = 5,
) -> dict:
    """
    Convenience wrapper — analyze a raw NSE option chain dict.

    Args:
        raw_option_chain: Raw JSON from NSE option chain API
        spot_price:       Override spot price
        n_walls:          OI walls to return per side

    Returns:
        Full OI analysis dict
    """
    analyzer = OIAnalyzer()
    return analyzer.analyze(raw_option_chain, spot_price=spot_price, n_walls=n_walls)
