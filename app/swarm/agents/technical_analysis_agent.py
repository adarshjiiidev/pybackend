"""
TechnicalAnalysisAgent — Full Technical Analysis Suite
=======================================================
Phase 2 · Swarm Agent

Runs a comprehensive technical analysis on any NSE symbol:
  • All TA indicators: RSI, MACD, Bollinger Bands, Stochastic, ATR, ADX,
    CCI, VWAP, Ichimoku Cloud, Supertrend, EMA/SMA crossovers, OBV, MFI
  • Candlestick pattern detection (last 5 bars)
  • Chart pattern detection (Double Top/Bottom, H&S, Triangles, Flags)
  • Support / Resistance levels (pivot-based + OI-informed if available)
  • Multi-timeframe signal confluence (daily + weekly)
  • Trading signal: STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL
  • Confidence score (0.0–1.0) based on indicator agreement

Input payload keys:
  symbol      (str, required)   NSE symbol, e.g. "RELIANCE"
  days        (int, default 365) Historical lookback in days
  interval    (str, default "1d") Bar interval
  include_patterns (bool, default True)

Output AgentResult.data keys:
  symbol, current_price, indicators, patterns, signals,
  support_resistance, trading_signal, confidence, summary_text
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..base_agent import AgentResult, AgentStatus, BaseSwarmAgent, SwarmMessage

logger = logging.getLogger(__name__)

AGENT_TYPE = "technical_analysis"


class TechnicalAnalysisAgent(BaseSwarmAgent):
    """
    Autonomous Technical Analysis Agent.

    Fetches OHLCV data (from pool or live), runs the full TA engine,
    detects patterns, scores signals, and returns a rich structured result
    the Orchestrator can use directly or feed into a ReportAgent.
    """

    AGENT_TYPE = AGENT_TYPE
    DEFAULT_TIMEOUT_S = 60.0

    # ── Indicator signal weights for overall score ──────────────────────────
    # Each indicator votes with weight W; final score → -100 to +100
    _WEIGHTS: Dict[str, float] = {
        "rsi": 1.5,
        "macd": 1.5,
        "bollinger": 1.0,
        "stochastic": 1.0,
        "adx_trend": 0.8,
        "ema_cross": 2.0,
        "sma_cross": 1.2,
        "supertrend": 2.0,
        "ichimoku": 1.5,
        "obv": 0.8,
        "mfi": 1.0,
        "cci": 0.8,
        "pattern": 2.5,  # candlestick / chart patterns carry high weight
    }

    # ────────────────────────────────────────────────────────────────────────
    # execute()
    # ────────────────────────────────────────────────────────────────────────

    async def execute(self, message: SwarmMessage) -> AgentResult:
        payload = message.payload
        symbol: str = str(payload.get("symbol", "NIFTY")).upper().strip()
        days: int = int(payload.get("days", 365))
        interval: str = str(payload.get("interval", "1d"))
        include_patterns: bool = bool(payload.get("include_patterns", True))

        self._log.info(f"TA agent starting: {symbol} | days={days} interval={interval}")

        # ── 1. Fetch OHLCV ────────────────────────────────────────────────
        df = await self.tools.get_ohlcv(symbol, days=days, interval=interval)

        if df is None or df.empty or len(df) < 20:
            return self._ok(
                data={"symbol": symbol, "error": "Insufficient historical data"},
                summary=f"Not enough data for {symbol} technical analysis.",
                signal="neutral",
                confidence=0.1,
            )

        # Ensure standard lowercase columns
        df = self._normalise_df(df)
        current_price = float(df["close"].iloc[-1])
        self._log.info(
            f"Fetched {len(df)} bars for {symbol} | latest close ₹{current_price:,.2f}"
        )

        # ── 2. Calculate all indicators ───────────────────────────────────
        indicators = self._calc_all_indicators(df)

        # ── 3. Detect patterns ────────────────────────────────────────────
        patterns: Dict[str, Any] = {}
        if include_patterns:
            try:
                detector = self.tools.pattern_detector
                patterns = detector.detect_all(df, lookback_candles=5)
            except Exception as exc:
                self._log.warning(f"Pattern detection error: {exc}")
                patterns = {
                    "candlestick": [],
                    "chart": [],
                    "summary": {"signal": "neutral"},
                }

        # ── 4. Support / Resistance ───────────────────────────────────────
        sr_levels = self._calc_support_resistance(df, current_price)

        # ── 5. Multi-timeframe confluence (weekly) ────────────────────────
        weekly_signal = await self._weekly_timeframe(symbol)

        # ── 6. Score all signals → overall verdict ────────────────────────
        signal_votes = self._collect_votes(indicators, patterns, weekly_signal)
        trading_signal, score, confidence = self._score_votes(signal_votes)

        # ── 7. Build summary text ─────────────────────────────────────────
        summary = self._build_summary(
            symbol,
            current_price,
            trading_signal,
            score,
            confidence,
            indicators,
            patterns,
            sr_levels,
        )

        # ── 8. Extract last 90 price bars for chart rendering ─────────────
        price_bars: List[Dict[str, Any]] = []
        try:
            chart_df = df.tail(90).copy()
            for idx, row in chart_df.iterrows():
                ts = idx
                # Convert index to ISO date string
                if hasattr(ts, 'strftime'):
                    time_str = ts.strftime('%Y-%m-%d')
                else:
                    time_str = str(ts)[:10]
                bar: Dict[str, Any] = {
                    "time": time_str,
                    "open": round(float(row.get("open", row["close"])), 2),
                    "high": round(float(row.get("high", row["close"])), 2),
                    "low": round(float(row.get("low", row["close"])), 2),
                    "close": round(float(row["close"]), 2),
                    "value": round(float(row["close"]), 2),
                }
                vol = row.get("volume")
                if vol is not None and not pd.isna(vol) and float(vol) > 0:
                    bar["volume"] = int(float(vol))
                price_bars.append(bar)
        except Exception as e:
            self._log.debug(f"Price bars extraction error: {e}")

        return self._ok(
            data={
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "bars_analyzed": len(df),
                "indicators": indicators,
                "patterns": patterns,
                "support_resistance": sr_levels,
                "signal_votes": signal_votes,
                "score": round(score, 2),
                "trading_signal": trading_signal,
                "weekly_signal": weekly_signal,
                "price_bars": price_bars,
                "analysis_date": pd.Timestamp.utcnow().isoformat(),
            },
            summary=summary,
            signal="bullish"
            if score > 20
            else ("bearish" if score < -20 else "neutral"),
            confidence=confidence,
            metadata={
                "agent": AGENT_TYPE,
                "symbol": symbol,
                "bars": len(df),
                "interval": interval,
            },
        )


    # ────────────────────────────────────────────────────────────────────────
    # Indicator calculations
    # ────────────────────────────────────────────────────────────────────────

    def _calc_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate every major technical indicator.
        Returns a flat dict with values + signals + interpretations.
        """
        results: Dict[str, Any] = {}

        closes = df["close"]
        highs = df["high"]
        lows = df["low"]
        opens = df["open"]
        volume = df.get("volume", pd.Series(dtype=float))
        n = len(closes)

        # ── RSI ─────────────────────────────────────────────────────────
        try:
            rsi_14 = self._rsi(closes, 14)
            rsi_val = float(rsi_14.iloc[-1]) if not pd.isna(rsi_14.iloc[-1]) else None

            if rsi_val is not None:
                if rsi_val >= 70:
                    rsi_signal, rsi_interp = (
                        "bearish",
                        f"RSI {rsi_val:.1f} — Overbought. Pullback risk.",
                    )
                elif rsi_val <= 30:
                    rsi_signal, rsi_interp = (
                        "bullish",
                        f"RSI {rsi_val:.1f} — Oversold. Bounce potential.",
                    )
                elif rsi_val >= 55:
                    rsi_signal, rsi_interp = (
                        "bullish",
                        f"RSI {rsi_val:.1f} — Bullish momentum zone.",
                    )
                elif rsi_val <= 45:
                    rsi_signal, rsi_interp = (
                        "bearish",
                        f"RSI {rsi_val:.1f} — Bearish momentum zone.",
                    )
                else:
                    rsi_signal, rsi_interp = (
                        "neutral",
                        f"RSI {rsi_val:.1f} — Neutral zone (45–55).",
                    )

                results["rsi"] = {
                    "value_14": round(rsi_val, 2),
                    "value_21": round(float(self._rsi(closes, 21).iloc[-1]), 2)
                    if n >= 21
                    else None,
                    "signal": rsi_signal,
                    "interpretation": rsi_interp,
                }
        except Exception as e:
            self._log.debug(f"RSI error: {e}")

        # ── MACD ─────────────────────────────────────────────────────────
        try:
            ema12 = closes.ewm(span=12, adjust=False).mean()
            ema26 = closes.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line

            macd_val = float(macd_line.iloc[-1])
            sig_val = float(signal_line.iloc[-1])
            hist_val = float(histogram.iloc[-1])
            prev_hist = float(histogram.iloc[-2]) if n >= 2 else 0.0

            if macd_val > sig_val and hist_val > 0 and hist_val > prev_hist:
                macd_signal = "bullish"
                macd_interp = f"MACD {macd_val:.2f} > Signal {sig_val:.2f} — Bullish momentum accelerating."
            elif macd_val > sig_val:
                macd_signal = "bullish"
                macd_interp = f"MACD above signal line — Bullish bias."
            elif macd_val < sig_val and hist_val < 0 and hist_val < prev_hist:
                macd_signal = "bearish"
                macd_interp = f"MACD {macd_val:.2f} < Signal {sig_val:.2f} — Bearish momentum accelerating."
            elif macd_val < sig_val:
                macd_signal = "bearish"
                macd_interp = "MACD below signal line — Bearish bias."
            else:
                macd_signal = "neutral"
                macd_interp = "MACD crossing — direction change imminent."

            results["macd"] = {
                "macd": round(macd_val, 4),
                "signal": round(sig_val, 4),
                "histogram": round(hist_val, 4),
                "signal_direction": macd_signal,
                "interpretation": macd_interp,
            }
        except Exception as e:
            self._log.debug(f"MACD error: {e}")

        # ── Bollinger Bands ───────────────────────────────────────────────
        try:
            sma20 = closes.rolling(20).mean()
            std20 = closes.rolling(20).std()
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            bb_mid = sma20

            price = float(closes.iloc[-1])
            bb_u = float(bb_upper.iloc[-1])
            bb_l = float(bb_lower.iloc[-1])
            bb_m = float(bb_mid.iloc[-1])
            bb_width = (bb_u - bb_l) / bb_m if bb_m > 0 else 0.0

            # %B position
            pct_b = (price - bb_l) / (bb_u - bb_l) if (bb_u - bb_l) > 0 else 0.5

            if price >= bb_u:
                bb_signal = "bearish"
                bb_interp = (
                    f"Price at upper BB (₹{bb_u:.1f}) — Overbought, potential reversal."
                )
            elif price <= bb_l:
                bb_signal = "bullish"
                bb_interp = (
                    f"Price at lower BB (₹{bb_l:.1f}) — Oversold, bounce likely."
                )
            elif price > bb_m:
                bb_signal = "bullish"
                bb_interp = f"Price above BB midline ₹{bb_m:.1f} — Bullish position."
            else:
                bb_signal = "bearish"
                bb_interp = f"Price below BB midline ₹{bb_m:.1f} — Bearish position."

            results["bollinger_bands"] = {
                "upper": round(bb_u, 2),
                "middle": round(bb_m, 2),
                "lower": round(bb_l, 2),
                "width_pct": round(bb_width * 100, 2),
                "pct_b": round(pct_b, 3),
                "signal": bb_signal,
                "interpretation": bb_interp,
            }
        except Exception as e:
            self._log.debug(f"BB error: {e}")

        # ── Stochastic Oscillator ─────────────────────────────────────────
        try:
            period = 14
            low14 = lows.rolling(period).min()
            high14 = highs.rolling(period).max()
            stoch_k = 100 * (closes - low14) / (high14 - low14 + 1e-9)
            stoch_d = stoch_k.rolling(3).mean()

            k_val = float(stoch_k.iloc[-1])
            d_val = float(stoch_d.iloc[-1])

            if k_val >= 80:
                st_signal = "bearish"
                st_interp = f"Stoch %K {k_val:.1f} — Overbought zone."
            elif k_val <= 20:
                st_signal = "bullish"
                st_interp = f"Stoch %K {k_val:.1f} — Oversold zone, potential reversal."
            elif k_val > d_val and k_val < 80:
                st_signal = "bullish"
                st_interp = (
                    f"Stoch %K ({k_val:.1f}) > %D ({d_val:.1f}) — Bullish crossover."
                )
            elif k_val < d_val:
                st_signal = "bearish"
                st_interp = (
                    f"Stoch %K ({k_val:.1f}) < %D ({d_val:.1f}) — Bearish crossover."
                )
            else:
                st_signal = "neutral"
                st_interp = f"Stochastic neutral at {k_val:.1f}."

            results["stochastic"] = {
                "k": round(k_val, 2),
                "d": round(d_val, 2),
                "signal": st_signal,
                "interpretation": st_interp,
            }
        except Exception as e:
            self._log.debug(f"Stochastic error: {e}")

        # ── ATR ───────────────────────────────────────────────────────────
        try:
            atr_val = self._atr(highs, lows, closes, 14)
            atr_pct = (atr_val / float(closes.iloc[-1])) * 100

            results["atr"] = {
                "value": round(atr_val, 2),
                "atr_pct": round(atr_pct, 2),
                "interpretation": (
                    f"ATR₁₄ = ₹{atr_val:.2f} ({atr_pct:.1f}% of price) — "
                    + (
                        "High volatility."
                        if atr_pct > 2.5
                        else "Moderate volatility."
                        if atr_pct > 1.5
                        else "Low volatility."
                    )
                ),
            }
        except Exception as e:
            self._log.debug(f"ATR error: {e}")

        # ── ADX (trend strength) ──────────────────────────────────────────
        try:
            adx_val, di_plus, di_minus = self._adx(highs, lows, closes, 14)

            if adx_val >= 40:
                trend_str = "Very strong trend"
            elif adx_val >= 25:
                trend_str = "Trending"
            elif adx_val >= 15:
                trend_str = "Developing trend"
            else:
                trend_str = "Ranging / no clear trend"

            if adx_val >= 20 and di_plus > di_minus:
                adx_signal = "bullish"
            elif adx_val >= 20 and di_minus > di_plus:
                adx_signal = "bearish"
            else:
                adx_signal = "neutral"

            results["adx"] = {
                "adx": round(adx_val, 2),
                "di_plus": round(di_plus, 2),
                "di_minus": round(di_minus, 2),
                "trend_strength": trend_str,
                "signal": adx_signal,
                "interpretation": (
                    f"ADX {adx_val:.1f} ({trend_str}) | +DI {di_plus:.1f} vs -DI {di_minus:.1f}"
                ),
            }
        except Exception as e:
            self._log.debug(f"ADX error: {e}")

        # ── CCI ───────────────────────────────────────────────────────────
        try:
            cci_period = 20
            tp = (highs + lows + closes) / 3
            tp_sma = tp.rolling(cci_period).mean()
            mean_dev = tp.rolling(cci_period).apply(
                lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
            )
            cci_val = float(((tp - tp_sma) / (0.015 * mean_dev + 1e-9)).iloc[-1])

            if cci_val >= 100:
                cci_signal = "bearish"
                cci_interp = f"CCI {cci_val:.0f} — Overbought above +100."
            elif cci_val <= -100:
                cci_signal = "bullish"
                cci_interp = f"CCI {cci_val:.0f} — Oversold below -100."
            elif cci_val > 0:
                cci_signal = "bullish"
                cci_interp = f"CCI {cci_val:.0f} — Positive, mild bullish."
            else:
                cci_signal = "bearish"
                cci_interp = f"CCI {cci_val:.0f} — Negative, mild bearish."

            results["cci"] = {
                "value": round(cci_val, 2),
                "signal": cci_signal,
                "interpretation": cci_interp,
            }
        except Exception as e:
            self._log.debug(f"CCI error: {e}")

        # ── EMA crossovers ────────────────────────────────────────────────
        try:
            ema9 = float(closes.ewm(span=9, adjust=False).mean().iloc[-1])
            ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = (
                float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
                if n >= 50
                else None
            )
            ema200 = (
                float(closes.ewm(span=200, adjust=False).mean().iloc[-1])
                if n >= 200
                else None
            )

            price = float(closes.iloc[-1])

            ema_signals = []
            if ema9 > ema20:
                ema_signals.append("bullish")
            else:
                ema_signals.append("bearish")

            if ema50 is not None:
                if price > ema50:
                    ema_signals.append("bullish")
                else:
                    ema_signals.append("bearish")

            if ema200 is not None:
                if price > ema200:
                    ema_signals.append("bullish")
                else:
                    ema_signals.append("bearish")

            bull_ema = ema_signals.count("bullish")
            bear_ema = ema_signals.count("bearish")
            ema_signal = (
                "bullish"
                if bull_ema > bear_ema
                else ("bearish" if bear_ema > bull_ema else "neutral")
            )

            results["ema"] = {
                "ema_9": round(ema9, 2),
                "ema_20": round(ema20, 2),
                "ema_50": round(ema50, 2) if ema50 else None,
                "ema_200": round(ema200, 2) if ema200 else None,
                "price_vs_ema50": ("above" if ema50 and price > ema50 else "below")
                if ema50
                else "n/a",
                "price_vs_ema200": ("above" if ema200 and price > ema200 else "below")
                if ema200
                else "n/a",
                "9x20_cross": "golden" if ema9 > ema20 else "death",
                "signal": ema_signal,
                "interpretation": (
                    f"EMA9 {'>' if ema9 > ema20 else '<'} EMA20 | "
                    f"Price {'above' if ema50 and price > ema50 else 'below'} EMA50"
                ),
            }
        except Exception as e:
            self._log.debug(f"EMA error: {e}")

        # ── SMA crossovers ────────────────────────────────────────────────
        try:
            sma20_val = float(closes.rolling(20).mean().iloc[-1])
            sma50_val = float(closes.rolling(50).mean().iloc[-1]) if n >= 50 else None
            sma200_val = (
                float(closes.rolling(200).mean().iloc[-1]) if n >= 200 else None
            )
            price = float(closes.iloc[-1])

            sma_signal = "neutral"
            sma_interp_parts = []

            if sma50_val and sma200_val:
                if sma50_val > sma200_val:
                    sma_signal = "bullish"
                    sma_interp_parts.append(
                        "Golden cross: SMA50 > SMA200 (long-term bullish)"
                    )
                else:
                    sma_signal = "bearish"
                    sma_interp_parts.append(
                        "Death cross: SMA50 < SMA200 (long-term bearish)"
                    )

            if price > sma20_val:
                sma_interp_parts.append(f"Price above SMA20 ₹{sma20_val:.1f}")
            else:
                sma_interp_parts.append(f"Price below SMA20 ₹{sma20_val:.1f}")

            results["sma"] = {
                "sma_20": round(sma20_val, 2),
                "sma_50": round(sma50_val, 2) if sma50_val else None,
                "sma_200": round(sma200_val, 2) if sma200_val else None,
                "signal": sma_signal,
                "interpretation": " | ".join(sma_interp_parts),
            }
        except Exception as e:
            self._log.debug(f"SMA error: {e}")

        # ── Supertrend ────────────────────────────────────────────────────
        try:
            st_signal, st_line = self._supertrend(
                highs, lows, closes, period=10, multiplier=3.0
            )
            price = float(closes.iloc[-1])
            st_val = float(st_line.iloc[-1]) if not pd.isna(st_line.iloc[-1]) else price

            results["supertrend"] = {
                "value": round(st_val, 2),
                "signal": st_signal,
                "interpretation": (
                    f"Supertrend(10,3) = ₹{st_val:.2f} — "
                    + (
                        "Price above Supertrend: BULLISH."
                        if st_signal == "bullish"
                        else "Price below Supertrend: BEARISH."
                    )
                ),
            }
        except Exception as e:
            self._log.debug(f"Supertrend error: {e}")

        # ── Ichimoku Cloud ────────────────────────────────────────────────
        try:
            ichimoku = self._ichimoku(highs, lows, closes)
            results["ichimoku"] = ichimoku
        except Exception as e:
            self._log.debug(f"Ichimoku error: {e}")

        # ── VWAP (intraday only — skip for daily) ─────────────────────────
        try:
            if not volume.empty and volume.sum() > 0:
                tp = (highs + lows + closes) / 3
                vwap = (tp * volume).cumsum() / volume.cumsum()
                vwap_val = float(vwap.iloc[-1])
                price = float(closes.iloc[-1])
                results["vwap"] = {
                    "value": round(vwap_val, 2),
                    "signal": "bullish" if price > vwap_val else "bearish",
                    "interpretation": (
                        f"VWAP ₹{vwap_val:.2f} | Price "
                        f"{'above' if price > vwap_val else 'below'} VWAP"
                    ),
                }
        except Exception as e:
            self._log.debug(f"VWAP error: {e}")

        # ── OBV (On Balance Volume) ───────────────────────────────────────
        try:
            if not volume.empty and volume.sum() > 0:
                direction = closes.diff().apply(
                    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
                )
                obv = (direction * volume).cumsum()
                obv_ema = obv.ewm(span=20, adjust=False).mean()
                obv_now = float(obv.iloc[-1])
                obv_ema_now = float(obv_ema.iloc[-1])

                obv_signal = "bullish" if obv_now > obv_ema_now else "bearish"
                results["obv"] = {
                    "value": round(obv_now, 0),
                    "ema20": round(obv_ema_now, 0),
                    "signal": obv_signal,
                    "interpretation": (
                        f"OBV {'above' if obv_signal == 'bullish' else 'below'} 20-EMA — "
                        f"{'Volume accumulation.' if obv_signal == 'bullish' else 'Volume distribution.'}"
                    ),
                }
        except Exception as e:
            self._log.debug(f"OBV error: {e}")

        # ── MFI (Money Flow Index) ────────────────────────────────────────
        try:
            if not volume.empty and volume.sum() > 0:
                tp = (highs + lows + closes) / 3
                raw_mf = tp * volume
                pos_mf = raw_mf.where(tp > tp.shift(1), 0.0)
                neg_mf = raw_mf.where(tp < tp.shift(1), 0.0)
                pos_sum = pos_mf.rolling(14).sum()
                neg_sum = neg_mf.rolling(14).sum()
                mfr = pos_sum / (neg_sum + 1e-9)
                mfi = 100 - (100 / (1 + mfr))
                mfi_val = float(mfi.iloc[-1])

                if mfi_val >= 80:
                    mfi_signal = "bearish"
                    mfi_interp = (
                        f"MFI {mfi_val:.1f} — Overbought. Selling pressure building."
                    )
                elif mfi_val <= 20:
                    mfi_signal = "bullish"
                    mfi_interp = (
                        f"MFI {mfi_val:.1f} — Oversold. Buying pressure building."
                    )
                elif mfi_val > 50:
                    mfi_signal = "bullish"
                    mfi_interp = f"MFI {mfi_val:.1f} — Positive money flow."
                else:
                    mfi_signal = "bearish"
                    mfi_interp = f"MFI {mfi_val:.1f} — Negative money flow."

                results["mfi"] = {
                    "value": round(mfi_val, 2),
                    "signal": mfi_signal,
                    "interpretation": mfi_interp,
                }
        except Exception as e:
            self._log.debug(f"MFI error: {e}")

        # ── Williams %R ───────────────────────────────────────────────────
        try:
            period = 14
            hh = highs.rolling(period).max()
            ll = lows.rolling(period).min()
            wr = -100 * (hh - closes) / (hh - ll + 1e-9)
            wr_val = float(wr.iloc[-1])

            if wr_val >= -20:
                wr_signal = "bearish"
                wr_interp = f"Williams %R {wr_val:.1f} — Overbought."
            elif wr_val <= -80:
                wr_signal = "bullish"
                wr_interp = f"Williams %R {wr_val:.1f} — Oversold."
            elif wr_val > -50:
                wr_signal = "bullish"
                wr_interp = f"Williams %R {wr_val:.1f} — Bullish zone."
            else:
                wr_signal = "bearish"
                wr_interp = f"Williams %R {wr_val:.1f} — Bearish zone."

            results["williams_r"] = {
                "value": round(wr_val, 2),
                "signal": wr_signal,
                "interpretation": wr_interp,
            }
        except Exception as e:
            self._log.debug(f"Williams %R error: {e}")

        # ── Pivot Points (Classic) ────────────────────────────────────────
        try:
            last = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
            pivot = (
                float(last["high"]) + float(last["low"]) + float(last["close"])
            ) / 3
            r1 = 2 * pivot - float(last["low"])
            r2 = pivot + (float(last["high"]) - float(last["low"]))
            s1 = 2 * pivot - float(last["high"])
            s2 = pivot - (float(last["high"]) - float(last["low"]))
            price = float(closes.iloc[-1])

            results["pivot_points"] = {
                "pivot": round(pivot, 2),
                "r1": round(r1, 2),
                "r2": round(r2, 2),
                "s1": round(s1, 2),
                "s2": round(s2, 2),
                "signal": "bullish" if price > pivot else "bearish",
                "interpretation": (
                    f"Classic Pivot: ₹{pivot:.2f} | "
                    f"R1 ₹{r1:.2f} | R2 ₹{r2:.2f} | "
                    f"S1 ₹{s1:.2f} | S2 ₹{s2:.2f}"
                ),
            }
        except Exception as e:
            self._log.debug(f"Pivot error: {e}")

        return results

    # ────────────────────────────────────────────────────────────────────────
    # Support / Resistance
    # ────────────────────────────────────────────────────────────────────────

    def _calc_support_resistance(
        self, df: pd.DataFrame, current_price: float
    ) -> Dict[str, Any]:
        """
        Identify key S/R levels using:
          1. Rolling window pivot highs / lows (last 252 bars)
          2. Round numbers near price
          3. 52-week high / low
        """
        closes = df["close"]
        highs = df["high"]
        lows = df["low"]
        n = len(df)

        resistance_levels: List[float] = []
        support_levels: List[float] = []

        window = 10
        for i in range(window, n - window):
            seg_h = highs.iloc[i - window : i + window + 1]
            seg_l = lows.iloc[i - window : i + window + 1]
            if float(highs.iloc[i]) == float(seg_h.max()):
                resistance_levels.append(float(highs.iloc[i]))
            if float(lows.iloc[i]) == float(seg_l.min()):
                support_levels.append(float(lows.iloc[i]))

        # Keep only levels above / below current price
        resistances = sorted(
            [r for r in resistance_levels if r > current_price * 1.005], reverse=False
        )[:5]
        supports = sorted(
            [s for s in support_levels if s < current_price * 0.995], reverse=True
        )[:5]

        # 52-week high / low
        wk52_high = float(highs.tail(252).max()) if n >= 252 else float(highs.max())
        wk52_low = float(lows.tail(252).min()) if n >= 252 else float(lows.min())

        # Nearest round number
        magnitude = 10 ** (len(str(int(current_price))) - 2)
        nearest_round = round(current_price / magnitude) * magnitude

        return {
            "current_price": round(current_price, 2),
            "immediate_resistance": round(resistances[0], 2) if resistances else None,
            "immediate_support": round(supports[0], 2) if supports else None,
            "resistance_levels": [round(r, 2) for r in resistances[:3]],
            "support_levels": [round(s, 2) for s in supports[:3]],
            "52w_high": round(wk52_high, 2),
            "52w_low": round(wk52_low, 2),
            "nearest_round_number": round(nearest_round, 2),
            "pct_from_52w_high": round(
                (current_price - wk52_high) / wk52_high * 100, 2
            ),
            "pct_from_52w_low": round((current_price - wk52_low) / wk52_low * 100, 2),
        }

    # ────────────────────────────────────────────────────────────────────────
    # Weekly timeframe signal (spawn minimal weekly fetch)
    # ────────────────────────────────────────────────────────────────────────

    async def _weekly_timeframe(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch weekly bars and return a simple trend signal.
        Used for higher-timeframe confluence.
        """
        try:
            df_w = await self.tools.get_ohlcv(symbol, days=730, interval="1wk")
            if df_w is None or len(df_w) < 10:
                return {
                    "signal": "neutral",
                    "interpretation": "Weekly data unavailable.",
                }

            df_w = self._normalise_df(df_w)
            closes_w = df_w["close"]
            ema20_w = float(closes_w.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50_w = (
                float(closes_w.ewm(span=50, adjust=False).mean().iloc[-1])
                if len(closes_w) >= 50
                else None
            )
            price_w = float(closes_w.iloc[-1])
            rsi_w = float(self._rsi(closes_w, 14).iloc[-1])

            if (
                price_w > ema20_w
                and (ema50_w is None or ema20_w > ema50_w)
                and rsi_w > 50
            ):
                sig = "bullish"
                interp = f"Weekly: price above EMA20 (₹{ema20_w:.0f}), RSI {rsi_w:.0f} > 50 — bullish higher-timeframe."
            elif price_w < ema20_w and rsi_w < 50:
                sig = "bearish"
                interp = f"Weekly: price below EMA20 (₹{ema20_w:.0f}), RSI {rsi_w:.0f} < 50 — bearish higher-timeframe."
            else:
                sig = "neutral"
                interp = f"Weekly: mixed signals — price ₹{price_w:.0f} vs EMA20 ₹{ema20_w:.0f}, RSI {rsi_w:.0f}."

            return {
                "signal": sig,
                "price": round(price_w, 2),
                "ema20_weekly": round(ema20_w, 2),
                "rsi_weekly": round(rsi_w, 2),
                "interpretation": interp,
            }
        except Exception as exc:
            self._log.debug(f"Weekly TF error: {exc}")
            return {
                "signal": "neutral",
                "interpretation": "Weekly analysis unavailable.",
            }

    # ────────────────────────────────────────────────────────────────────────
    # Signal voting + scoring
    # ────────────────────────────────────────────────────────────────────────

    def _collect_votes(
        self,
        indicators: Dict[str, Any],
        patterns: Dict[str, Any],
        weekly_signal: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Gather directional signal (bullish/bearish/neutral) from each indicator.
        Returns a dict of {indicator_name: signal}.
        """
        votes: Dict[str, str] = {}

        signal_keys = [
            ("rsi", "signal"),
            ("macd", "signal_direction"),
            ("bollinger_bands", "signal"),
            ("stochastic", "signal"),
            ("adx", "signal"),
            ("ema", "signal"),
            ("sma", "signal"),
            ("supertrend", "signal"),
            ("ichimoku", "signal"),
            ("obv", "signal"),
            ("mfi", "signal"),
            ("cci", "signal"),
            ("vwap", "signal"),
            ("williams_r", "signal"),
            ("pivot_points", "signal"),
        ]

        for key, sig_field in signal_keys:
            if key in indicators:
                votes[key] = indicators[key].get(sig_field, "neutral")

        # Pattern signal
        if patterns:
            summary = patterns.get("summary", {})
            votes["pattern"] = summary.get("signal", "neutral")

        # Weekly higher-timeframe
        votes["weekly_htf"] = weekly_signal.get("signal", "neutral")

        return votes

    def _score_votes(self, votes: Dict[str, str]) -> tuple:
        """
        Convert votes → weighted score → trading signal.

        Returns:
            (trading_signal str, score float [-100, +100], confidence float [0, 1])
        """
        score = 0.0
        total_weight = 0.0

        for indicator, signal in votes.items():
            weight = self._WEIGHTS.get(indicator, 1.0)
            total_weight += weight
            if signal == "bullish":
                score += weight
            elif signal == "bearish":
                score -= weight

        # Normalise to -100 / +100
        if total_weight > 0:
            normalised = (score / total_weight) * 100
        else:
            normalised = 0.0

        # Map to human label
        if normalised >= 60:
            trading_signal = "STRONG BUY"
        elif normalised >= 25:
            trading_signal = "BUY"
        elif normalised <= -60:
            trading_signal = "STRONG SELL"
        elif normalised <= -25:
            trading_signal = "SELL"
        else:
            trading_signal = "NEUTRAL"

        # Confidence = abs normalised / 100 (range 0–1)
        confidence = min(1.0, abs(normalised) / 80.0)

        return trading_signal, normalised, confidence

    # ────────────────────────────────────────────────────────────────────────
    # Human-readable summary
    # ────────────────────────────────────────────────────────────────────────

    def _build_summary(
        self,
        symbol: str,
        price: float,
        trading_signal: str,
        score: float,
        confidence: float,
        indicators: Dict[str, Any],
        patterns: Dict[str, Any],
        sr: Dict[str, Any],
    ) -> str:
        emoji_map = {
            "STRONG BUY": "🟢🚀",
            "BUY": "🟢",
            "NEUTRAL": "🟡",
            "SELL": "🔴",
            "STRONG SELL": "🔴💀",
        }
        emoji = emoji_map.get(trading_signal, "⚪")

        rsi_val = indicators.get("rsi", {}).get("value_14", "—")
        macd_sig = indicators.get("macd", {}).get("signal_direction", "—")
        sup_trend = indicators.get("supertrend", {}).get("signal", "—")
        top_pattern = (
            patterns.get("summary", {}).get("top_pattern") if patterns else None
        )

        imm_res = sr.get("immediate_resistance")
        imm_sup = sr.get("immediate_support")

        lines = [
            f"{emoji} **{symbol} — {trading_signal}** (Score: {score:+.1f}/100 | Confidence: {confidence * 100:.0f}%)",
            f"  • Price: ₹{price:,.2f}",
            f"  • RSI(14): {rsi_val} | MACD: {macd_sig} | Supertrend: {sup_trend}",
        ]
        if top_pattern:
            lines.append(f"  • Top Pattern: {top_pattern}")
        if imm_res:
            lines.append(f"  • Key Resistance: ₹{imm_res:,.2f}")
        if imm_sup:
            lines.append(f"  • Key Support: ₹{imm_sup:,.2f}")

        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────────────────
    # Math helpers (pure pandas/numpy — no external TA-lib required)
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                df[col] = float("nan")
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df

    @staticmethod
    def _rsi(closes: pd.Series, period: int = 14) -> pd.Series:
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(
        highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14
    ) -> float:
        tr = pd.concat(
            [
                highs - lows,
                (highs - closes.shift(1)).abs(),
                (lows - closes.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    @staticmethod
    def _adx(
        highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14
    ) -> tuple:
        """Returns (ADX, +DI, -DI)."""
        tr = pd.concat(
            [
                highs - lows,
                (highs - closes.shift(1)).abs(),
                (lows - closes.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        dm_plus = highs.diff()
        dm_minus = -lows.diff()
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)

        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        di_pos = 100 * dm_plus.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-9)
        di_neg = (
            100 * dm_minus.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-9)
        )
        dx = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-9)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        return (
            float(adx.iloc[-1]),
            float(di_pos.iloc[-1]),
            float(di_neg.iloc[-1]),
        )

    @staticmethod
    def _supertrend(
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> tuple:
        """
        Returns (signal: 'bullish'|'bearish', supertrend_series).
        """
        atr_series = (
            pd.concat(
                [
                    highs - lows,
                    (highs - closes.shift(1)).abs(),
                    (lows - closes.shift(1)).abs(),
                ],
                axis=1,
            )
            .max(axis=1)
            .rolling(period)
            .mean()
        )

        hl2 = (highs + lows) / 2
        upper_band = hl2 + multiplier * atr_series
        lower_band = hl2 - multiplier * atr_series

        supertrend = pd.Series(index=closes.index, dtype=float)
        in_uptrend = pd.Series(index=closes.index, dtype=bool)

        for i in range(period, len(closes)):
            prev_upper = float(upper_band.iloc[i - 1])
            prev_lower = float(lower_band.iloc[i - 1])
            curr_upper = float(upper_band.iloc[i])
            curr_lower = float(lower_band.iloc[i])
            close_prev = float(closes.iloc[i - 1])
            close_curr = float(closes.iloc[i])

            # Adjust bands
            if curr_upper > prev_upper or close_prev < prev_upper:
                curr_upper = curr_upper
            else:
                curr_upper = prev_upper
            upper_band.iloc[i] = curr_upper

            if curr_lower < prev_lower or close_prev > prev_lower:
                curr_lower = curr_lower
            else:
                curr_lower = prev_lower
            lower_band.iloc[i] = curr_lower

            # Trend direction
            prev_up = bool(in_uptrend.iloc[i - 1]) if i > period else True
            if prev_up:
                if close_curr < curr_lower:
                    in_uptrend.iloc[i] = False
                    supertrend.iloc[i] = curr_upper
                else:
                    in_uptrend.iloc[i] = True
                    supertrend.iloc[i] = curr_lower
            else:
                if close_curr > curr_upper:
                    in_uptrend.iloc[i] = True
                    supertrend.iloc[i] = curr_lower
                else:
                    in_uptrend.iloc[i] = False
                    supertrend.iloc[i] = curr_upper

        signal = "bullish" if bool(in_uptrend.iloc[-1]) else "bearish"
        return signal, supertrend

    @staticmethod
    def _ichimoku(
        highs: pd.Series, lows: pd.Series, closes: pd.Series
    ) -> Dict[str, Any]:
        """
        Ichimoku Cloud components + signal.
        """

        def midrange(h: pd.Series, l: pd.Series, n: int) -> pd.Series:
            return (h.rolling(n).max() + l.rolling(n).min()) / 2

        tenkan = midrange(highs, lows, 9)  # Conversion line
        kijun = midrange(highs, lows, 26)  # Base line
        span_a = ((tenkan + kijun) / 2).shift(26)
        span_b = midrange(highs, lows, 52).shift(26)
        chikou = closes.shift(-26)

        price = float(closes.iloc[-1])
        tenkan_v = float(tenkan.iloc[-1])
        kijun_v = float(kijun.iloc[-1])
        span_a_v = float(span_a.iloc[-1]) if not pd.isna(span_a.iloc[-1]) else None
        span_b_v = float(span_b.iloc[-1]) if not pd.isna(span_b.iloc[-1]) else None

        # Cloud signal
        cloud_top = max(span_a_v, span_b_v) if span_a_v and span_b_v else None
        cloud_bottom = min(span_a_v, span_b_v) if span_a_v and span_b_v else None

        if cloud_top and price > cloud_top:
            ichi_signal = "bullish"
            interp = f"Price ₹{price:.0f} above cloud (₹{cloud_bottom:.0f}–₹{cloud_top:.0f}) — bullish."
        elif cloud_bottom and price < cloud_bottom:
            ichi_signal = "bearish"
            interp = f"Price ₹{price:.0f} below cloud — bearish."
        elif cloud_top and cloud_bottom:
            ichi_signal = "neutral"
            interp = f"Price inside cloud — indecision / transition."
        else:
            ichi_signal = "neutral"
            interp = "Ichimoku cloud building (not enough data for Senkou span)."

        # Tenkan / Kijun cross
        tk_cross = "bullish" if tenkan_v > kijun_v else "bearish"

        return {
            "tenkan_sen": round(tenkan_v, 2),
            "kijun_sen": round(kijun_v, 2),
            "senkou_a": round(span_a_v, 2) if span_a_v else None,
            "senkou_b": round(span_b_v, 2) if span_b_v else None,
            "cloud_top": round(cloud_top, 2) if cloud_top else None,
            "cloud_bottom": round(cloud_bottom, 2) if cloud_bottom else None,
            "tk_cross": tk_cross,
            "signal": ichi_signal,
            "interpretation": interp,
        }
