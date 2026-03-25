"""
Technical Analysis Engine — DaddysAI Analytics Module
======================================================
Phase 2 · Feature 2 · Full TA Engine

Provides `calculate_all_indicators(df)` which accepts a standard OHLCV DataFrame
and returns a rich dictionary containing every major technical indicator.

Expected DataFrame columns (lowercase): open, high, low, close, volume
Index: any (integer or DatetimeIndex both work)

Every indicator section returns at minimum:
  value          – latest scalar or sub-dict of scalars
  signal         – 'bullish' | 'bearish' | 'neutral'
  interpretation – human-readable one-liner
  confidence     – 0.0–1.0

Requires: pandas, numpy, pandas-ta (0.4.71b0)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_float(val: Any, decimals: int = 4) -> Optional[float]:
    """Convert anything to a rounded float; return None on NaN / Inf / error."""
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, decimals)
    except Exception:
        return None


def _last(series: pd.Series, decimals: int = 4) -> Optional[float]:
    """Return the last non-NaN value of a Series as a rounded float."""
    try:
        s = series.dropna()
        if s.empty:
            return None
        return _safe_float(s.iloc[-1], decimals)
    except Exception:
        return None


def _prev(series: pd.Series, n: int = 1) -> Optional[float]:
    """Return the n-th last non-NaN value."""
    try:
        s = series.dropna()
        if len(s) <= n:
            return None
        return _safe_float(s.iloc[-(n + 1)])
    except Exception:
        return None


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, coerce numerics, drop rows with no close price."""
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _try_ta(fn, *args, **kwargs) -> Optional[Any]:
    """Execute a pandas-ta call; return None on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.debug(
            "pandas-ta call failed (%s): %s",
            fn.__name__ if hasattr(fn, "__name__") else str(fn),
            exc,
        )
        return None


def _ta_col(df_result: pd.DataFrame, prefix: str) -> Optional[str]:
    """Find first column whose name starts with *prefix* (case-sensitive)."""
    if df_result is None:
        return None
    for c in df_result.columns:
        if c.startswith(prefix):
            return c
    return None


def _insufficient() -> Dict[str, Any]:
    return {
        "value": None,
        "signal": "neutral",
        "interpretation": "Insufficient data",
        "confidence": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TechnicalEngine class
# ─────────────────────────────────────────────────────────────────────────────


class TechnicalEngine:
    """
    Stateful engine wrapping a normalised OHLCV DataFrame.

    Usage
    -----
    >>> engine = TechnicalEngine(df)
    >>> result = engine.calculate_all()
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = _normalise_df(df)
        self._ta_available = False
        self._init_ta()

    # ── setup ─────────────────────────────────────────────────────────────────

    def _init_ta(self) -> None:
        try:
            import pandas_ta as ta  # noqa: F401

            self.df.ta.cores = 0  # Disable multiprocessing – keep it deterministic
            self._ta_available = True
            logger.debug("pandas-ta loaded successfully")
        except Exception as exc:
            logger.warning(
                "pandas-ta not available, using manual calculations: %s", exc
            )

    # ─────────────────────────────────────────────────────────────────────────
    # RSI
    # ─────────────────────────────────────────────────────────────────────────

    def rsi(self, length: int = 14) -> Dict[str, Any]:
        """Relative Strength Index."""
        if self._ta_available:
            s = _try_ta(self.df.ta.rsi, length=length)
        else:
            s = None
        if s is None:
            s = self._calc_rsi(self.df["close"], length)

        val = _last(s)

        if val is None:
            return _insufficient()

        if val >= 80:
            sig, interp, conf = "bearish", f"RSI {val:.1f} — extremely overbought", 0.90
        elif val >= 70:
            sig, interp, conf = (
                "bearish",
                f"RSI {val:.1f} — overbought; potential reversal",
                0.75,
            )
        elif val <= 20:
            sig, interp, conf = "bullish", f"RSI {val:.1f} — extremely oversold", 0.90
        elif val <= 30:
            sig, interp, conf = (
                "bullish",
                f"RSI {val:.1f} — oversold; potential bounce",
                0.75,
            )
        elif val >= 60:
            sig, interp, conf = "bullish", f"RSI {val:.1f} — bullish momentum", 0.55
        elif val <= 40:
            sig, interp, conf = "bearish", f"RSI {val:.1f} — bearish momentum", 0.55
        else:
            sig, interp, conf = "neutral", f"RSI {val:.1f} — neutral zone (40–60)", 0.35

        return {
            "value": val,
            "signal": sig,
            "interpretation": interp,
            "confidence": conf,
        }

    @staticmethod
    def _calc_rsi(close: pd.Series, length: int = 14) -> pd.Series:
        delta = close.diff()
        gain = (
            delta.clip(lower=0)
            .ewm(com=length - 1, min_periods=length, adjust=False)
            .mean()
        )
        loss = (
            (-delta.clip(upper=0))
            .ewm(com=length - 1, min_periods=length, adjust=False)
            .mean()
        )
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # ─────────────────────────────────────────────────────────────────────────
    # MACD
    # ─────────────────────────────────────────────────────────────────────────

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Any]:
        """MACD line, signal line, and histogram."""
        if self._ta_available:
            result = _try_ta(self.df.ta.macd, fast=fast, slow=slow, signal=signal)
        else:
            result = None

        if result is not None and not result.empty:
            macd_col = _ta_col(result, f"MACD_{fast}_{slow}_{signal}")
            sig_col = _ta_col(result, f"MACDs_{fast}_{slow}_{signal}")
            hist_col = _ta_col(result, f"MACDh_{fast}_{slow}_{signal}")
            # Fallback to positional columns
            cols = result.columns.tolist()
            macd_val = _last(result[macd_col]) if macd_col else _last(result[cols[0]])
            sig_val = (
                _last(result[sig_col])
                if sig_col
                else (_last(result[cols[1]]) if len(cols) > 1 else None)
            )
            hist_val = (
                _last(result[hist_col])
                if hist_col
                else (_last(result[cols[2]]) if len(cols) > 2 else None)
            )
            # Prev histogram for divergence direction
            prev_hist = _prev(result[hist_col]) if hist_col else None
        else:
            close = self.df["close"]
            ema_f = close.ewm(span=fast, adjust=False).mean()
            ema_s = close.ewm(span=slow, adjust=False).mean()
            macd_s = ema_f - ema_s
            sig_s = macd_s.ewm(span=signal, adjust=False).mean()
            hist_s = macd_s - sig_s
            macd_val, sig_val, hist_val = _last(macd_s), _last(sig_s), _last(hist_s)
            prev_hist = _prev(hist_s)

        if macd_val is None or sig_val is None:
            return {
                "macd": None,
                "signal_line": None,
                "histogram": None,
                "signal": "neutral",
                "interpretation": "Insufficient data",
                "confidence": 0.0,
            }

        hist_increasing = (
            hist_val is not None and prev_hist is not None and hist_val > prev_hist
        )

        if macd_val > sig_val and macd_val > 0:
            sig_str, interp, conf = (
                "bullish",
                f"MACD {macd_val:.3f} above signal & zero — strong bullish",
                0.80,
            )
        elif macd_val > sig_val:
            sig_str, interp, conf = (
                "bullish",
                f"MACD {macd_val:.3f} crossed above signal {sig_val:.3f}",
                0.65,
            )
        elif macd_val < sig_val and macd_val < 0:
            sig_str, interp, conf = (
                "bearish",
                f"MACD {macd_val:.3f} below signal & zero — strong bearish",
                0.80,
            )
        elif macd_val < sig_val:
            sig_str, interp, conf = (
                "bearish",
                f"MACD {macd_val:.3f} crossed below signal {sig_val:.3f}",
                0.65,
            )
        else:
            sig_str, interp, conf = (
                "neutral",
                "MACD at signal line — no clear direction",
                0.30,
            )

        return {
            "macd": macd_val,
            "signal_line": sig_val,
            "histogram": hist_val,
            "histogram_increasing": hist_increasing,
            "signal": sig_str,
            "interpretation": interp,
            "confidence": conf,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Bollinger Bands
    # ─────────────────────────────────────────────────────────────────────────

    def bollinger_bands(self, length: int = 20, std: float = 2.0) -> Dict[str, Any]:
        """Upper, middle, lower bands with %B and bandwidth."""
        if self._ta_available:
            result = _try_ta(self.df.ta.bbands, length=length, std=std)
        else:
            result = None

        close = self.df["close"]
        price = _last(close)

        if result is not None and not result.empty:
            cols = result.columns.tolist()

            def _get(pref):
                return (
                    _last(result[[c for c in cols if c.startswith(pref)][0]])
                    if any(c.startswith(pref) for c in cols)
                    else None
                )

            lower = _get("BBL_")
            mid = _get("BBM_")
            upper = _get("BBU_")
            pct_b = _get("BBP_")
            bw = _get("BBB_")
        else:
            mid_s = close.rolling(length).mean()
            std_s = close.rolling(length).std(ddof=0)
            upper_s = mid_s + std * std_s
            lower_s = mid_s - std * std_s
            upper, mid, lower = _last(upper_s), _last(mid_s), _last(lower_s)
            pct_b = (
                _safe_float(((price - lower) / (upper - lower)), 4)
                if (price and upper and lower and upper != lower)
                else None
            )
            bw = _safe_float(((upper - lower) / mid * 100), 4) if mid else None

        if price is None or upper is None or lower is None:
            return {
                **_insufficient(),
                "upper": None,
                "middle": None,
                "lower": None,
                "pct_b": None,
                "bandwidth": None,
            }

        if price >= upper:
            sig, interp, conf = (
                "bearish",
                f"Price {price:.2f} at/above upper BB {upper:.2f} — overbought",
                0.70,
            )
        elif price <= lower:
            sig, interp, conf = (
                "bullish",
                f"Price {price:.2f} at/below lower BB {lower:.2f} — oversold",
                0.70,
            )
        elif mid and price > mid:
            sig, interp, conf = (
                "bullish",
                f"Price {price:.2f} in upper BB half (above {mid:.2f})",
                0.45,
            )
        else:
            sig, interp, conf = (
                "bearish",
                f"Price {price:.2f} in lower BB half (below {mid:.2f})",
                0.45,
            )

        return {
            "upper": upper,
            "middle": mid,
            "lower": lower,
            "pct_b": pct_b,
            "bandwidth": bw,
            "signal": sig,
            "interpretation": interp,
            "confidence": conf,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Stochastic Oscillator
    # ─────────────────────────────────────────────────────────────────────────

    def stochastic(self, k: int = 14, d: int = 3, smooth_k: int = 3) -> Dict[str, Any]:
        """Stochastic %K and %D."""
        if self._ta_available:
            result = _try_ta(self.df.ta.stoch, k=k, d=d, smooth_k=smooth_k)
        else:
            result = None

        if result is not None and not result.empty:
            cols = result.columns.tolist()
            k_col = next((c for c in cols if "STOCHk_" in c), cols[0])
            d_col = next(
                (c for c in cols if "STOCHd_" in c),
                cols[1] if len(cols) > 1 else cols[0],
            )
            k_val, d_val = _last(result[k_col]), _last(result[d_col])
        else:
            low_min = self.df["low"].rolling(k).min()
            high_max = self.df["high"].rolling(k).max()
            raw_k = (
                100
                * (self.df["close"] - low_min)
                / (high_max - low_min).replace(0, np.nan)
            )
            k_s = raw_k.rolling(smooth_k).mean()
            d_s = k_s.rolling(d).mean()
            k_val, d_val = _last(k_s), _last(d_s)

        if k_val is None:
            return {**_insufficient(), "k": None, "d": None}

        if k_val > 80 and d_val and k_val < d_val:
            sig, interp, conf = (
                "bearish",
                f"Stoch %K {k_val:.1f} overbought + bearish cross",
                0.80,
            )
        elif k_val > 80:
            sig, interp, conf = "bearish", f"Stoch %K {k_val:.1f} — overbought", 0.65
        elif k_val < 20 and d_val and k_val > d_val:
            sig, interp, conf = (
                "bullish",
                f"Stoch %K {k_val:.1f} oversold + bullish cross",
                0.80,
            )
        elif k_val < 20:
            sig, interp, conf = "bullish", f"Stoch %K {k_val:.1f} — oversold", 0.65
        elif d_val and k_val > d_val:
            sig, interp, conf = (
                "bullish",
                f"%K {k_val:.1f} above %D {d_val:.1f} — bullish",
                0.50,
            )
        elif d_val and k_val < d_val:
            sig, interp, conf = (
                "bearish",
                f"%K {k_val:.1f} below %D {d_val:.1f} — bearish",
                0.50,
            )
        else:
            sig, interp, conf = "neutral", f"Stoch %K {k_val:.1f} — neutral", 0.30

        return {
            "k": k_val,
            "d": d_val,
            "signal": sig,
            "interpretation": interp,
            "confidence": conf,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # ATR
    # ─────────────────────────────────────────────────────────────────────────

    def atr(self, length: int = 14) -> Dict[str, Any]:
        """Average True Range."""
        if self._ta_available:
            s = _try_ta(self.df.ta.atr, length=length)
        else:
            s = None
        if s is None:
            s = self._calc_atr(self.df, length)

        val = _last(s)
        price = _last(self.df["close"])
        pct = _safe_float(val / price * 100, 2) if (val and price) else None

        if val is None:
            return {**_insufficient(), "pct_of_price": None}

        if pct and pct > 3.0:
            sig, interp = (
                "neutral",
                f"ATR {val:.2f} ({pct:.1f}% of price) — high volatility",
            )
        elif pct and pct > 1.5:
            sig, interp = (
                "neutral",
                f"ATR {val:.2f} ({pct:.1f}% of price) — moderate volatility",
            )
        else:
            sig, interp = (
                "neutral",
                f"ATR {val:.2f} ({pct:.1f}% of price) — low volatility",
            )

        return {
            "value": val,
            "pct_of_price": pct,
            "signal": sig,
            "interpretation": interp,
            "confidence": 0.5,
        }

    @staticmethod
    def _calc_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(com=length - 1, min_periods=length, adjust=False).mean()

    # ─────────────────────────────────────────────────────────────────────────
    # ADX
    # ─────────────────────────────────────────────────────────────────────────

    def adx(self, length: int = 14) -> Dict[str, Any]:
        """Average Directional Index with +DI and -DI."""
        if self._ta_available:
            result = _try_ta(self.df.ta.adx, length=length)
        else:
            result = None

        if result is not None and not result.empty:
            cols = result.columns.tolist()
            adx_col = next((c for c in cols if c.startswith("ADX_")), cols[0])
            dmp_col = next((c for c in cols if "DMP_" in c), None)
            dmn_col = next((c for c in cols if "DMN_" in c), None)
            adx_val = _last(result[adx_col])
            dmp_val = _last(result[dmp_col]) if dmp_col else None
            dmn_val = _last(result[dmn_col]) if dmn_col else None
        else:
            adx_val, dmp_val, dmn_val = self._calc_adx(self.df, length)

        if adx_val is None:
            return {**_insufficient(), "plus_di": None, "minus_di": None}

        trending = adx_val >= 25
        if trending and dmp_val and dmn_val:
            if dmp_val > dmn_val:
                sig, interp, conf = (
                    "bullish",
                    f"ADX {adx_val:.1f} — strong uptrend (+DI {dmp_val:.1f} > -DI {dmn_val:.1f})",
                    0.80,
                )
            else:
                sig, interp, conf = (
                    "bearish",
                    f"ADX {adx_val:.1f} — strong downtrend (-DI {dmn_val:.1f} > +DI {dmp_val:.1f})",
                    0.80,
                )
        elif trending:
            sig, interp, conf = (
                "neutral",
                f"ADX {adx_val:.1f} — strong trend (direction unclear)",
                0.50,
            )
        else:
            sig, interp, conf = (
                "neutral",
                f"ADX {adx_val:.1f} — weak/no trend (< 25)",
                0.25,
            )

        return {
            "adx": adx_val,
            "plus_di": dmp_val,
            "minus_di": dmn_val,
            "signal": sig,
            "interpretation": interp,
            "confidence": conf,
        }

    @staticmethod
    def _calc_adx(df: pd.DataFrame, length: int = 14) -> Tuple:
        high, low, close = df["high"], df["low"], df["close"]
        up = high.diff()
        dn = -low.diff()
        plus_dm = up.where((up > dn) & (up > 0), 0.0)
        minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(com=length - 1, adjust=False).mean()
        plus_di = (
            100
            * plus_dm.ewm(com=length - 1, adjust=False).mean()
            / atr.replace(0, np.nan)
        )
        minus_di = (
            100
            * minus_dm.ewm(com=length - 1, adjust=False).mean()
            / atr.replace(0, np.nan)
        )
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(com=length - 1, adjust=False).mean()
        return _last(adx), _last(plus_di), _last(minus_di)

    # ─────────────────────────────────────────────────────────────────────────
    # CCI
    # ─────────────────────────────────────────────────────────────────────────

    def cci(self, length: int = 20) -> Dict[str, Any]:
        """Commodity Channel Index."""
        if self._ta_available:
            s = _try_ta(self.df.ta.cci, length=length)
        else:
            s = None
        if s is None:
            tp = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
            sma = tp.rolling(length).mean()
            mad = tp.rolling(length).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True
            )
            s = (tp - sma) / (0.015 * mad.replace(0, np.nan))

        val = _last(s)
        if val is None:
            return _insufficient()

        if val > 200:
            sig, interp, conf = (
                "bearish",
                f"CCI {val:.1f} — extreme overbought (>200)",
                0.85,
            )
        elif val > 100:
            sig, interp, conf = "bearish", f"CCI {val:.1f} — overbought (>100)", 0.65
        elif val < -200:
            sig, interp, conf = (
                "bullish",
                f"CCI {val:.1f} — extreme oversold (<-200)",
                0.85,
            )
        elif val < -100:
            sig, interp, conf = "bullish", f"CCI {val:.1f} — oversold (<-100)", 0.65
        elif val > 0:
            sig, interp, conf = (
                "bullish",
                f"CCI {val:.1f} — moderate bullish momentum",
                0.40,
            )
        else:
            sig, interp, conf = (
                "bearish",
                f"CCI {val:.1f} — moderate bearish momentum",
                0.40,
            )

        return {
            "value": val,
            "signal": sig,
            "interpretation": interp,
            "confidence": conf,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # VWAP
    # ─────────────────────────────────────────────────────────────────────────

    def vwap(self) -> Dict[str, Any]:
        """Volume Weighted Average Price."""
        if "volume" not in self.df.columns:
            return {**_insufficient(), "interpretation": "No volume data available"}

        if self._ta_available:
            s = _try_ta(self.df.ta.vwap)
        else:
            s = None
        if s is None:
            tp = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
            vol = self.df["volume"].replace(0, np.nan)
            s = (tp * vol).cumsum() / vol.cumsum()

        val = _last(s)
        price = _last(self.df["close"])

        if val is None or price is None:
            return _insufficient()

        deviation_pct = _safe_float((price - val) / val * 100, 2)

        if price > val * 1.02:
            sig, interp, conf = (
                "bullish",
                f"Price {price:.2f} is {deviation_pct:.1f}% above VWAP {val:.2f} — strong buying",
                0.70,
            )
        elif price > val:
            sig, interp, conf = (
                "bullish",
                f"Price {price:.2f} above VWAP {val:.2f} — buyers in control",
                0.55,
            )
        elif price < val * 0.98:
            sig, interp, conf = (
                "bearish",
                f"Price {price:.2f} is {abs(deviation_pct):.1f}% below VWAP {val:.2f} — strong selling",
                0.70,
            )
        else:
            sig, interp, conf = (
                "bearish",
                f"Price {price:.2f} below VWAP {val:.2f} — sellers in control",
                0.55,
            )

        return {
            "value": val,
            "current_price": price,
            "deviation_pct": deviation_pct,
            "signal": sig,
            "interpretation": interp,
            "confidence": conf,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Ichimoku Cloud
    # ─────────────────────────────────────────────────────────────────────────

    def ichimoku(
        self, tenkan: int = 9, kijun: int = 26, senkou: int = 52
    ) -> Dict[str, Any]:
        """Ichimoku Kinko Hyo."""
        try:
            high, low, close = self.df["high"], self.df["low"], self.df["close"]

            def _midrange(h: pd.Series, l: pd.Series, n: int) -> pd.Series:
                return (h.rolling(n).max() + l.rolling(n).min()) / 2

            tenkan_s = _midrange(high, low, tenkan)
            kijun_s = _midrange(high, low, kijun)
            senkou_a = ((tenkan_s + kijun_s) / 2).shift(kijun)
            senkou_b = _midrange(high, low, senkou).shift(kijun)
            chikou = close.shift(-kijun)

            ts, ks = _last(tenkan_s), _last(kijun_s)
            sa, sb = _last(senkou_a), _last(senkou_b)
            ck = _last(chikou)
            price = _last(close)

            cloud_top = max(sa, sb) if (sa is not None and sb is not None) else None
            cloud_bottom = min(sa, sb) if (sa is not None and sb is not None) else None

            # Bullish signals: price above cloud, TK cross (tenkan > kijun), chikou above price
            bullish_count = 0
            bearish_count = 0

            if price is not None and cloud_top is not None:
                if price > cloud_top:
                    bullish_count += 2
                    tk_loc = "above"
                elif price < cloud_bottom:
                    bearish_count += 2
                    tk_loc = "below"
                else:
                    tk_loc = "inside"
            else:
                tk_loc = "unknown"

            if ts is not None and ks is not None:
                if ts > ks:
                    bullish_count += 1
                else:
                    bearish_count += 1

            if sa is not None and sb is not None:
                if sa > sb:
                    bullish_count += 1  # Bullish cloud (green)
                else:
                    bearish_count += 1  # Bearish cloud (red)

            total = bullish_count + bearish_count
            if total == 0:
                sig, interp, conf = "neutral", "Insufficient data for Ichimoku", 0.0
            elif bullish_count > bearish_count:
                conf = bullish_count / (total + 2)
                sig, interp = (
                    "bullish",
                    f"Price {tk_loc} cloud — Ichimoku bullish ({bullish_count}/{total} signals)",
                )
            elif bearish_count > bullish_count:
                conf = bearish_count / (total + 2)
                sig, interp = (
                    "bearish",
                    f"Price {tk_loc} cloud — Ichimoku bearish ({bearish_count}/{total} signals)",
                )
            else:
                sig, interp, conf = (
                    "neutral",
                    f"Mixed Ichimoku signals — price {tk_loc} cloud",
                    0.35,
                )

            return {
                "tenkan_sen": ts,
                "kijun_sen": ks,
                "senkou_a": sa,
                "senkou_b": sb,
                "chikou_span": ck,
                "cloud_top": cloud_top,
                "cloud_bottom": cloud_bottom,
                "signal": sig,
                "confidence": round(conf, 3),
                "interpretation": interp,
            }
        except Exception as exc:
            return {"error": str(exc), "signal": "neutral", "confidence": 0.0}

    # ── public interface ──────────────────────────────────────────────────

    def calculate_all(self) -> dict:
        """
        Run all indicators on self.df and return a flat dict.
        Calls each indicator method and merges results.
        """
        results: dict = {}

        for method_name in [
            "rsi",
            "macd",
            "bollinger_bands",
            "stochastic",
            "atr",
            "adx",
            "cci",
            "vwap",
            "ichimoku",
        ]:
            try:
                method = getattr(self, method_name, None)
                if method is not None:
                    result = method()
                    if result:
                        results[method_name] = result
            except Exception as exc:
                results[method_name] = {"error": str(exc)}

        # Current price
        try:
            results["current_price"] = float(self.df["close"].iloc[-1])
        except Exception:
            pass

        return results


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def calculate_all_indicators(df: "pd.DataFrame") -> dict:
    """
    Convenience wrapper — runs all technical indicators on an OHLCV DataFrame.

    Args:
        df: DataFrame with lowercase columns: open, high, low, close, volume
            and a DatetimeIndex (or integer index).

    Returns:
        Rich dict with all indicator results, signals, and interpretations.
        Structure::

            {
              "current_price": 22400.5,
              "rsi": {"value_14": 58.3, "signal": "bullish", ...},
              "macd": {"macd": 45.2, "signal": "bullish", ...},
              "bollinger_bands": {"upper": ..., "signal": "neutral", ...},
              "stochastic": {"k": 67.4, "d": 63.1, "signal": "bullish", ...},
              "atr":  {"value": 180.2, "atr_pct": 0.81, ...},
              "adx":  {"adx": 28.5, "trend_strength": "Trending", ...},
              "cci":  {"value": 85.3, "signal": "bullish", ...},
              "vwap": {"value": 22310.0, "signal": "bullish", ...},
              "ichimoku": {"tenkan_sen": ..., "signal": "bullish", ...},
            }
    """
    try:
        engine = TechnicalEngine(df)
        return engine.calculate_all()
    except Exception as exc:
        return {"error": str(exc)}
