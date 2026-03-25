"""
PatternDetector — Candlestick & Chart Pattern Recognition Engine.

Detects:
  Candlestick patterns  — Doji, Hammer, Engulfing, Morning/Evening Star,
                          Harami, Three Soldiers/Crows, Marubozu, etc.
  Chart patterns        — Head & Shoulders, Double Top/Bottom, Triangle,
                          Flag, Wedge, Cup & Handle (swing-based detection)

Every detected pattern carries:
  name        — canonical pattern name
  signal      — 'bullish' | 'bearish' | 'neutral'
  confidence  — 0.0–1.0  (quality of the setup)
  bar_index   — row index (int) where the pattern was confirmed
  description — human-readable explanation

Usage::

    detector = PatternDetector()
    result = detector.detect_all(df)
    # result['candlestick'] → list of recent candlestick patterns
    # result['chart']       → list of detected chart patterns
    # result['summary']     → aggregated signal + confidence

    # convenience wrapper
    result = detect_patterns(df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Pattern:
    name: str
    signal: str  # 'bullish' | 'bearish' | 'neutral'
    confidence: float  # 0.0 – 1.0
    bar_index: int  # position in the DataFrame
    description: str = ""
    pattern_type: str = "candlestick"  # 'candlestick' | 'chart'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "signal": self.signal,
            "confidence": round(self.confidence, 3),
            "bar_index": self.bar_index,
            "description": self.description,
            "type": self.pattern_type,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(o: float, c: float) -> float:
    return abs(c - o)


def _upper_shadow(o: float, h: float, c: float) -> float:
    return h - max(o, c)


def _lower_shadow(o: float, l: float, c: float) -> float:
    return min(o, c) - l


def _total_range(h: float, l: float) -> float:
    return max(h - l, 1e-9)  # avoid division by zero


def _is_bullish_bar(o: float, c: float) -> bool:
    return c > o


def _is_bearish_bar(o: float, c: float) -> bool:
    return c < o


def _avg_body(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Rolling average body size."""
    return (df["close"] - df["open"]).abs().rolling(window).mean()


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Candlestick pattern checkers (each returns Pattern | None)
# ---------------------------------------------------------------------------


class CandlestickPatterns:
    """
    Individual candlestick pattern detectors.
    Each method takes OHLC values (one or more bars) and returns a Pattern or None.
    """

    # ── Single-bar patterns ────────────────────────────────────────────────

    @staticmethod
    def doji(
        o: float, h: float, l: float, c: float, avg_body: float, idx: int
    ) -> Optional[Pattern]:
        """Doji: body is ≤ 10% of total range."""
        rng = _total_range(h, l)
        body_ratio = _body(o, c) / rng
        if body_ratio <= 0.10:
            conf = _clamp(1.0 - body_ratio * 5)
            return Pattern(
                name="Doji",
                signal="neutral",
                confidence=conf,
                bar_index=idx,
                description="Body ≤ 10% of range — indecision, potential reversal.",
            )
        return None

    @staticmethod
    def dragonfly_doji(
        o: float, h: float, l: float, c: float, idx: int
    ) -> Optional[Pattern]:
        """Long lower shadow, tiny body near the top."""
        rng = _total_range(h, l)
        body = _body(o, c)
        lower = _lower_shadow(o, l, c)
        upper = _upper_shadow(o, h, c)
        if body / rng <= 0.08 and lower / rng >= 0.60 and upper / rng <= 0.10:
            conf = _clamp(lower / rng)
            return Pattern(
                name="Dragonfly Doji",
                signal="bullish",
                confidence=conf,
                bar_index=idx,
                description="Long lower wick with tiny body near top — bullish reversal signal.",
            )
        return None

    @staticmethod
    def gravestone_doji(
        o: float, h: float, l: float, c: float, idx: int
    ) -> Optional[Pattern]:
        """Long upper shadow, tiny body near the bottom."""
        rng = _total_range(h, l)
        body = _body(o, c)
        upper = _upper_shadow(o, h, c)
        lower = _lower_shadow(o, l, c)
        if body / rng <= 0.08 and upper / rng >= 0.60 and lower / rng <= 0.10:
            conf = _clamp(upper / rng)
            return Pattern(
                name="Gravestone Doji",
                signal="bearish",
                confidence=conf,
                bar_index=idx,
                description="Long upper wick with tiny body near bottom — bearish reversal.",
            )
        return None

    @staticmethod
    def hammer(
        o: float, h: float, l: float, c: float, avg_body: float, idx: int
    ) -> Optional[Pattern]:
        """
        Hammer (after downtrend):
          - Lower shadow ≥ 2× body
          - Small or no upper shadow
          - Body in upper 1/3 of range
        """
        rng = _total_range(h, l)
        body = _body(o, c)
        lower = _lower_shadow(o, l, c)
        upper = _upper_shadow(o, h, c)
        if (
            body > 0
            and lower >= 2.0 * body
            and upper <= 0.3 * body
            and (max(o, c) - l) / rng >= 0.60
        ):
            conf = _clamp(lower / (2.0 * body) * 0.7 + 0.3)
            return Pattern(
                name="Hammer",
                signal="bullish",
                confidence=conf,
                bar_index=idx,
                description="Long lower shadow with small body — potential bullish reversal.",
            )
        return None

    @staticmethod
    def inverted_hammer(
        o: float, h: float, l: float, c: float, avg_body: float, idx: int
    ) -> Optional[Pattern]:
        """Upper shadow ≥ 2× body, small lower shadow."""
        rng = _total_range(h, l)
        body = _body(o, c)
        upper = _upper_shadow(o, h, c)
        lower = _lower_shadow(o, l, c)
        if body > 0 and upper >= 2.0 * body and lower <= 0.3 * body:
            conf = _clamp(upper / (2.0 * body) * 0.6 + 0.3)
            return Pattern(
                name="Inverted Hammer",
                signal="bullish",
                confidence=conf * 0.85,  # slightly less reliable
                bar_index=idx,
                description="Long upper wick, small body — possible bullish reversal (needs confirmation).",
            )
        return None

    @staticmethod
    def shooting_star(
        o: float, h: float, l: float, c: float, avg_body: float, idx: int
    ) -> Optional[Pattern]:
        """Long upper shadow, small body near bottom — bearish."""
        rng = _total_range(h, l)
        body = _body(o, c)
        upper = _upper_shadow(o, h, c)
        lower = _lower_shadow(o, l, c)
        if (
            body > 0
            and upper >= 2.0 * body
            and lower <= 0.3 * body
            and _is_bearish_bar(o, c)
        ):
            conf = _clamp(upper / rng)
            return Pattern(
                name="Shooting Star",
                signal="bearish",
                confidence=conf,
                bar_index=idx,
                description="Long upper shadow with bearish close — potential topping signal.",
            )
        return None

    @staticmethod
    def spinning_top(
        o: float, h: float, l: float, c: float, avg_body: float, idx: int
    ) -> Optional[Pattern]:
        """Small body with relatively equal upper and lower shadows."""
        rng = _total_range(h, l)
        body = _body(o, c)
        upper = _upper_shadow(o, h, c)
        lower = _lower_shadow(o, l, c)
        body_ratio = body / rng
        shadow_balance = min(upper, lower) / max(upper, lower, 1e-9)
        if 0.05 <= body_ratio <= 0.30 and shadow_balance >= 0.50:
            return Pattern(
                name="Spinning Top",
                signal="neutral",
                confidence=0.55,
                bar_index=idx,
                description="Small body with balanced shadows — indecision.",
            )
        return None

    @staticmethod
    def marubozu_bullish(
        o: float, h: float, l: float, c: float, avg_body: float, idx: int
    ) -> Optional[Pattern]:
        """Almost no shadows; strong full-body bullish candle."""
        rng = _total_range(h, l)
        body = _body(o, c)
        if (
            _is_bullish_bar(o, c)
            and body / rng >= 0.90
            and _upper_shadow(o, h, c) / rng <= 0.05
            and _lower_shadow(o, l, c) / rng <= 0.05
            and body >= avg_body * 1.2
        ):
            conf = _clamp(body / rng)
            return Pattern(
                name="Bullish Marubozu",
                signal="bullish",
                confidence=conf,
                bar_index=idx,
                description="Full bullish body with no shadows — strong buying pressure.",
            )
        return None

    @staticmethod
    def marubozu_bearish(
        o: float, h: float, l: float, c: float, avg_body: float, idx: int
    ) -> Optional[Pattern]:
        """Almost no shadows; strong full-body bearish candle."""
        rng = _total_range(h, l)
        body = _body(o, c)
        if (
            _is_bearish_bar(o, c)
            and body / rng >= 0.90
            and _upper_shadow(o, h, c) / rng <= 0.05
            and _lower_shadow(o, l, c) / rng <= 0.05
            and body >= avg_body * 1.2
        ):
            conf = _clamp(body / rng)
            return Pattern(
                name="Bearish Marubozu",
                signal="bearish",
                confidence=conf,
                bar_index=idx,
                description="Full bearish body with no shadows — strong selling pressure.",
            )
        return None

    # ── Two-bar patterns ──────────────────────────────────────────────────

    @staticmethod
    def bullish_engulfing(
        o1: float, c1: float, o2: float, c2: float, avg_body: float, idx: int
    ) -> Optional[Pattern]:
        """Bar2 (bullish) completely engulfs bar1 (bearish)."""
        if (
            _is_bearish_bar(o1, c1)
            and _is_bullish_bar(o2, c2)
            and o2 <= c1
            and c2 >= o1
            and _body(o2, c2) > _body(o1, c1)
        ):
            ratio = _body(o2, c2) / max(_body(o1, c1), 1e-9)
            conf = _clamp(0.5 + (ratio - 1.0) * 0.3)
            return Pattern(
                name="Bullish Engulfing",
                signal="bullish",
                confidence=conf,
                bar_index=idx,
                description="Bullish bar engulfs prior bearish bar — strong reversal signal.",
            )
        return None

    @staticmethod
    def bearish_engulfing(
        o1: float, c1: float, o2: float, c2: float, avg_body: float, idx: int
    ) -> Optional[Pattern]:
        """Bar2 (bearish) completely engulfs bar1 (bullish)."""
        if (
            _is_bullish_bar(o1, c1)
            and _is_bearish_bar(o2, c2)
            and o2 >= c1
            and c2 <= o1
            and _body(o2, c2) > _body(o1, c1)
        ):
            ratio = _body(o2, c2) / max(_body(o1, c1), 1e-9)
            conf = _clamp(0.5 + (ratio - 1.0) * 0.3)
            return Pattern(
                name="Bearish Engulfing",
                signal="bearish",
                confidence=conf,
                bar_index=idx,
                description="Bearish bar engulfs prior bullish bar — strong reversal signal.",
            )
        return None

    @staticmethod
    def bullish_harami(
        o1: float, c1: float, o2: float, c2: float, idx: int
    ) -> Optional[Pattern]:
        """Bar2 (small bullish) inside bar1 (large bearish)."""
        if (
            _is_bearish_bar(o1, c1)
            and _is_bullish_bar(o2, c2)
            and o2 >= c1
            and c2 <= o1
            and _body(o2, c2) <= 0.6 * _body(o1, c1)
        ):
            return Pattern(
                name="Bullish Harami",
                signal="bullish",
                confidence=0.60,
                bar_index=idx,
                description="Small bullish bar inside large bearish bar — potential reversal.",
            )
        return None

    @staticmethod
    def bearish_harami(
        o1: float, c1: float, o2: float, c2: float, idx: int
    ) -> Optional[Pattern]:
        """Bar2 (small bearish) inside bar1 (large bullish)."""
        if (
            _is_bullish_bar(o1, c1)
            and _is_bearish_bar(o2, c2)
            and o2 <= c1
            and c2 >= o1
            and _body(o2, c2) <= 0.6 * _body(o1, c1)
        ):
            return Pattern(
                name="Bearish Harami",
                signal="bearish",
                confidence=0.60,
                bar_index=idx,
                description="Small bearish bar inside large bullish bar — potential reversal.",
            )
        return None

    @staticmethod
    def piercing_line(
        o1: float, c1: float, o2: float, h2: float, l2: float, c2: float, idx: int
    ) -> Optional[Pattern]:
        """
        Piercing Line (bullish): after bearish bar, bullish bar opens below prior low
        and closes above midpoint of prior bar.
        """
        midpoint1 = (o1 + c1) / 2
        if (
            _is_bearish_bar(o1, c1)
            and _is_bullish_bar(o2, c2)
            and o2 < c1  # gap down open
            and c2 > midpoint1
            and c2 < o1
        ):
            penetration = (c2 - midpoint1) / max(_body(o1, c1), 1e-9)
            conf = _clamp(0.55 + penetration * 0.2)
            return Pattern(
                name="Piercing Line",
                signal="bullish",
                confidence=conf,
                bar_index=idx,
                description="Bullish bar opens below prior low and pierces above midpoint — reversal.",
            )
        return None

    @staticmethod
    def dark_cloud_cover(
        o1: float, c1: float, o2: float, h2: float, l2: float, c2: float, idx: int
    ) -> Optional[Pattern]:
        """
        Dark Cloud Cover (bearish): after bullish bar, bearish bar opens above prior high
        and closes below midpoint.
        """
        midpoint1 = (o1 + c1) / 2
        if (
            _is_bullish_bar(o1, c1)
            and _is_bearish_bar(o2, c2)
            and o2 > c1  # gap up open
            and c2 < midpoint1
            and c2 > o1
        ):
            penetration = (midpoint1 - c2) / max(_body(o1, c1), 1e-9)
            conf = _clamp(0.55 + penetration * 0.2)
            return Pattern(
                name="Dark Cloud Cover",
                signal="bearish",
                confidence=conf,
                bar_index=idx,
                description="Bearish bar opens above prior high and closes below midpoint — reversal.",
            )
        return None

    # ── Three-bar patterns ─────────────────────────────────────────────────

    @staticmethod
    def morning_star(
        o1: float,
        c1: float,
        o2: float,
        c2: float,
        o3: float,
        c3: float,
        avg_body: float,
        idx: int,
    ) -> Optional[Pattern]:
        """
        Morning Star (bullish):
          bar1 = large bearish
          bar2 = small body (star)
          bar3 = large bullish closing above midpoint of bar1
        """
        mid1 = (o1 + c1) / 2
        is_star = _body(o2, c2) <= 0.30 * _body(o1, c1)
        if (
            _is_bearish_bar(o1, c1)
            and is_star
            and _is_bullish_bar(o3, c3)
            and c3 > mid1
            and _body(o1, c1) >= avg_body * 0.8
            and _body(o3, c3) >= avg_body * 0.8
        ):
            recovery = (c3 - mid1) / max(_body(o1, c1), 1e-9)
            conf = _clamp(0.65 + recovery * 0.2)
            return Pattern(
                name="Morning Star",
                signal="bullish",
                confidence=conf,
                bar_index=idx,
                description="Three-bar bullish reversal: large bearish → star → large bullish above midpoint.",
            )
        return None

    @staticmethod
    def evening_star(
        o1: float,
        c1: float,
        o2: float,
        c2: float,
        o3: float,
        c3: float,
        avg_body: float,
        idx: int,
    ) -> Optional[Pattern]:
        """
        Evening Star (bearish):
          bar1 = large bullish
          bar2 = small body (star)
          bar3 = large bearish closing below midpoint of bar1
        """
        mid1 = (o1 + c1) / 2
        is_star = _body(o2, c2) <= 0.30 * _body(o1, c1)
        if (
            _is_bullish_bar(o1, c1)
            and is_star
            and _is_bearish_bar(o3, c3)
            and c3 < mid1
            and _body(o1, c1) >= avg_body * 0.8
            and _body(o3, c3) >= avg_body * 0.8
        ):
            drop = (mid1 - c3) / max(_body(o1, c1), 1e-9)
            conf = _clamp(0.65 + drop * 0.2)
            return Pattern(
                name="Evening Star",
                signal="bearish",
                confidence=conf,
                bar_index=idx,
                description="Three-bar bearish reversal: large bullish → star → large bearish below midpoint.",
            )
        return None

    @staticmethod
    def three_white_soldiers(
        bars: List[Tuple[float, float, float, float]],  # [(o,h,l,c), ...]
        avg_body: float,
        idx: int,
    ) -> Optional[Pattern]:
        """
        Three consecutive bullish bars, each opening inside prior body and closing higher.
        """
        if len(bars) < 3:
            return None
        (o1, h1, l1, c1), (o2, h2, l2, c2), (o3, h3, l3, c3) = (
            bars[-3],
            bars[-2],
            bars[-1],
        )
        conditions = (
            _is_bullish_bar(o1, c1)
            and _is_bullish_bar(o2, c2)
            and _is_bullish_bar(o3, c3)
            and o2 >= o1
            and o2 <= c1  # opens inside bar1
            and o3 >= o2
            and o3 <= c2  # opens inside bar2
            and c2 > c1
            and c3 > c2
            and _body(o1, c1) >= avg_body * 0.7
            and _body(o2, c2) >= avg_body * 0.7
            and _body(o3, c3) >= avg_body * 0.7
        )
        if conditions:
            conf = _clamp(
                0.70
                + min(_body(o1, c1), _body(o2, c2), _body(o3, c3))
                / max(avg_body, 1e-9)
                * 0.1
            )
            return Pattern(
                name="Three White Soldiers",
                signal="bullish",
                confidence=conf,
                bar_index=idx,
                description="Three consecutive strong bullish bars — powerful trend continuation/reversal signal.",
            )
        return None

    @staticmethod
    def three_black_crows(
        bars: List[Tuple[float, float, float, float]], avg_body: float, idx: int
    ) -> Optional[Pattern]:
        """
        Three consecutive bearish bars, each opening inside prior body and closing lower.
        """
        if len(bars) < 3:
            return None
        (o1, h1, l1, c1), (o2, h2, l2, c2), (o3, h3, l3, c3) = (
            bars[-3],
            bars[-2],
            bars[-1],
        )
        conditions = (
            _is_bearish_bar(o1, c1)
            and _is_bearish_bar(o2, c2)
            and _is_bearish_bar(o3, c3)
            and o2 <= o1
            and o2 >= c1
            and o3 <= o2
            and o3 >= c2
            and c2 < c1
            and c3 < c2
            and _body(o1, c1) >= avg_body * 0.7
            and _body(o2, c2) >= avg_body * 0.7
            and _body(o3, c3) >= avg_body * 0.7
        )
        if conditions:
            conf = _clamp(
                0.70
                + min(_body(o1, c1), _body(o2, c2), _body(o3, c3))
                / max(avg_body, 1e-9)
                * 0.1
            )
            return Pattern(
                name="Three Black Crows",
                signal="bearish",
                confidence=conf,
                bar_index=idx,
                description="Three consecutive strong bearish bars — powerful downtrend signal.",
            )
        return None


# ---------------------------------------------------------------------------
# Chart Pattern Detectors
# ---------------------------------------------------------------------------


class ChartPatterns:
    """
    Higher-level chart pattern detection using swing highs/lows.
    Operates on the last N bars of daily OHLCV data.
    """

    @staticmethod
    def _find_local_highs(
        closes: np.ndarray, window: int = 5
    ) -> List[Tuple[int, float]]:
        """Return (index, value) of local high pivots."""
        pivots = []
        for i in range(window, len(closes) - window):
            seg = closes[i - window : i + window + 1]
            if closes[i] == seg.max():
                pivots.append((i, float(closes[i])))
        return pivots

    @staticmethod
    def _find_local_lows(
        closes: np.ndarray, window: int = 5
    ) -> List[Tuple[int, float]]:
        """Return (index, value) of local low pivots."""
        pivots = []
        for i in range(window, len(closes) - window):
            seg = closes[i - window : i + window + 1]
            if closes[i] == seg.min():
                pivots.append((i, float(closes[i])))
        return pivots

    @staticmethod
    def double_top(df: pd.DataFrame, tolerance: float = 0.03) -> Optional[Pattern]:
        """
        Double Top: two highs of similar height with a trough between them.
        tolerance = max allowed difference between the two peaks as a fraction.
        """
        if len(df) < 30:
            return None
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        pivots = ChartPatterns._find_local_highs(highs, window=5)
        if len(pivots) < 2:
            return None

        # Check last two significant highs
        p1_idx, p1_val = pivots[-2]
        p2_idx, p2_val = pivots[-1]

        if p2_idx <= p1_idx:
            return None

        diff = abs(p1_val - p2_val) / max(p1_val, p2_val)
        if diff > tolerance:
            return None

        # Trough between peaks must be meaningful
        trough = lows[p1_idx:p2_idx].min()
        depth = (min(p1_val, p2_val) - trough) / min(p1_val, p2_val)

        if depth < 0.04:  # at least 4% pullback
            return None

        conf = _clamp(0.55 + (tolerance - diff) / tolerance * 0.3 + depth * 0.5)
        return Pattern(
            name="Double Top",
            signal="bearish",
            confidence=conf,
            bar_index=p2_idx,
            pattern_type="chart",
            description=(
                f"Two peaks near ₹{p1_val:.1f} and ₹{p2_val:.1f} "
                f"({diff * 100:.1f}% apart) — bearish reversal pattern."
            ),
        )

    @staticmethod
    def double_bottom(df: pd.DataFrame, tolerance: float = 0.03) -> Optional[Pattern]:
        """
        Double Bottom: two lows of similar depth with a peak between them.
        """
        if len(df) < 30:
            return None
        lows = df["low"].values
        highs = df["high"].values

        pivots = ChartPatterns._find_local_lows(lows, window=5)
        if len(pivots) < 2:
            return None

        p1_idx, p1_val = pivots[-2]
        p2_idx, p2_val = pivots[-1]

        if p2_idx <= p1_idx:
            return None

        diff = abs(p1_val - p2_val) / max(p1_val, p2_val, 1e-9)
        if diff > tolerance:
            return None

        peak = highs[p1_idx:p2_idx].max()
        height = (peak - max(p1_val, p2_val)) / max(p1_val, p2_val)

        if height < 0.04:
            return None

        conf = _clamp(0.55 + (tolerance - diff) / tolerance * 0.3 + height * 0.5)
        return Pattern(
            name="Double Bottom",
            signal="bullish",
            confidence=conf,
            bar_index=p2_idx,
            pattern_type="chart",
            description=(
                f"Two troughs near ₹{p1_val:.1f} and ₹{p2_val:.1f} "
                f"({diff * 100:.1f}% apart) — bullish reversal pattern."
            ),
        )

    @staticmethod
    def head_and_shoulders(df: pd.DataFrame) -> Optional[Pattern]:
        """
        Head & Shoulders: left shoulder < head > right shoulder (roughly equal shoulders).
        Bearish reversal.
        """
        if len(df) < 60:
            return None
        highs = df["high"].values

        pivots = ChartPatterns._find_local_highs(highs, window=7)
        if len(pivots) < 3:
            return None

        # Try last three pivots
        ls_idx, ls_val = pivots[-3]
        h_idx, h_val = pivots[-2]
        rs_idx, rs_val = pivots[-1]

        if not (ls_idx < h_idx < rs_idx):
            return None

        # Head must be higher than both shoulders
        if h_val <= ls_val or h_val <= rs_val:
            return None

        # Shoulders should be roughly equal (within 8%)
        shoulder_diff = abs(ls_val - rs_val) / max(ls_val, rs_val)
        if shoulder_diff > 0.08:
            return None

        conf = _clamp(0.60 + (0.08 - shoulder_diff) / 0.08 * 0.25)
        return Pattern(
            name="Head & Shoulders",
            signal="bearish",
            confidence=conf,
            bar_index=rs_idx,
            pattern_type="chart",
            description=(
                f"Left shoulder ₹{ls_val:.1f}, head ₹{h_val:.1f}, "
                f"right shoulder ₹{rs_val:.1f} — classic bearish reversal."
            ),
        )

    @staticmethod
    def inverse_head_and_shoulders(df: pd.DataFrame) -> Optional[Pattern]:
        """
        Inverse Head & Shoulders: left shoulder > head < right shoulder.
        Bullish reversal.
        """
        if len(df) < 60:
            return None
        lows = df["low"].values

        pivots = ChartPatterns._find_local_lows(lows, window=7)
        if len(pivots) < 3:
            return None

        ls_idx, ls_val = pivots[-3]
        h_idx, h_val = pivots[-2]
        rs_idx, rs_val = pivots[-1]

        if not (ls_idx < h_idx < rs_idx):
            return None

        # Head must be lower than both shoulders
        if h_val >= ls_val or h_val >= rs_val:
            return None

        shoulder_diff = abs(ls_val - rs_val) / max(ls_val, rs_val)
        if shoulder_diff > 0.08:
            return None

        conf = _clamp(0.60 + (0.08 - shoulder_diff) / 0.08 * 0.25)
        return Pattern(
            name="Inverse Head & Shoulders",
            signal="bullish",
            confidence=conf,
            bar_index=rs_idx,
            pattern_type="chart",
            description=(
                f"Left shoulder ₹{ls_val:.1f}, head ₹{h_val:.1f}, "
                f"right shoulder ₹{rs_val:.1f} — classic bullish reversal."
            ),
        )

    @staticmethod
    def ascending_triangle(df: pd.DataFrame) -> Optional[Pattern]:
        """
        Ascending Triangle: flat resistance top + rising lows → bullish breakout pattern.
        """
        if len(df) < 40:
            return None
        highs = df["high"].values[-40:]
        lows = df["low"].values[-40:]

        # Flat resistance: last few highs within 2% of each other
        recent_highs = highs[-20:]
        resistance = recent_highs.max()
        high_variation = (recent_highs.max() - recent_highs.min()) / resistance
        if high_variation > 0.02:
            return None

        # Rising lows: fit a line to lows
        x = np.arange(len(lows[-20:]))
        slope = np.polyfit(x, lows[-20:], 1)[0]
        if slope <= 0:
            return None

        conf = _clamp(0.55 + slope / lows[-20:].mean() * 50)
        return Pattern(
            name="Ascending Triangle",
            signal="bullish",
            confidence=conf,
            bar_index=len(df) - 1,
            pattern_type="chart",
            description=(
                f"Flat resistance near ₹{resistance:.1f} with rising lows — "
                "bullish continuation/breakout pattern."
            ),
        )

    @staticmethod
    def descending_triangle(df: pd.DataFrame) -> Optional[Pattern]:
        """
        Descending Triangle: flat support bottom + falling highs → bearish breakdown pattern.
        """
        if len(df) < 40:
            return None
        highs = df["high"].values[-20:]
        lows = df["low"].values[-20:]

        # Flat support
        support = lows.min()
        low_variation = (lows.max() - lows.min()) / max(support, 1e-9)
        if low_variation > 0.02:
            return None

        # Falling highs
        x = np.arange(len(highs))
        slope = np.polyfit(x, highs, 1)[0]
        if slope >= 0:
            return None

        conf = _clamp(0.55 + abs(slope) / highs.mean() * 50)
        return Pattern(
            name="Descending Triangle",
            signal="bearish",
            confidence=conf,
            bar_index=len(df) - 1,
            pattern_type="chart",
            description=(
                f"Flat support near ₹{support:.1f} with falling highs — "
                "bearish breakdown pattern."
            ),
        )

    @staticmethod
    def symmetrical_triangle(df: pd.DataFrame) -> Optional[Pattern]:
        """
        Symmetrical Triangle: converging trendlines (coiling) — neutral continuation.
        """
        if len(df) < 40:
            return None
        highs = df["high"].values[-20:]
        lows = df["low"].values[-20:]
        x = np.arange(20)

        high_slope = np.polyfit(x, highs, 1)[0]
        low_slope = np.polyfit(x, lows, 1)[0]

        # High slope negative, low slope positive → converging
        if high_slope >= 0 or low_slope <= 0:
            return None

        conf = 0.55
        return Pattern(
            name="Symmetrical Triangle",
            signal="neutral",
            confidence=conf,
            bar_index=len(df) - 1,
            pattern_type="chart",
            description="Converging highs and lows — coiling for breakout (direction uncertain).",
        )

    @staticmethod
    def bull_flag(df: pd.DataFrame) -> Optional[Pattern]:
        """
        Bull Flag: sharp rally (pole) followed by a tight consolidation channel (flag).
        """
        if len(df) < 30:
            return None
        closes = df["close"].values

        # Pole: last 30 bars, first 15 = strong rally
        pole = closes[-30:-15]
        flag = closes[-15:]

        pole_gain = (pole[-1] - pole[0]) / max(pole[0], 1e-9)
        if pole_gain < 0.05:  # at least 5% rally in pole
            return None

        # Flag: tight range + mild pullback
        flag_range = (flag.max() - flag.min()) / max(flag.mean(), 1e-9)
        flag_drift = (flag[-1] - flag[0]) / max(flag[0], 1e-9)

        if flag_range > 0.05 or flag_drift > 0.02:  # flag must be tight
            return None

        conf = _clamp(0.60 + pole_gain * 0.5)
        return Pattern(
            name="Bull Flag",
            signal="bullish",
            confidence=conf,
            bar_index=len(df) - 1,
            pattern_type="chart",
            description=(
                f"Strong rally (+{pole_gain * 100:.1f}%) followed by tight "
                "consolidation — bullish continuation setup."
            ),
        )

    @staticmethod
    def bear_flag(df: pd.DataFrame) -> Optional[Pattern]:
        """
        Bear Flag: sharp decline (pole) followed by a tight consolidation (flag).
        """
        if len(df) < 30:
            return None
        closes = df["close"].values

        pole = closes[-30:-15]
        flag = closes[-15:]

        pole_drop = (pole[0] - pole[-1]) / max(pole[0], 1e-9)
        if pole_drop < 0.05:
            return None

        flag_range = (flag.max() - flag.min()) / max(flag.mean(), 1e-9)
        flag_drift = (flag[-1] - flag[0]) / max(flag[0], 1e-9)

        if flag_range > 0.05 or flag_drift < -0.02:
            return None

        conf = _clamp(0.60 + pole_drop * 0.5)
        return Pattern(
            name="Bear Flag",
            signal="bearish",
            confidence=conf,
            bar_index=len(df) - 1,
            pattern_type="chart",
            description=(
                f"Sharp decline ({pole_drop * 100:.1f}%) followed by tight "
                "consolidation — bearish continuation setup."
            ),
        )

    @staticmethod
    def cup_and_handle(df: pd.DataFrame) -> Optional[Pattern]:
        """
        Cup and Handle: rounded bottom (cup) followed by a small pullback (handle).
        Bullish continuation.
        """
        if len(df) < 60:
            return None
        closes = df["close"].values[-60:]

        # Cup: U-shaped curve — minimum in the middle
        n = len(closes)
        mid_third = closes[n // 3 : 2 * n // 3]
        left_third = closes[: n // 3]
        right_third = closes[2 * n // 3 :]

        cup_low = mid_third.min()
        left_high = left_third.max()
        right_high = right_third.max()

        # Both rims should be at similar levels
        rim_diff = abs(left_high - right_high) / max(left_high, right_high)
        depth = (min(left_high, right_high) - cup_low) / max(
            min(left_high, right_high), 1e-9
        )

        if rim_diff > 0.05 or depth < 0.08:
            return None

        # Handle: small pullback in the last 10 bars
        handle = closes[-10:]
        handle_pullback = (handle.max() - handle.min()) / max(handle.max(), 1e-9)
        if handle_pullback > 0.05 or handle[-1] < handle[0] * 0.97:
            return None

        conf = _clamp(0.60 + (0.05 - rim_diff) / 0.05 * 0.2 + depth * 0.5)
        return Pattern(
            name="Cup and Handle",
            signal="bullish",
            confidence=conf,
            bar_index=len(df) - 1,
            pattern_type="chart",
            description=(
                f"Rounded U-shaped base (depth {depth * 100:.1f}%) with small handle — "
                "bullish continuation breakout setup."
            ),
        )


# ---------------------------------------------------------------------------
# Main PatternDetector class
# ---------------------------------------------------------------------------


class PatternDetector:
    """
    Orchestrates all candlestick and chart pattern detection.

    Usage::

        detector = PatternDetector()
        result = detector.detect_all(df, lookback_candles=3)
        # result['candlestick'] → list[dict]
        # result['chart']       → list[dict]
        # result['summary']     → {signal, confidence, total_patterns}
    """

    def __init__(self) -> None:
        self._cs = CandlestickPatterns()
        self._cp = ChartPatterns()

    def detect_candlestick(
        self, df: pd.DataFrame, lookback: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Scan the last `lookback` bars for candlestick patterns.

        Returns list of pattern dicts, sorted by confidence desc.
        """
        if len(df) < 4:
            return []

        detected: List[Pattern] = []
        avg_body_series = _avg_body(df, window=14)

        # Scan recent bars
        window = min(lookback, len(df))
        scan_range = range(max(3, len(df) - window), len(df))

        for i in scan_range:
            o, h, l, c = (
                df["open"].iloc[i],
                df["high"].iloc[i],
                df["low"].iloc[i],
                df["close"].iloc[i],
            )
            ab = (
                float(avg_body_series.iloc[i])
                if not pd.isna(avg_body_series.iloc[i])
                else 1.0
            )

            # Single-bar
            for fn in [
                lambda: self._cs.doji(o, h, l, c, ab, i),
                lambda: self._cs.dragonfly_doji(o, h, l, c, i),
                lambda: self._cs.gravestone_doji(o, h, l, c, i),
                lambda: self._cs.hammer(o, h, l, c, ab, i),
                lambda: self._cs.inverted_hammer(o, h, l, c, ab, i),
                lambda: self._cs.shooting_star(o, h, l, c, ab, i),
                lambda: self._cs.spinning_top(o, h, l, c, ab, i),
                lambda: self._cs.marubozu_bullish(o, h, l, c, ab, i),
                lambda: self._cs.marubozu_bearish(o, h, l, c, ab, i),
            ]:
                try:
                    p = fn()
                    if p:
                        detected.append(p)
                except Exception:
                    pass

            # Two-bar
            if i >= 1:
                o1 = df["open"].iloc[i - 1]
                c1 = df["close"].iloc[i - 1]
                h1 = df["high"].iloc[i - 1]
                l1 = df["low"].iloc[i - 1]
                for fn2 in [
                    lambda: self._cs.bullish_engulfing(o1, c1, o, c, ab, i),
                    lambda: self._cs.bearish_engulfing(o1, c1, o, c, ab, i),
                    lambda: self._cs.bullish_harami(o1, c1, o, c, i),
                    lambda: self._cs.bearish_harami(o1, c1, o, c, i),
                    lambda: self._cs.piercing_line(o1, c1, o, h, l, c, i),
                    lambda: self._cs.dark_cloud_cover(o1, c1, o, h, l, c, i),
                ]:
                    try:
                        p = fn2()
                        if p:
                            detected.append(p)
                    except Exception:
                        pass

            # Three-bar
            if i >= 2:
                o2 = df["open"].iloc[i - 2]
                c2 = df["close"].iloc[i - 2]
                o1_ = df["open"].iloc[i - 1]
                c1_ = df["close"].iloc[i - 1]
                for fn3 in [
                    lambda: self._cs.morning_star(o2, c2, o1_, c1_, o, c, ab, i),
                    lambda: self._cs.evening_star(o2, c2, o1_, c1_, o, c, ab, i),
                ]:
                    try:
                        p = fn3()
                        if p:
                            detected.append(p)
                    except Exception:
                        pass

            # Three-bar soldiers/crows
            if i >= 2:
                bars = [
                    (
                        df["open"].iloc[j],
                        df["high"].iloc[j],
                        df["low"].iloc[j],
                        df["close"].iloc[j],
                    )
                    for j in (i - 2, i - 1, i)
                ]
                for fn4 in [
                    lambda: self._cs.three_white_soldiers(bars, ab, i),
                    lambda: self._cs.three_black_crows(bars, ab, i),
                ]:
                    try:
                        p = fn4()
                        if p:
                            detected.append(p)
                    except Exception:
                        pass

        # Sort by confidence descending
        detected.sort(key=lambda x: x.confidence, reverse=True)
        return [p.to_dict() for p in detected]

    def detect_chart(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run all chart pattern detectors on the full DataFrame.

        Returns list of detected chart pattern dicts.
        """
        detected: List[Pattern] = []

        checkers = [
            self._cp.double_top,
            self._cp.double_bottom,
            self._cp.head_and_shoulders,
            self._cp.inverse_head_and_shoulders,
            self._cp.ascending_triangle,
            self._cp.descending_triangle,
            self._cp.symmetrical_triangle,
            self._cp.bull_flag,
            self._cp.bear_flag,
            self._cp.cup_and_handle,
        ]

        for checker in checkers:
            try:
                p = checker(df)
                if p:
                    detected.append(p)
            except Exception as exc:
                logger.debug(f"Chart pattern error ({checker.__name__}): {exc}")

        detected.sort(key=lambda x: x.confidence, reverse=True)
        return [p.to_dict() for p in detected]

    def detect_all(self, df: pd.DataFrame, lookback_candles: int = 5) -> Dict[str, Any]:
        """
        Run all pattern detectors and return a comprehensive result dict.

        Returns::

            {
                'candlestick': [...],   # recent candlestick patterns
                'chart': [...],         # chart patterns detected in full series
                'summary': {
                    'signal': 'bullish' | 'bearish' | 'neutral',
                    'confidence': float,
                    'total_patterns': int,
                    'bullish_count': int,
                    'bearish_count': int,
                    'top_pattern': str | None,
                    'interpretation': str,
                }
            }
        """
        cs_patterns = self.detect_candlestick(df, lookback=lookback_candles)
        chart_pats = self.detect_chart(df)
        all_pats = cs_patterns + chart_pats

        bullish = [p for p in all_pats if p["signal"] == "bullish"]
        bearish = [p for p in all_pats if p["signal"] == "bearish"]

        bull_score = sum(p["confidence"] for p in bullish)
        bear_score = sum(p["confidence"] for p in bearish)

        if bull_score > bear_score * 1.2:
            signal = "bullish"
            conf = bull_score / max(bull_score + bear_score, 1e-9)
            interp = (
                f"{len(bullish)} bullish pattern(s) detected — momentum favours upside."
            )
        elif bear_score > bull_score * 1.2:
            signal = "bearish"
            conf = bear_score / max(bull_score + bear_score, 1e-9)
            interp = f"{len(bearish)} bearish pattern(s) detected — momentum favours downside."
        else:
            signal = "neutral"
            conf = 0.5
            interp = "Mixed signals — no clear directional bias from patterns."

        top_pattern = all_pats[0]["name"] if all_pats else None

        return {
            "candlestick": cs_patterns,
            "chart": chart_pats,
            "summary": {
                "signal": signal,
                "confidence": round(_clamp(conf), 3),
                "total_patterns": len(all_pats),
                "bullish_count": len(bullish),
                "bearish_count": len(bearish),
                "top_pattern": top_pattern,
                "interpretation": interp,
            },
        }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def detect_patterns(df: pd.DataFrame, lookback_candles: int = 5) -> Dict[str, Any]:
    """
    Convenience wrapper — detect all candlestick and chart patterns.

    Args:
        df:               OHLCV DataFrame (lowercase columns, DatetimeIndex)
        lookback_candles: How many recent bars to scan for candlestick patterns

    Returns:
        Rich dict with 'candlestick', 'chart', and 'summary' keys.
    """
    detector = PatternDetector()
    return detector.detect_all(df, lookback_candles=lookback_candles)
