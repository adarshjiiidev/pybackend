"""
Analytics Package — Phase 2
Advanced market analytics: technical indicators, pattern detection,
OI analysis, market breadth, buyer/seller pressure, and shock detection.
"""

import logging

logger = logging.getLogger(__name__)

# ── Always-available modules ──────────────────────────────────────────────────

from .oi_analyzer import OIAnalyzer, analyze_oi
from .pattern_detector import PatternDetector, detect_patterns
from .technical_engine import TechnicalEngine, calculate_all_indicators

# ── Optional modules (created lazily if missing) ──────────────────────────────

try:
    from .buyer_seller_analyzer import (
        BuyerSellerAnalyzer,
        analyze_buyer_seller_pressure,
    )
except ImportError:
    logger.debug("buyer_seller_analyzer not available yet")

    class BuyerSellerAnalyzer:  # type: ignore[no-redef]
        """Stub until buyer_seller_analyzer.py is created."""

        def analyze(self, df, fii_dii_data=None):
            return {"signal": "neutral", "note": "Module not yet implemented"}

    def analyze_buyer_seller_pressure(df, fii_dii_data=None):  # type: ignore[misc]
        return BuyerSellerAnalyzer().analyze(df, fii_dii_data)


try:
    from .market_breadth import MarketBreadthAnalyzer
except ImportError:
    logger.debug("market_breadth not available yet")

    class MarketBreadthAnalyzer:  # type: ignore[no-redef]
        """Stub until market_breadth.py is created."""

        def analyze(self, symbols_data):
            return {"signal": "neutral", "note": "Module not yet implemented"}


try:
    from .shock_detector import ShockDetector, detect_shocks
except ImportError:
    logger.debug("shock_detector not available yet — using ShockDetectionAgent instead")

    class ShockDetector:  # type: ignore[no-redef]
        """Stub — use swarm.agents.shock_detection_agent for full shock detection."""

        def detect(self, df, vix_data=None):
            return {
                "signal": "neutral",
                "note": "Use ShockDetectionAgent for full analysis",
            }

    def detect_shocks(df, vix_data=None):  # type: ignore[misc]
        return ShockDetector().detect(df, vix_data)


__all__ = [
    "TechnicalEngine",
    "calculate_all_indicators",
    "PatternDetector",
    "detect_patterns",
    "OIAnalyzer",
    "analyze_oi",
    "BuyerSellerAnalyzer",
    "analyze_buyer_seller_pressure",
    "ShockDetector",
    "detect_shocks",
    "MarketBreadthAnalyzer",
]
