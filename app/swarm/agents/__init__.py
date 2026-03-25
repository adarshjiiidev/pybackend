"""
DaddysAI Swarm Agents Package
==============================
All specialized sub-agents that the Master Orchestrator can dynamically
create, run, and dispose.

Available agents:
  TechnicalAnalysisAgent  — Full TA suite: RSI, MACD, BB, patterns, S/R levels
  OIAnalysisAgent         — Option chain: PCR, Max Pain, GEX, OI walls
  WebResearchAgent        — Autonomous multi-step web research + KB learning
  PredictionAgent         — ML ensemble price/trend prediction
  GlobalMarketAgent       — Global indices, DXY, bonds, commodities monitor
  ShockDetectionAgent     — Crash/shock/volatility anomaly detection
  FundamentalsAgent       — Company fundamentals, PE, ROE, financials
  ReportAgent             — Synthesize all findings into structured reports
  DataFetchAgent          — Fetch + store OHLCV for any symbol
  SentimentAgent          — News sentiment + social media signal analysis
"""

from .data_fetch_agent import DataFetchAgent
from .fundamentals_agent import FundamentalsAgent
from .global_market_agent import GlobalMarketAgent
from .oi_analysis_agent import OIAnalysisAgent
from .prediction_agent import PredictionAgent
from .report_agent import ReportAgent
from .sentiment_agent import SentimentAgent
from .shock_detection_agent import ShockDetectionAgent
from .technical_analysis_agent import TechnicalAnalysisAgent
from .web_research_agent import WebResearchAgent

__all__ = [
    "TechnicalAnalysisAgent",
    "OIAnalysisAgent",
    "WebResearchAgent",
    "PredictionAgent",
    "GlobalMarketAgent",
    "ShockDetectionAgent",
    "FundamentalsAgent",
    "ReportAgent",
    "DataFetchAgent",
    "SentimentAgent",
]
