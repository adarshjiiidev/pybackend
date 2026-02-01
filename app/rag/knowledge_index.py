"""
Comprehensive Knowledge Base Index
Built by analyzing all 21 domain knowledge files to create structured retrieval.
"""

# COMPREHENSIVE INDEX - Built from analyzing all knowledge base files

KNOWLEDGE_INDEX = {
    # === CORE CONCEPTS ===
    "wtb": {
        "primary_files": ["wtb.txt"],
        "related_files": ["shifting_pressure.txt", "game_of_percentage.txt"],
        "keywords": ["weak towards bottom", "75% rule", "bottom side", "bearish pressure", "support weakness"],
        "category": "pressure_analysis",
        "description": "Weak Towards Bottom - when 2nd highest OI >=75% at bottom side"
    },
    
    "wtt": {
        "primary_files": ["wtt.txt"],
        "related_files": ["shifting_pressure.txt", "game_of_percentage.txt"],
        "keywords": ["weak towards top", "75% rule", "top side", "bullish pressure", "resistance weakness"],
        "category": "pressure_analysis",
        "description": "Weak Towards Top - when 2nd highest OI >=75% at top side"
    },
    
    "shifting": {
        "primary_files": ["shifting_pressure.txt"],
        "related_files": ["wtb.txt", "wtt.txt", "game_of_percentage.txt"],
        "keywords": ["shift", "pressure change", "reversal", "natural weakness", "blood bath"],
        "category": "pressure_analysis",
        "description": "Support/Resistance shifting rules and pressure analysis"
    },
    
    # === LTP & SCENARIOS ===
    "ltp": {
        "primary_files": ["ai_ltp.txt", "ltp_features.txt"],
        "related_files": ["coa_1_0.txt", "scenarios.txt"],
        "keywords": ["ltp calculator", "max pain", "ai features", "blast", "swing", "arbitrage"],
        "category": "ltp_system",
        "description": "LTP Calculator features and AI-based analysis"
    },
    
    "coa": {
        "primary_files": ["coa_1_0.txt", "coa_2_0.txt"],
        "related_files": ["scenarios.txt", "ltp_features.txt"],
        "keywords": ["chart of accuracy", "9 scenarios", "oi change", "call writers", "put writers"],
        "category": "ltp_system",
        "description": "Chart of Accuracy - 9 scenarios for market direction"
    },
    
    "scenarios": {
        "primary_files": ["scenarios.txt"],
        "related_files": ["coa_1_0.txt", "coa_2_0.txt"],
        "keywords": ["9 scenarios", "scenario analysis", "market direction"],
        "category": "ltp_system",
        "description": "Detailed breakdown of 9 market scenarios"
    },
    
    # === MARKET STATES ===
    "soc": {
        "primary_files": ["soc.txt"],
        "related_files": ["wtb.txt", "wtt.txt"],
        "keywords": ["state of confusion", "1hr rule", "2hr rule", "3hr rule", "range bound"],
        "category": "market_state",
        "description": "State of Confusion - when WTB and WTT both present"
    },
    
    "strong": {
        "primary_files": ["strong.txt"],
        "related_files": ["wtb.txt", "wtt.txt"],
        "keywords": ["strong market", "second highest", "less than 75%"],
        "category": "market_state",
        "description": "Strong market - when 2nd highest <75%"
    },
    
    # === SUPPORT & RESISTANCE ===
    "support_resistance": {
        "primary_files": ["support_resistance_basics.txt", "imaginary_line.txt"],
        "related_files": ["wtb.txt", "wtt.txt", "strong.txt"],
        "keywords": ["support", "resistance", "sheesh aasan", "head legs", "pushing", "imaginary line"],
        "category": "fundamentals",
        "description": "Support/Resistance basics and Sheesh Aasan concept"
    },
    
    "imaginary_line": {
        "primary_files": ["imaginary_line.txt"],
        "related_files": ["support_resistance_basics.txt"],
        "keywords": ["itm", "otm", "at the money", "dividing line"],
        "category": "fundamentals",
        "description": "Imaginary line concept - ITM/OTM dividing line"
    },
    
    # === TRADING STRATEGIES ===
    "trading_strategies": {
        "primary_files": ["trading_strategies.txt"],
        "related_files": ["coa_1_0.txt", "ltp_features.txt"],
        "keywords": ["9:20 strategy", "reversal types", "entry exit", "6 reversals"],
        "category": "strategies",
        "description": "Trading strategies including 9:20 AM strategy and reversals"
    },
    
    "weekly_range": {
        "primary_files": ["weekly_range.txt"],
        "related_files": ["scenarios.txt"],
        "keywords": ["l1", "l2", "l3", "weekly levels", "range"],
        "category": "strategies",
        "description": "Weekly Range levels (L1, L2, L3)"
    },
    
    # === PERCENTAGE & CALCULATIONS ===
    "percentage": {
        "primary_files": ["game_of_percentage.txt"],
        "related_files": ["wtb.txt", "wtt.txt", "shifting_pressure.txt"],
        "keywords": ["increasing percentage", "decreasing percentage", "percentage change", "75%"],
        "category": "analysis",
        "description": "Game of Percentage - analyzing percentage changes in OI"
    },
    
    # === OPTIONS & TECHNICAL ===
    "options": {
        "primary_files": ["options_trading.txt"],
        "related_files": ["imaginary_line.txt", "coa_1_0.txt"],
        "keywords": ["call", "put", "greeks", "option chain", "strike price", "premium"],
        "category": "trading_concepts",
        "description": "Options trading basics, Greeks, and option chain analysis"
    },
    
    "technical_analysis": {
        "primary_files": ["technical_analysis.txt"],
        "related_files": ["general_finance.txt"],
        "keywords": ["candlestick", "moving average", "rsi", "macd", "patterns", "indicators"],
        "category": "trading_concepts",
        "description": "Technical analysis concepts and indicators"
    },
    
    # === GENERAL FINANCE ===
    "finance_basics": {
        "primary_files": ["general_finance.txt"],
        "related_files": ["technical_analysis.txt"],
        "keywords": ["market terms", "order types", "corporate actions", "fundamentals"],
        "category": "trading_concepts",
        "description": "General finance concepts and market terminology"
    },
    
    # === LTP FEATURES ===
    "ltp_blast": {
        "primary_files": ["ltp_features.txt"],
        "related_files": ["ai_ltp.txt"],
        "keywords": ["intraday", "c1", "c2", "p1", "p2", "bullish trigger", "bearish trigger"],
        "category": "ltp_system",
        "description": "LTP Blast feature for intraday stock selection"
    },
    
    "ltp_swing": {
        "primary_files": ["ltp_features.txt"],
        "related_files": ["ai_ltp.txt"],
        "keywords": ["positional", "swing trading", "delivery", "oi to oi"],
        "category": "ltp_system",
        "description": "LTP Swing feature for positional trading"
    },
    
    "arbitrage": {
        "primary_files": ["ltp_features.txt"],
        "related_files": [],
        "keywords": ["arbitrage", "future spot difference", "risk free", "expiry"],
        "category": "ltp_system",
        "description": "Arbitrage trading - exploiting Future vs Spot price difference"
    },
    
    # === CONSTRAINTS & RULES ===
    "constraints": {
        "primary_files": ["constraints.txt"],
        "related_files": ["master_index.txt"],
        "keywords": ["global rules", "banned terms", "currency rules", "hallucination prevention"],
        "category": "rules",
        "description": "CRITICAL global constraints and rules - HIGHEST PRIORITY"
    },
    
    # === ABOUT ===
    "about": {
        "primary_files": ["about.txt"],
        "related_files": [],
        "keywords": ["vinay sir", "daddy's international school", "mission", "investing daddy"],
        "category": "meta",
        "description": "About Daddy's International School and mission"
    }
}

# Category definitions
CATEGORIES = {
    "pressure_analysis": {
        "description": "WTB, WTT, shifting pressure analysis",
        "priority": 1  # Highest priority
    },
    "ltp_system": {
        "description": "LTP Calculator, COA, scenarios, features",
        "priority": 1
    },
    "market_state": {
        "description": "SOC, Strong market states",
        "priority": 2
    },
    "fundamentals": {
        "description": "Support/Resistance basics, core concepts",
        "priority": 1
    },
    "strategies": {
        "description": "Trading strategies and techniques",
        "priority": 2
    },
    "analysis": {
        "description": "Analysis methods and calculations",
        "priority": 2
    },
    "trading_concepts": {
        "description": "Options, technical analysis, general finance",
        "priority": 3
    },
    "rules": {
        "description": "Constraints and global rules",
        "priority": 0  # ALWAYS check first
    },
    "meta": {
        "description": "About and general information",
        "priority": 4
    }
}

# Quick lookup for common queries
QUICK_LOOKUP = {
    # WTB/WTT related
    "what is wtb": "wtb",
    "wtb meaning": "wtb",
    "wtb rules": "wtb",
    "weak towards bottom": "wtb",
    "what is wtt": "wtt",
    "wtt meaning": "wtt",
    "wtt rules": "wtt",
    "weak towards top": "wtt",
    "75 percent rule": ["wtb", "wtt"],
    "75% rule": ["wtb", "wtt"],
    
    # Shifting
    "what is shifting": "shifting",
    "shifting pressure": "shifting",
    "how pressure shifts": "shifting",
    "blood bath": "shifting",
    "natural weakness": "shifting",
    
    # LTP
    "ltp calculator": "ltp",
    "how ltp works": "ltp",
    "ltp features": ["ltp", "ltp_blast", "ltp_swing"],
    "ltp blast": "ltp_blast",
    "ltp swing": "ltp_swing",
    
    # COA & Scenarios
    "chart of accuracy": "coa",
    "9 scenarios": ["coa", "scenarios"],
    "coa 1.0": "coa",
    "coa 2.0": "coa",
    
    # Market States
    "state of confusion": "soc",
    "soc": "soc",
    "strong market": "strong",
    
    # Support/Resistance
    "support resistance": "support_resistance",
    "sheesh aasan": "support_resistance",
    "imaginary line": "imaginary_line",
    "itm otm": "imaginary_line",
    
    # Trading
    "trading strategy": "trading_strategies",
    "920 strategy": "trading_strategies",
    "9:20 strategy": "trading_strategies",
    "reversals": "trading_strategies",
    
    # Options
    "option trading": "options",
    "greeks": "options",
    "call put": "options",
    
    # Technical
    "technical analysis": "technical_analysis",
    "candlestick": "technical_analysis",
    "indicators": "technical_analysis"
}
