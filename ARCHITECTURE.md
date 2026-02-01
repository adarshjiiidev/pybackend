# Production Architecture - Daaddys AI

## Complete System with Safety Layers

```
User Input
    ↓
┌─────────────────────────────────┐
│ Safety Classifier (Pre-check)   │ ← Detect risk level
│ Risk: LOW | MEDIUM | HIGH       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Router/Planner Agent            │ ← Classify intent
│ + Research Loop Detection       │ ← Decide deep search
└─────────────────────────────────┘
    ↓
    ├── Simple Query ──→ Specialist Agent
    │
    └── Complex Query ──→ Autonomous Research Loop
                          ├── Plan (2-4 steps)
                          ├── Research (iterate)
                          ├── Evaluate (confidence)
                          └── Decide (continue/stop)
    ↓
┌─────────────────────────────────┐
│ Validation Agent (Post-check)   │ ← Scan for hallucinations
│ - Unsourced claims              │ ← Auto-sanitize
│ - Overconfident language        │
│ - Price predictions             │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Final Safety Gate               │ ← Add disclaimer
│ + Verifier Agent               │ ← Clean formatting
└─────────────────────────────────┘
    ↓
Streaming SSE Response
```

## Agents (13 Total)

### Core Routing (2)
1. **SafetyClassifierAgent** - Pre-check risk detection
2. **RouterAgent** - Intent classification + research detection

### Specialist Agents (6)
3. **MarketResearchAgent** - Indian equity fundamentals
4. **RealtimeAnalysisAgent** - Intraday + technical
5. **PortfolioAgent** - Allocation + risk management
6. **ExplainerAgent** - Educational content
7. **CryptoAgent** - Cryptocurrency analysis
8. **CompoundAgent** - Web search + real-time data

### Research Loop (4)
9. **PlannerAgent** - Break into research steps
10. **ResearcherAgent** - Execute steps iteratively
11. **EvaluatorAgent** - Score confidence (quality/freshness/agreement)
12. **RefinerAgent** - Synthesize findings

### Safety & Quality (2)
13. **ValidationAgent** - Post-check fact validation
14. **VerifierAgent** - Final formatting + disclaimers

## Safety Flow

**Pre-Check (Safety Classifier):**
- HIGH risk → education_only mode
- MEDIUM risk → analysis_with_disclaimer mode
- LOW risk → general_info mode

**Post-Check (Validator):**
- Scan for: overconfident language, unsourced claims, price predictions
- Auto-sanitize: "will" → "may", "definitely" → "potentially"
- Add risk-appropriate disclaimers

**Degraded Mode (HIGH risk):**
- No buy/sell instructions
- Scenario-based reasoning only
- Explicit uncertainty language
- Educational framing

## Tools (9)

1. get_stock_fundamentals
2. get_technical_indicators
3. get_stock_news
4. get_crypto_narrative
5. compare_stocks
6. get_sector_analysis
7. calculate_portfolio_optimization
8. search_knowledge_base (21 domain files)
9. search_web (Groq Compound)

## Knowledge Base (21 Files)

- WTB/WTT rules (weak towards bottom/top)
- LTP calculator (chart of accuracy, scenarios)
- Trading strategies (9:20 AM, reversals)
- Support/Resistance basics
- Options trading
- Technical analysis
- Constraints (anti-hallucination rules)

## API Endpoints

**POST /chat/stream**
```json
{
  "query": "Compare TCS vs Infosys",
  "mode": "auto",
  "session_id": "uuid",
  "enable_deep_search": false
}
```

**Response (SSE):**
```
data: {"content": "...", "done": false}
data: {"content": "...", "done": true, "metadata": {...}}
```

## Configuration (.env)

```env
# Models
MODEL_REASONING_DEEP=openai/gpt-oss-120b
MODEL_ANALYSIS=llama-3.3-70b-versatile
MODEL_FAST=llama-3.1-8b-instant
MODEL_COMPOUND=groq/compound

# Research Loop
ENABLE_RESEARCH_LOOP=true
MAX_RESEARCH_ITERATIONS=5
CONFIDENCE_THRESHOLD=0.75

# Safety
TEMPERATURE_CREATIVE=0.3  # Low for factual accuracy
ENABLE_VALIDATION=true
```

## Competitive Edge vs Perplexity

| Feature | Daaddys AI | Perplexity |
|---------|------------|------------|
| Speed | 300-1000 tok/s | ~30-50 tok/s |
| Reasoning | Explicit (GPT-OSS) | Hidden |
| Research | Multi-iteration | Single-shot |
| Domain Knowledge | 21 specialized files | General web |
| Safety Layers | 3-stage validation | Basic |
| Indian Markets | Deep expertise | Generic |
| Confidence Scoring | Yes (0-1) | No |

**Result:** 5-10x faster, deeper, safer, more reliable for Indian equity & global crypto markets.
