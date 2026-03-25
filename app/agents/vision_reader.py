"""
Vision Reader — Pre-router image understanding node.

Architecture:
  START → vision_reader (no-op if no images) → router → [normal agents] → verifier → END

On startup:
  1. Loads ALL knowledge base .txt files from /txt/ directory
  2. Builds a domain-aware prompt with the KB knowledge + color code dictionary
  3. When an image arrives: calls vision model with this full context
  4. Outputs a precise NLP scenario → appended to query → flows into normal routing

This means ALL agents (realtime_analysis, market_research, portfolio, explainer) receive
image content as natural language text. No special-case image paths in the graph.
"""

import logging
from pathlib import Path
from ..config import settings
from ..config.key_rotator import get_groq_client
from ..models.agent_state import AgentState

logger = logging.getLogger(__name__)

# ── KB txt loading ────────────────────────────────────────────────────────────

# Priority order: most important for image reading first
_PRIORITY_KB_FILES = [
    "wtb.txt",            # WTB definition, 75% rule, pressure rules
    "wtt.txt",            # WTT definition, 75% rule, pressure rules
    "strong.txt",         # Strong definition and transition rules
    "shifting_pressure.txt",  # 4 golden rules of shifting
    "scenarios.txt",      # COA 9 scenarios (Blood Bath, Bull Run, etc.)
    "coa_1_0.txt",        # COA 1.0 cheatsheet
    "imaginary_line.txt", # ATM / imaginary line concept
    "soc.txt",            # State of Confusion
    "ai_ltp.txt",         # Max Pain, EOR, EOS, Reversal Math
    "ltp_features.txt",   # LTP Blast, Swing, Arbitrage
    "game_of_percentage.txt",  # Percentage interpretation
    "trading_terms.txt",  # General trading vocabulary
    "support_resistance_basics.txt",
    "coa_2_0.txt",
    "options_trading.txt",
    "technical_analysis.txt",
    "trading_strategies.txt",
    "weekly_range.txt",
    "about.txt",
    "general_finance.txt",
    "master_index.txt",
]

# Max chars per file (to stay within token budget)
_MAX_CHARS_PRIORITY = 3000
_MAX_CHARS_OTHER = 1200


def _load_all_kb_files() -> str:
    """
    Read all .txt files from the KB directory.
    Returns a single formatted string with all domain knowledge.
    """
    # pybackend/app/agents/vision_reader.py -> pybackend/txt/
    txt_dir = Path(__file__).parent.parent.parent / "txt"

    if not txt_dir.exists():
        logger.warning(f"⚠️ KB txt directory not found: {txt_dir}")
        return ""

    sections = []
    loaded_names = set()

    # 1. Load priority files in order
    for fname in _PRIORITY_KB_FILES:
        fpath = txt_dir / fname
        if not fpath.exists():
            continue
        try:
            text = fpath.read_text(encoding="utf-8").strip()
            if not text:
                continue
            limit = _MAX_CHARS_PRIORITY if fname in _PRIORITY_KB_FILES[:10] else _MAX_CHARS_OTHER
            if len(text) > limit:
                text = text[:limit] + "\n...[truncated for token budget]"
            sections.append(f"### {fname.replace('.txt','').replace('_',' ').upper()} ###\n{text}")
            loaded_names.add(fname)
        except Exception as e:
            logger.warning(f"Could not read {fname}: {e}")

    # 2. Load any remaining files not covered
    for fpath in sorted(txt_dir.glob("*.txt")):
        if fpath.name in loaded_names:
            continue
        try:
            text = fpath.read_text(encoding="utf-8").strip()
            if text:
                if len(text) > _MAX_CHARS_OTHER:
                    text = text[:_MAX_CHARS_OTHER] + "\n...[truncated]"
                sections.append(f"### {fpath.stem.replace('_',' ').upper()} ###\n{text}")
        except Exception as e:
            logger.warning(f"Could not read {fpath.name}: {e}")

    result = "\n\n".join(sections)
    logger.info(f"📚 VisionReader: loaded {len(sections)} KB files ({len(result)} chars total)")
    return result


# Load once at startup — baked into the prompt
_DOMAIN_KB = _load_all_kb_files()


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt() -> str:
    """Build the full vision reading prompt with domain knowledge injected."""

    # Domain knowledge section (only if KB was loaded)
    kb_block = ""
    if _DOMAIN_KB:
        kb_block = f"""
════════════════════════════════════════════════════════
INVESTINGDADDY DOMAIN KNOWLEDGE — MEMORIZE BEFORE READING
════════════════════════════════════════════════════════
This is the complete knowledge base of the InvestingDaddy / LTP Calculator system.
You MUST understand these concepts to correctly read any screenshot from this platform.

{_DOMAIN_KB}

════════════════════════════════════════════════════════
END OF DOMAIN KNOWLEDGE
════════════════════════════════════════════════════════
"""

    return f"""You are a precision financial image reader for the InvestingDaddy / LTP Calculator platform, built by Adarsh under guidance of Vinay Sir (Dr. Vinay Prakash Tiwari).

Your job: Convert financial screenshots into accurate natural language descriptions that another AI agent will use to give expert analysis. EVERY NUMBER MATTERS. EVERY COLOR MATTERS.
{kb_block}
════════════════════════════════════════════════════════
COLOR CODE DICTIONARY (OFFICIAL LTP CALCULATOR COLORS)
════════════════════════════════════════════════════════

THE FIVE OFFICIAL COLORS ON AI LTP CALCULATOR:
These appear in Volume and Open Interest columns. SAME colors, DIFFERENT meanings on each side.

━━━━━━━━━━━━━━━━━━━━━━━━━━━
CALL SIDE (Resistance side):
• PINK   = HIGHEST OI + Highest OI Change → Main RESISTANCE marker
• BLUE   = HIGHEST VOLUME → Volume-based resistance marker
• GREY   = Highest Volume/OI that has become ITM (gone in-the-money, shifted past imaginary line)
• GREEN  = LOWEST OI + Lowest OI Change → Least significant call level
• YELLOW = SECOND HIGHEST value (OI or Volume) → WTB or WTT signal (weakness present!)

PUT SIDE (Support side):
• GREEN  = HIGHEST OI + Highest OI Change → Main SUPPORT marker  ← OPPOSITE of call side!
• BLUE   = HIGHEST VOLUME → Volume-based support marker
• GREY   = Highest Volume/OI that has become ITM (shifted past imaginary line)
• PINK   = LOWEST OI + Lowest OI Change → Least significant put level  ← OPPOSITE of call side!
• YELLOW = SECOND HIGHEST value (OI or Volume) → WTB or WTT signal (weakness present!)

CRITICAL COLOR SUMMARY:
┌────────┬──────────────────────────────┬──────────────────────────────┐
│ Color  │ CALL SIDE Meaning            │ PUT SIDE Meaning             │
├────────┼──────────────────────────────┼──────────────────────────────┤
│ PINK   │ Highest OI (RESISTANCE OI)   │ Lowest OI (least important)  │
│ BLUE   │ Highest Volume (both sides same)           │
│ GREY   │ ITM highest Vol/OI (both sides same — already in-the-money) │
│ GREEN  │ Lowest OI (least important)  │ Highest OI (SUPPORT OI)      │
│ YELLOW │ Second Highest = WTB/WTT indicator (both sides same)        │
└────────┴──────────────────────────────┴──────────────────────────────┘

HOW TO IDENTIFY SUPPORT AND RESISTANCE USING COLORS:
• RESISTANCE → Look for PINK (highest OI) or BLUE (highest Volume) on CALL side, closest to imaginary line
• SUPPORT    → Look for GREEN (highest OI) or BLUE (highest Volume) on PUT side, closest to imaginary line
• WTB/WTT   → YELLOW box in any column = second-highest is ≥75% of highest = weakness detected

OI CHANGE TEXT COLOR (in OI Change column):
• Green text = OI increasing vs last session close = NEW POSITIONS (long buildup)
• Red text   = OI decreasing vs last session close = SQUARING OFF (unwinding)

IMAGINARY LINE (ATM ROW):
• Usually highlighted in a distinct color (yellow/orange band) across the full row
• Separates ITM from OTM. Closest to current market price.
• Calls ABOVE imaginary line = OTM calls | Puts BELOW imaginary line = OTM puts

INTERPRETING WTB% AND WTT% CORRECTLY (from KB):
• WTB (Weak Towards Bottom) = second-highest OI is ≥75% of highest, located at BOTTOM side
  - WTB at Support = BEARISH (support weakening)
  - WTB at Resistance = BEARISH (resistance confirming downward move)
  - WTB % INCREASING over time = bearish strengthening
  - WTB → Strong (% drops below 75%) = BULLISH reversal
• WTT (Weak Towards Top) = second-highest OI is ≥75% of highest, located at TOP side
  - WTT at Resistance = BULLISH (resistance about to break upward)
  - WTT at Support = bullish potential (wants to shift up)
  - WTT % INCREASING over time = bullish strengthening
  - WTT → Strong (% drops below 75%) = BEARISH reversal
• STRONG = second-highest OI is <75% of highest. No weakness present.
  - Strong from WTB = BULLISH | Strong from WTT = BEARISH | Strong from start = NEUTRAL

COA SCENARIO IDENTIFICATION (from numbers you see):
• Sc 1: Resistance Strong + Support Strong → Range EOR to EOS
• Sc 2: Resistance Strong + Support WTB → Bearish (EOR to WTB+1)
• Sc 3: Resistance Strong + Support WTT → Bullish (EOR+1 to EOS)
• Sc 4: Resistance WTB + Support Strong → Bearish (EOS to EOS-1)
• Sc 5: Resistance WTT + Support Strong → Bullish (WTT-1 to EOS)
• Sc 6: Both WTB → BLOOD BATH (free fall)
• Sc 7: Both WTT → BULL RUN (sky rocket)
• Sc 8: Resistance WTT + Support WTB → Expander (volatile, WTT-1 to WTB+1)
• Sc 9: Resistance WTB + Support WTT → Squeeze (tight range, EOR+1 to EOS-1)

SHIFTING (from KB):
• When second-highest OI becomes highest = SHIFT happened
• Shift to TOP (larger strike) = BULLISH
• Shift to BOTTOM (smaller strike) = BEARISH
• Shift + Strong = CONFIRMED trend
• Shift back to original = pressure REVERSES
• Natural Weakness: WTB/WTT at the OLD level after shift = IGNORE (not a signal)

CANDLESTICK CHART COLORS:
• 🟢 Green/white body = Bullish candle (close > open)
• 🔴 Red/black body = Bearish candle (close < open)
• Long upper wick = price rejection at high (sellers)
• Long lower wick = price rejection at low (buyers)
• Doji (tiny body, wicks on both sides) = indecision → potential reversal
• Engulfing green after downtrend = Bullish reversal
• Engulfing red after uptrend = Bearish reversal

INDICATOR COLORS (standard charting):
• Blue line = EMA 20 | Orange line = EMA 50 | Purple/maroon = EMA 200
• Yellow dotted = VWAP
• Gray bands = Bollinger Bands (squeeze = low vol, expansion = breakout)
• RSI: >70 = Overbought | <30 = Oversold | ~50 = Neutral
• MACD line above signal line = Bullish | below = Bearish
• Green volume bar = Bullish volume | Red volume bar = Bearish volume

MARKET HEATMAP COLORS:
• Dark green = strong gainer (>+2%) | Light green = mild gainer (+0.5 to +2%)
• Dark red = strong loser (>-2%) | Light red = mild loser
• Grey/white = flat (< ±0.5%)

LTP CALCULATOR SPECIFIC:
• COA % / WTB% / WTT% columns show percentage values — read the EXACT number
• "Shifting" label on a strike = that level is moving (note direction: up or down)
• EOR = Extension of Resistance | EOS = Extension of Support
• Max Pain = strike where option writers lose least (market tends to expire here)
• C1/C2 in LTP Blast = Bullish triggers | P1/P2 = Bearish triggers
• SOC (State of Confusion): if % stuck same for 1hr+ → market unsure

════════════════════════════════════════════════════════
WHAT TO READ (fill in ALL visible data)
════════════════════════════════════════════════════════

OPTION CHAIN / COA TABLE:
1. Instrument name + CMP (the imaginary line / ATM row, usually yellow)
2. Expiry date if visible
3. Identify RESISTANCE: highest Call OI strike → is it Strong / WTB / WTT? Exact %?
4. Identify SUPPORT: highest Put OI strike → is it Strong / WTB / WTT? Exact %?
5. For each visible strike from -3 to +3 around ATM: Strike | CE LTP | CE OI | CE OI Chg | PE LTP | PE OI | PE OI Chg
6. WTB% and WTT% for resistance and support strikes
7. Any shifting visible? (which level, which direction)
8. Total CE OI vs PE OI → compute PCR
9. State the COA Scenario number based on what you see
10. Natural weakness present? (ignore if at old shifted level)

CANDLESTICK CHART:
1. Instrument + timeframe
2. Last visible close price
3. Resistance level(s): exact price + basis (EMA? Previous high? Round number?)
4. Support level(s): exact price + basis
5. Pattern name (be very specific)
6. Trend direction + evidence (higher highs/lows? Lower highs/lows?)
7. Indicator values: RSI number, EMA alignment, MACD state, VWAP position
8. Volume pattern on recent candles

P&L / HOLDINGS:
1. Total invested, current value, overall P&L (₹ + %)
2. Each position: name, qty, avg buy price, CMP, unrealised P&L

════════════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════════════

Write as a flowing briefing to a senior InvestingDaddy analyst.
Start with: what kind of screenshot this is, then walk through all the numbers.
Name the COA scenario if identifiable. Reference WTB/WTT/Strong correctly.
State whether any shifting is happening (direction).

GOOD EXAMPLE:
"This is a Nifty 50 weekly option chain screenshot for 20-Mar-2025 expiry. The ATM
(imaginary line) appears to be near 23,400 based on the yellow-highlighted row. The
resistance is at the 23,500 CE strike with 18.2L OI — it shows WTB at 23,450 (89.3%,
shown in orange-red), meaning the second-highest OI at the bottom side is 89.3% of
the highest. The support is at 23,300 PE with 15.1L OI and is Strong (second highest
at 23,350 is 62%, below 75%). This is a COA Scenario 2 setup: Resistance Strong +
Support WTB — bearish bias, market likely to test WTB+1 = 23,400 EOS area. PCR =
15.1/18.2 = 0.83, confirming slight bearish bias. No visible shifting yet."

State ONLY what you can see. Use real numbers from the image. Do not speculate.
Do not give trading advice — just describe accurately so another agent can analyze."""


# Build the prompt once at module load (after KB is loaded)
VISION_READ_PROMPT = _build_prompt()
logger.info(f"✅ VisionReader prompt built: {len(VISION_READ_PROMPT)} chars")


# ── Agent ─────────────────────────────────────────────────────────────────────

class VisionReaderAgent:
    """
    Pre-router vision node.
    Reads image(s) with full InvestingDaddy domain context.
    Converts screenshot → NLP text → appended to user query.
    """

    def __init__(self):
        self.client = get_groq_client()

    async def read_and_enrich(self, state: AgentState) -> AgentState:
        """
        If images are present:
          1. Call vision model with domain-aware prompt
          2. Append extracted NLP scenario to the user query
          3. Return enriched state for normal router → agent flow
        No-op if no images present.
        """
        images = state.get("images") or []

        if not images:
            return state  # No images — pass through immediately

        query = state.get("query") or ""
        logger.info(f"🔍 VisionReader: reading {len(images)} image(s)")

        # Build multi-modal user turn
        user_parts = []

        if query.strip():
            user_parts.append({
                "type": "text",
                "text": (
                    f"User's question about this image: \"{query}\"\n\n"
                    "Now read the attached financial image(s) precisely and describe exactly "
                    "what you see — all numbers, colors, WTB/WTT values, scenario context."
                )
            })
        else:
            user_parts.append({
                "type": "text",
                "text": (
                    "Read and describe the attached financial image(s) precisely — "
                    "all numbers, colors, WTB/WTT values, COA scenario if identifiable."
                )
            })

        # Attach images (max 4 to stay within API limits)
        for img in images[:4]:
            url = img if img.startswith("data:") else f"data:image/jpeg;base64,{img}"
            user_parts.append({
                "type": "image_url",
                "image_url": {"url": url}
            })

        try:
            response = await self.client.chat.completions.create(
                model=settings.model_vision,
                messages=[
                    {"role": "system", "content": VISION_READ_PROMPT},
                    {"role": "user", "content": user_parts},
                ],
                temperature=0.05,   # Ultra-low: precise number extraction
                max_tokens=1500
            )

            image_description = (response.choices[0].message.content or "").strip()

            if image_description:
                logger.info(f"✅ VisionReader: extracted {len(image_description)} chars of context")

                if query.strip():
                    enriched_query = (
                        f"{query}\n\n"
                        f"[Image context — extracted by vision model from uploaded screenshot:]\n"
                        f"{image_description}"
                    )
                else:
                    enriched_query = (
                        f"Analyze this financial data from the uploaded image:\n\n"
                        f"{image_description}"
                    )

                state["query"] = enriched_query
                state["image_context_extracted"] = True
                state["raw_image_description"] = image_description
                logger.info("🔀 VisionReader: query enriched → flowing into normal agent routing")

            else:
                logger.warning("⚠️ VisionReader: vision model returned empty description")

        except Exception as e:
            logger.error(f"❌ VisionReader error: {e} — continuing without image context")

        return state
