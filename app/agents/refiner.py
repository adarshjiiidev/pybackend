"""
Refiner Agent - Synthesizes research into final answer.
Removes noise, structures findings, and adds risk assessment.
"""

from groq import AsyncGroq
import logging
import json
from datetime import datetime

from ..config import settings, ModelType
try:
    from ..config.openrouter_client import get_openrouter_client
    _HAS_OPENROUTER = True
except ImportError:
    _HAS_OPENROUTER = False

from ..config.key_rotator import get_groq_client
from ..models.research_state import ResearchState

logger = logging.getLogger(__name__)


class RefinerAgent:
    """
    Final synthesis of all gathered research.
    Produces structured, confidence-scored answer.
    """

    def __init__(self):
        # OpenRouter first priority, Groq fallback
        if _HAS_OPENROUTER and settings.openrouter_available:
            from ..config.openrouter_client import get_openrouter_client as _get_or
            self.client = _get_or()
            self._provider = "openrouter"
        else:
            self.client = get_groq_client()
            self._provider = "groq"
        if hasattr(self, '_provider') and self._provider == "openrouter":
            self.model = settings.get_openrouter_model(ModelType.REASONING_DEEP)
        else:
            self.model = settings.get_model_for_task(ModelType.REASONING_DEEP)
        self.temperature = 0.6
        self.max_tokens = 4096

    async def refine(self, state: ResearchState) -> ResearchState:
        """
        Synthesize all gathered data into final answer.
        """
        query = state["query"]
        gathered_data = state["gathered_data"]
        confidence_score = state["confidence_score"]

        # Build context from gathered data
        context = self._build_context(gathered_data)

        system_prompt = """You are Daddy's AI — synthesizing research into the final answer for an Indian investor.

=== YOUR WRITING STYLE ===
Write like you're explaining to a curious, smart friend who loves learning but hates jargon.
- Tell a STORY. Lead with the most striking insight, not the methodology.
- Use flowing paragraphs, NOT bullet lists. Bullets kill the narrative.
- Build curiosity: "Here's what most people miss about this..." "The real story is..."
- Be specific: "₹22,340" not "around 22,000". Indian context always: ₹, lakhs, crores.
- Tables for data comparison. Carousel blocks for step-by-step concepts.

=== MANDATORY STRUCTURE ===

## [Punchy headline capturing the #1 insight] [emoji]

[Opening paragraph: Lead with the most important finding. Make it compelling.
Don't say "Based on our analysis" — just say it. 2-4 sentences.]

## [Why This Is Happening] [emoji]

[Explanation paragraph: Connect the dots. What's really driving this?
Use an analogy if it helps. "Think of it like..." 2-3 sentences.]

## 📊 The Numbers

[Use a markdown table for key metrics:]
| Metric | Value | Meaning |
|--------|-------|---------|
| ...    | ...   | ...     |

## [What Happens Next] [emoji]

[Forward-looking paragraph: Bull case AND bear case. Give your actual view.
Be direct. Don't hedge everything into meaninglessness.]

## ⚠️ The Fine Print

[2-sentence honest assessment: What data was shaky? What could change this?
End with: "⚠️ *Not financial advice. Consult a SEBI-registered advisor.*"]

=== RULES ===
- Paragraphs > bullets. Always weave data INTO sentences.
- Tables for structured comparisons. Never bullet a table.
- Emojis on headings make it scannable. 2-4 total, natural.
- No "In conclusion" or "Hope this helps" — just end strongly."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"""Research Context:
{context}

Original Query: {query}

Confidence Score: {confidence_score:.2f}

Synthesize findings into a rich, narrative final answer."""
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            final_answer = response.choices[0].message.content

            state["final_answer"] = final_answer
            state["key_findings"] = self._extract_findings(final_answer)
            state["risks_uncertainties"] = self._extract_risks(final_answer)
            state["data_freshness_indicator"] = self._determine_freshness(gathered_data)
            state["completed_at"] = datetime.utcnow()

            logger.info("Research synthesis completed")
            return state

        except Exception as e:
            logger.error(f"Refiner error: {e}")
            state["final_answer"] = f"Research completed with {len(gathered_data)} data points, but synthesis failed."
            return state

    def _build_context(self, data) -> str:
        """Build context string from gathered data."""
        context_parts = []
        for item in data:
            step_num = item.get("step_number", "?")
            question = item.get("question", "")
            source = item.get("source", "")
            result = item.get("data", {})

            context_parts.append(f"Step {step_num}: {question} (from {source})")
            context_parts.append(str(result)[:300])
            context_parts.append("---")

        return "\n".join(context_parts)

    def _extract_findings(self, text: str) -> list:
        """Extract key findings from answer."""
        if "Key Findings" in text:
            section = text.split("Key Findings")[1].split("##")[0]
            return [line.strip() for line in section.split("\n") if line.strip().startswith(("-", "*", "1.", "2.", "3."))]
        return []

    def _extract_risks(self, text: str) -> list:
        """Extract risks from answer."""
        if "Risks" in text or "Uncertainties" in text or "Fine Print" in text:
            key = "Fine Print" if "Fine Print" in text else ("Risks" if "Risks" in text else "Uncertainties")
            section = text.split(key)[1].split("##")[0]
            return [line.strip() for line in section.split("\n") if line.strip()]
        return []

    def _determine_freshness(self, data) -> str:
        """Determine overall data freshness."""
        if not data:
            return "Unknown"

        now = datetime.utcnow()
        most_recent = None

        for item in data:
            try:
                timestamp = datetime.fromisoformat(item.get("timestamp", now.isoformat()))
                if most_recent is None or timestamp > most_recent:
                    most_recent = timestamp
            except Exception:
                continue

        if most_recent:
            age_hours = (now - most_recent).total_seconds() / 3600
            if age_hours < 1:
                return "Real-time"
            elif age_hours < 24:
                return "Recent (today)"
            elif age_hours < 168:
                return "This week"
            else:
                return "Dated"

        return "Unknown"
