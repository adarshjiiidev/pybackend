"""
Refiner Agent - Synthesizes research into final answer.
Removes noise, structures findings, and adds risk assessment.
"""

from groq import AsyncGroq
import logging
import json
from datetime import datetime

from ..config import settings, ModelType
from ..config.key_rotator import get_groq_client
from ..models.research_state import ResearchState

logger = logging.getLogger(__name__)


class RefinerAgent:
    """
    Final synthesis of all gathered research.
    Produces structured, confidence-scored answer.
    """
    
    def __init__(self):
        self.client = get_groq_client()
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
        
        system_prompt = """You are synthesizing research into a final answer. Imagine you're writing a report for someone who asked a question - they want the answer first, then the supporting details.

=== STEP 1: GATHER WHAT WE FOUND ===
Look at all the research steps we completed. What data did we get? What are the key numbers, facts, or insights?

=== STEP 2: STRUCTURE THE ANSWER ===

PART A - Key Findings (bullet points)
- List 3-5 main takeaways. One line each.
- Use - for bullets
- Be specific: "TCS PE is 28x vs Infosys 24x" not "TCS and Infosys have different valuations"

PART B - Analysis (2-3 paragraphs)
- Expand on the findings. What do they mean?
- Connect the dots. "This suggests..." or "The data shows..."
- Be direct. No fluff.

PART C - Risks & Uncertainties
- What could we be wrong about?
- What data was missing or old?
- "Market conditions may change" - that kind of thing

PART D - Data Freshness
- When was our data from? Say one of: "Real-time" | "Recent (today)" | "Dated"
- Real-time = fetched in last hour
- Recent = today
- Dated = older than 24 hours

=== STEP 3: OUTPUT RULES ===
- Use the confidence score we calculated - mention it: "Confidence: 0.75"
- Ground everything in the research - cite "Based on our analysis of..."
- Use ₹ for Indian currency
- Be professional. No "Hope this helps!" """

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

Synthesize findings into final answer."""
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            final_answer = response.choices[0].message.content
            
            # Extract structured elements (simplified - could use JSON mode)
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
            context_parts.append(str(result)[:300])  # Truncate
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def _extract_findings(self, text: str) -> list:
        """Extract key findings from answer."""
        # Simplified - look for bullet points or numbered lists
        if "Key Findings" in text:
            section = text.split("Key Findings")[1].split("##")[0]
            return [line.strip() for line in section.split("\n") if line.strip().startswith(("-", "*", "1.", "2.", "3."))]
        return []
    
    def _extract_risks(self, text: str) -> list:
        """Extract risks from answer."""
        if "Risks" in text or "Uncertainties" in text:
            section = text.split("Risks")[1].split("##")[0] if "Risks" in text else text.split("Uncertainties")[1].split("##")[0]
            return [line.strip() for line in section.split("\n") if line.strip().startswith(("-", "*"))]
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
            except:
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
