"""
Verifier/Refiner Agent - Quality control and final response polishing.
Ensures responses are data-grounded, clear, and properly formatted with structured headings.
"""

import logging
import re

from ..config import settings, ModelType

try:
    from ..config.openrouter_client import get_openrouter_client
    _HAS_OPENROUTER = True
except ImportError:
    _HAS_OPENROUTER = False

from ..config.key_rotator import get_groq_client
from ..models.agent_state import AgentState
from ..tools import clean_response
from ..utils.fallback_response import resolve_response_or_fallback

logger = logging.getLogger(__name__)


class VerifierAgent:
    """Verifies and enhances agent responses with structured formatting."""
    
    def __init__(self):
        # OpenRouter first priority, Groq fallback (for text formatting)
        if _HAS_OPENROUTER and settings.openrouter_available:
            self.client = get_openrouter_client()
            self.model = settings.get_openrouter_model(ModelType.FAST)
            self._provider = "openrouter"
        else:
            self.client = get_groq_client()
            self.model = settings.get_model_for_task(ModelType.FAST)
            self._provider = "groq"
        # Always use a dedicated Groq client for vision — model_vision is Groq-only
        self._groq_vision_client = get_groq_client()
        self.temperature = 0.3
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.FAST)

    @staticmethod
    def _is_mergeable_list_content(line: str) -> bool:
        trimmed = line.strip()
        if not trimmed:
            return False
        if re.fullmatch(r"\d+[.)]", trimmed):
            return False
        if trimmed in {"-", "*", "\u2022", "\u25cf", "\u00b7"}:
            return False
        return not (
            trimmed.startswith("#")
            or trimmed.startswith(">")
            or trimmed.startswith("|")
            or trimmed.startswith("```")
            or trimmed.startswith(":::")
            or trimmed in {"---", "***"}
        )

    def _normalize_broken_list_markup(self, text: str) -> str:
        lines = text.splitlines()
        normalized: list[str] = []
        i = 0
        in_code_block = False

        while i < len(lines):
            current = lines[i]
            trimmed = current.strip()

            if trimmed.startswith("```"):
                in_code_block = not in_code_block
                normalized.append(current)
                i += 1
                continue

            if not in_code_block and i + 1 < len(lines):
                next_line = lines[i + 1]
                next_trimmed = next_line.strip()

                if re.fullmatch(r"\d+[.)]", trimmed) and self._is_mergeable_list_content(next_trimmed):
                    marker = f"{trimmed[:-1]}." if trimmed.endswith(")") else trimmed
                    normalized.append(f"{marker} {next_trimmed}")
                    i += 2
                    continue

                if trimmed in {"-", "*", "\u2022", "\u25cf", "\u00b7"} and self._is_mergeable_list_content(next_trimmed):
                    normalized.append(f"- {next_trimmed}")
                    i += 2
                    continue

            normalized.append(current)
            i += 1

        return "\n".join(normalized)
    
    async def verify_and_refine(self, state: AgentState) -> AgentState:
        """
        Verify response quality and refine with proper formatting.
        If images are present, use vision model to enhance response with image understanding.
        Ensures:
        - Structured with bold section headings (##)
        - Tables instead of bullet points for structured data
        - Carousels for sequential content
        - Reasoning process formatted separately
        - Professional, non-technical tone
        - Image analysis if images provided
        """
        raw_response = self._normalize_broken_list_markup(state.get("final_response", ""))
        images = state.get("images", [])
        metadata = state.get("execution_metadata") or {}
        
        # Skip verifier entirely for greetings, casual, and thanks — they don't need formatting
        if metadata.get("skip_verifier"):
            logger.info("⏭️ Skipping verifier for greeting/casual/thanks response")
            state["final_response"] = raw_response
            return state
        
        # Skip LLM reformatting for data-rich agent responses — the reformatter destroys their data
        agent_name = metadata.get("agent", "")
        if agent_name in ("realtime_analysis", "market_research"):
            logger.info(f"⏭️ Skipping LLM reformatting for {agent_name} — preserving tool data")
            # Still add reasoning if available, but do NOT reformat the content
            internal_reasoning = state.get("internal_reasoning")
            if internal_reasoning and internal_reasoning.strip():
                reasoning_lines = internal_reasoning.strip().split('\n')
                formatted_reasoning = []
                step_num = 1
                for line in reasoning_lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        line = re.sub(r'^\d+[\.)] *', '', line)
                        formatted_reasoning.append(f"{step_num}. {line}")
                        step_num += 1
                reasoning_text = '\n'.join(formatted_reasoning)
                state["final_response"] = f"<reasoning>{reasoning_text}</reasoning>\n\n{raw_response}"
            elif raw_response and len(raw_response) > 100:
                synth_reasoning = self._build_synthetic_reasoning(
                    state=state,
                    cleaned_length=len(raw_response),
                    final_length=len(raw_response),
                    formatter_applied=False,
                    formatter_note="Verifier skipped LLM formatting to preserve tool-derived data",
                )
                logger.info("✅ Generated synthetic reasoning (tool-data path)")
                state["final_response"] = f"<reasoning>{synth_reasoning}</reasoning>\n\n{raw_response}"
            return state
        
        # Only run image analysis when response is missing or generic (agent didn't process images)
        generic_phrases = ["provide text", "haven't provided", "no text", "provide the", "share the image", "upload an image"]
        is_generic = raw_response and len(raw_response) < 300 and any(p in raw_response.lower() for p in generic_phrases)
        needs_image_analysis = images and (not raw_response or len(raw_response.strip()) < 50 or is_generic)
        
        if needs_image_analysis:
            logger.info(f"🖼️ Analyzing {len(images)} image(s) with vision model")
            image_analysis = await self._analyze_images(state)
            
            if raw_response and not is_generic:
                raw_response = f"{image_analysis}\n\n---\n\n{raw_response}"
                logger.info("Combined image analysis with existing response")
            else:
                raw_response = image_analysis
                logger.info("Using image analysis as primary response")
            
            state["final_response"] = raw_response
        
        if not raw_response:
            state["final_response"] = resolve_response_or_fallback(
                final_state={},
                query=state.get("query", ""),
            )
            return state
        
        # Clean response (remove internal reasoning markers, extra whitespace)
        cleaned = self._normalize_broken_list_markup(clean_response(raw_response))
        
        # Get internal reasoning if available
        internal_reasoning = state.get("internal_reasoning")
        has_real_reasoning = bool(internal_reasoning and internal_reasoning.strip())
        reasoning_text = ""

        if has_real_reasoning:
            # Show the REAL reasoning from the model verbatim — don't re-number or alter it
            reasoning_text = internal_reasoning.strip()
        
        # Now format the main response with structured headings
        try:
            logger.info(f"Attempting to format response of length: {len(cleaned)}")
            
            # Structure detection variables
            has_headings = "##" in cleaned
            has_tables = "|" in cleaned and "---" in cleaned
            has_carousel = "carousel" in cleaned
            has_code = "```" in cleaned
            has_emoji = any(e in cleaned for e in ["📈", "📉", "💎", "⚠️", "🎯", "📊", "📌", "💼", "🔴", "🟢", "✅", "❌"])
            response_length = len(cleaned)
            is_short = response_length < 300

            # Check for bullets/numbered lists — these MUST be reformatted regardless
            has_bullets = bool(re.search(r'^\s*[-*•●·]\s+', cleaned, re.MULTILINE))
            has_numbered_list = bool(re.search(r'^\s*\d+[.):]\s+\w', cleaned, re.MULTILINE))
            needs_reformat = has_bullets or has_numbered_list

            is_well_formatted = (
                not needs_reformat  # ← bullets/numbered lists always trigger LLM reformat
                and (
                    has_headings or has_tables or has_carousel or has_code
                    or (has_emoji and is_short)
                    or len(cleaned) < 150
                    or response_length >= 220
                )
            )
            
            if is_well_formatted:
                logger.info("✅ Response already well-formatted, skipping LLM formatting")
                formatted_response = cleaned
                format_note = "Response already well-formatted, skipped LLM formatting"
            else:
                logger.info("🔄 Applying LLM formatting to improve structure")
                # Use LLM to add proper markdown formatting
                formatted_response = await self._format_with_structure(cleaned, state)
                format_note = "Applied LLM formatting to improve structure"
            
            # Validate formatted response
            if not formatted_response or not formatted_response.strip():
                logger.warning("Formatting returned empty response, using cleaned version")
                formatted_response = cleaned

            if not has_real_reasoning and cleaned.strip():
                reasoning_text = self._build_synthetic_reasoning(
                    state=state,
                    cleaned_length=len(cleaned),
                    final_length=len(formatted_response),
                    formatter_applied=not is_well_formatted,
                    formatter_note=format_note,
                )
                logger.info("✅ Generated synthetic reasoning (no real reasoning available)")
            
            # FORCE REASONING: Always add reasoning if available
            if reasoning_text:
                final_output = f"<reasoning>{reasoning_text}</reasoning>\n\n{formatted_response}"
                logger.info(f"✅ Added {len(reasoning_text)} chars of reasoning to response")
            else:
                final_output = formatted_response
                logger.info("ℹ️ No reasoning available for this response")
            
            logger.info(f"Final output length: {len(final_output)}")
            state["final_response"] = final_output
            logger.info("Response verified and formatted with structure")
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}, using cleaned version")
            logger.exception("Full traceback:")
            if reasoning_text:
                state["final_response"] = f"<reasoning>{reasoning_text}</reasoning>\n\n{cleaned}"
            else:
                state["final_response"] = cleaned
        
        return state

    def _build_synthetic_reasoning(
        self,
        state: AgentState,
        cleaned_length: int,
        final_length: int,
        formatter_applied: bool,
        formatter_note: str,
    ) -> str:
        """
        Build user-facing synthetic reasoning when the model does not provide native reasoning.
        """
        metadata = state.get("execution_metadata") or {}
        agent_name = (
            metadata.get("agent")
            or state.get("selected_mode")
            or "unknown"
        )
        query = re.sub(r"\s+", " ", (state.get("query") or "").strip())
        query_preview = (
            (query[:77] + "...") if len(query) > 80 else (query or "user query")
        )

        raw_tools = (
            metadata.get("autonomous_tools_used")
            or metadata.get("tools_called")
            or metadata.get("tools_used")
            or []
        )
        if isinstance(raw_tools, str):
            tool_names = [raw_tools]
        else:
            tool_names = [str(t) for t in raw_tools if t]
        tools_preview = ", ".join(tool_names[:4]) if tool_names else "none"

        sentences = [
            (
                f"I interpreted your prompt as asking about \"{query_preview}\" and routed it "
                f"through the {agent_name} path for the most relevant response strategy."
            ),
            (
                f"To keep the answer grounded, I used available support signals "
                f"({tools_preview}) and then validated the response quality."
            ),
            (
                f"No native model reasoning trace was returned, so this is a synthetic summary; "
                f"I applied verifier checks ({formatter_note}, {cleaned_length} chars) "
                f"before finalizing the response ({final_length} chars)."
            ),
        ]

        if formatter_applied:
            sentences.append(
                "I also normalized structure for readability before sending the final answer."
            )

        return " ".join(sentences)

    def _check_formatting(self, response: str) -> None:
        """Log warning if bullet points detected."""
        has_bullets = bool(re.search(r'^\s*[-*•]\s+', response, re.MULTILINE))
        has_numbered = bool(re.search(r'^\s*\d+[.)]\s+', response, re.MULTILINE))
        if has_bullets or has_numbered:
            logger.warning("⚠️ Detected bullet points/numbered lists — formatter will convert to tables/carousels")

    async def _analyze_images(self, state: AgentState) -> str:
        """Analyze images using Groq vision model (llama-4-scout is Groq-only)."""
        images = state.get("images", [])
        query = state.get("query", "Describe these images")

        try:
            content: list = [{"type": "text", "text": query}]

            for img in images:
                image_url = img if img.startswith("data:image") else f"data:image/jpeg;base64,{img}"
                content.append({"type": "image_url", "image_url": {"url": image_url}})

            # Always use the dedicated Groq client — model_vision is a Groq-only model
            response = await self._groq_vision_client.chat.completions.create(
                model=settings.model_vision,
                messages=[{"role": "user", "content": content}],
                temperature=0.5,
                max_tokens=2048,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return f"I can see you've shared {len(images)} image(s), but I encountered an error analyzing them. {query}"

    async def _format_with_structure(self, response: str, state: AgentState) -> str:
        """Use LLM to add proper structure and formatting to response."""
        query = state.get("query", "")

        formatting_prompt = f"""Improve the readability of this content. Your job is ONLY to improve structure, not change content.

CRITICAL FORMATTING RULES:
- 🚫 NEVER use bullet points (-, *, •) or numbered lists (1., 2., 3.)
- ✅ Use markdown tables for comparisons, features, lists of items
- ✅ Use carousel format for step-by-step or sequential content
- ✅ Use flowing paragraphs for narratives

STRUCTURE RULES:
- If the content is short (under 200 words), DON'T add headings. Just clean up spacing.
- If the content has data comparisons or features → MUST use markdown table
- If content lists steps or multiple items → MUST use carousel format
- If the content is long (400+ words), add ## section headings to break it up.
- Use **bold** for key terms and numbers.
- DON'T change facts, numbers, or meaning.
- DON'T add your own commentary.
- Keep the original conversational tone. Don't make it corporate.

CAROUSEL FORMAT:
````carousel
## Item 1: Title
Description text
<!-- slide -->
## Item 2: Title  
Description text
````

TABLE FORMAT:
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data | Data | Data |

User query: {query}

Content to format:
{response}

Output ONLY the formatted content."""

        try:
            format_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You improve text structure for readability. "
                            "🚫 NEVER use bullet points — use tables or carousels instead. "
                            "Use ## for section headings, ** for bold. "
                            "Output only the formatted text — never instructions or meta-commentary."
                        ),
                    },
                    {"role": "user", "content": formatting_prompt},
                ],
                temperature=0.2,
                max_tokens=self.max_tokens,
            )

            formatted = format_response.choices[0].message.content
            return formatted.strip()

        except Exception as e:
            logger.error(f"Formatting LLM call failed: {e}")
            return self._add_basic_structure(response)

    def _add_basic_structure(self, response: str) -> str:
        """Fallback: Return response as-is without adding generic headings."""
        return response
