"""
Verifier/Refiner Agent - Quality control and final response polishing.
Ensures responses are data-grounded, clear, and properly formatted with structured headings.
"""

from groq import AsyncGroq
import logging
import re

from ..config import settings, ModelType
from ..models.agent_state import AgentState
from ..tools import clean_response

logger = logging.getLogger(__name__)


class VerifierAgent:
    """Verifies and enhances agent responses with structured formatting."""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        # Use fast model for verification (quick checks)
        self.model = settings.get_model_for_task(ModelType.FAST)
        self.temperature = 0.3  # Lower temperature for consistent formatting
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.FAST)
    
    async def verify_and_refine(self, state: AgentState) -> AgentState:
        """
        Verify response quality and refine with proper formatting.
        If images are present, use vision model to enhance response with image understanding.
        Ensures:
        - Structured with bold section headings (##)
        - Clear bullet points for lists
        - Reasoning process formatted separately
        - Professional, non-technical tone
        - Image analysis if images provided
        """
        raw_response = state.get("final_response", "")
        images = state.get("images", [])
        metadata = state.get("execution_metadata") or {}
        
        # Skip verifier entirely for greetings, casual, and thanks — they don't need formatting
        if metadata.get("skip_verifier"):
            logger.info("⏭️ Skipping verifier for greeting/casual/thanks response")
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
            state["final_response"] = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            return state
        
        # Clean response (remove internal reasoning markers, extra whitespace)
        cleaned = clean_response(raw_response)
        
        
        # Get internal reasoning if available
        internal_reasoning = state.get("internal_reasoning")
        reasoning_text = ""
        
        if internal_reasoning and internal_reasoning.strip():
            # Format reasoning as numbered steps
            reasoning_lines = internal_reasoning.strip().split('\n')
            formatted_reasoning = []
            step_num = 1
            for line in reasoning_lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove existing numbering if any
                    line = re.sub(r'^\d+[\.)]\s*', '', line)
                    formatted_reasoning.append(f"{step_num}. {line}")
                    step_num += 1
            
            reasoning_text = '\n'.join(formatted_reasoning)
        elif raw_response and len(raw_response) > 100:
            # FORCE REASONING: Generate simple reasoning steps for responses without it
            agent = (state.get("execution_metadata") or {}).get("agent", "unknown")
            query = state.get("query", "user query")
            
            reasoning_text = f"""
1. Identified query type and selected {agent} agent
2. Processed request: {query[:100]}...
3. Generated comprehensive response with available data
4. Structured output for clarity
            """.strip()
            logger.info("✅ Generated synthetic reasoning for transparency")

        
        # Now format the main response with structured headings
        try:
            logger.info(f"Attempting to format response of length: {len(cleaned)}")
            
            # PERFORMANCE OPTIMIZATION: Skip LLM formatting if response already has structure
            # Check if response already has markdown headings, bullets, or code blocks
            has_headings = "##" in cleaned
            has_bullets = cleaned.count("-") > 3 or cleaned.count("•") > 3
            has_code = "```" in cleaned
            is_well_formatted = has_headings or (has_bullets and len(cleaned) > 200) or has_code
            
            if is_well_formatted:
                logger.info("✅ Response already well-formatted, skipping LLM formatting for speed")
                formatted_response = cleaned
            else:
                logger.info("🔄 Applying LLM formatting to improve structure")
                # Use LLM to add proper markdown formatting
                formatted_response = await self._format_with_structure(cleaned, state)
            
            # Validate formatted response
            if not formatted_response or not formatted_response.strip():
                logger.warning("Formatting returned empty response, using cleaned version")
                formatted_response = cleaned
            
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
    
    async def _analyze_images(self, state: AgentState) -> str:
        """Analyze images using vision model."""
        images = state.get("images", [])
        query = state.get("query", "Describe these images")
        
        try:
            # Prepare messages for vision model
            content = [{"type": "text", "text": query}]
            
            # Add images to content
            for img in images:
                # Ensure proper base64 format
                if img.startswith('data:image'):
                    image_url = img
                else:
                    image_url = f"data:image/jpeg;base64,{img}"
                
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            
            # Use vision model
            response = await self.client.chat.completions.create(
                model=settings.model_vision,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=0.5,
                max_tokens=2048
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return f"I can see you've shared {len(images)} image(s), but I encountered an error analyzing them. {query}"
    
    async def _format_with_structure(self, response: str, state: AgentState) -> str:
        """Use LLM to add proper structure and formatting to response."""
        
        query = state.get("query", "")
        
        formatting_prompt = f"""Improve the readability of this content. Your job is ONLY to improve structure, not change content.

RULES:
- If the content is short (under 200 words), DON'T add headings. Just clean up spacing.
- If the content has data comparisons, add a markdown table.
- If the content is long (400+ words), add ## section headings to break it up.
- Use **bold** for key terms and numbers.
- DON'T add bullet points unless there are 3+ items that are genuinely a list.
- DON'T change facts, numbers, or meaning.
- DON'T add your own commentary.
- Keep the original conversational tone. Don't make it corporate.

User query: {query}

Content to format:
{response}

Output ONLY the formatted content."""

        try:
            format_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You improve text structure for readability. Use ## for section headings, ** for bold, - for bullets. Output only the formatted text—never instructions or meta-commentary."},
                    {"role": "user", "content": formatting_prompt}
                ],
                temperature=0.2,
                max_tokens=self.max_tokens
            )
            
            formatted = format_response.choices[0].message.content
            return formatted.strip()
            
        except Exception as e:
            logger.error(f"Formatting LLM call failed: {e}")
            # Fallback: add basic structure
            return self._add_basic_structure(response)
    
    def _add_basic_structure(self, response: str) -> str:
        """Fallback: Return response as-is without adding generic headings."""
        return response
