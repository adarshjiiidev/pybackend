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
        
        # If images present, ALWAYS analyze them (whether response exists or not)
        if images:
            logger.info(f"🖼️ Analyzing {len(images)} image(s) with vision model")
            image_analysis = await self._analyze_images(state)
            
            # If we have both image analysis and existing response, combine them
            if raw_response:
                # Prepend image analysis to existing response
                raw_response = f"{image_analysis}\n\n---\n\n{raw_response}"
                logger.info("Combined image analysis with existing response")
            else:
                # Use image analysis as the response
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
        
        # Now format the main response with structured headings
        try:
            logger.info(f"Attempting to format response of length: {len(cleaned)}")
            
            # Use LLM to add proper markdown formatting
            formatted_response = await self._format_with_structure(cleaned, state)
            
            # Validate formatted response
            if not formatted_response or not formatted_response.strip():
                logger.warning("Formatting returned empty response, using cleaned version")
                formatted_response = cleaned
            
            # Combine reasoning and response
            if reasoning_text:
                final_output = f"<reasoning>{reasoning_text}</reasoning>\n\n{formatted_response}"
            else:
                final_output = formatted_response
            
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
        
        formatting_prompt = f"""You are a formatting assistant. Your job is to take an AI response and format it with clear structure using markdown.

FORMATTING RULES:
1. **Section Headings**: Use ## for main sections (e.g., ## Direct Answer, ## What Is Needed for a Complete Analysis)
2. **Bold**: Use **bold** for important terms, key concepts, and emphasis
3. **Bullet Points**: Use - for bullet points, with sub-bullets indented
4. **Italics**: Use *italics* for technical terms or examples in context
5. Keep the content DATA-DRIVEN and PROFESSIONAL
6. Do NOT add pleasantries or fluffy language
7. Structure should be: Main answer first, then supporting details
8. Each section should have a clear, descriptive heading

Original Query: {query}

Response to Format:
{response}

Return the formatted response with proper ## headings, **bold** emphasis, and - bullet points. Keep all original data and insights intact."""

        try:
            format_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a markdown formatting expert. Format responses with clear structure using ## headings, **bold**, and bullet points."},
                    {"role": "user", "content": formatting_prompt}
                ],
                temperature=self.temperature,
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
