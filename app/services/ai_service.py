"""
AI service for handling chat completions using Groq API.
"""

from groq import Groq
from app.config.settings import settings
import logging
from typing import AsyncGenerator, List, Dict

logger = logging.getLogger(__name__)


class AIService:
    """Service for AI chat completions."""
    
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = "llama-3.3-70b-versatile"
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True
    ):
        """
        Generate AI response for given messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
        
        Yields:
            Chunks of response text if streaming, otherwise returns complete response
        """
        try:
            if stream:
                # Streaming response
                stream_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=4096,
                )
                
                for chunk in stream_response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                # Non-streaming response
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096,
                )
                yield response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            yield f"Error: {str(e)}"
    
    async def generate_title(self, first_message: str) -> str:
        """
        Generate a conversation title based on the first message.
        
        Args:
            first_message: The first user message
            
        Returns:
            Generated title (max 50 chars)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a short, concise title (max 6 words) for a conversation that starts with the following message. Only return the title, nothing else."
                    },
                    {
                        "role": "user",
                        "content": first_message
                    }
                ],
                temperature=0.5,
                max_tokens=20,
            )
            
            title = response.choices[0].message.content.strip()
            # Limit to 50 chars
            return title[:50] if len(title) > 50 else title
            
        except Exception as e:
            logger.error(f"Error generating title: {str(e)}")
            return "New Chat"


# Global AI service instance
ai_service = AIService()
