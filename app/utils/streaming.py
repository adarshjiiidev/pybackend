"""
Server-Sent Events (SSE) streaming utilities.
Handles streaming responses from agents to clients.
"""

import asyncio
import json
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)


async def stream_groq_response(response_text: str, chunk_size: int = 50) -> AsyncGenerator[str, None]:
    """
    Stream response text in chunks (simulated streaming).
    For true streaming, integrate with Groq's streaming API.
    
    Args:
        response_text: Complete response text
        chunk_size: Number of characters per chunk
    
    Yields:
        SSE formatted chunks
    """
    # Split by words to avoid breaking words
    words = response_text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for space
        
        if current_length >= chunk_size:
            chunk_text = " ".join(current_chunk)
            # Format as SSE
            yield f"data: {json.dumps({'content': chunk_text, 'done': False})}\n\n"
            await asyncio.sleep(0.01)  # Small delay for smooth streaming
            current_chunk = []
            current_length = 0
    
    # Send remaining chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        yield f"data: {json.dumps({'content': chunk_text, 'done': False})}\n\n"
    
    # Send final done message
    yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"


async def format_sse_message(content: str, done: bool = False, metadata: dict = None) -> str:
    """
    Format a message as Server-Sent Event.
    
    Args:
        content: Message content
        done: Whether this is the final message
        metadata: Additional metadata to include
    
    Returns:
        SSE formatted string
    """
    message = {
        "content": content,
        "done": done
    }
    
    if metadata:
        message["metadata"] = metadata
    
    return f"data: {json.dumps(message)}\n\n"
