"""
Audio transcription API endpoint using Groq Whisper.
Supports fast speech-to-text conversion for voice input.
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from groq import Groq

from ..config.settings import settings

router = APIRouter(prefix="/audio", tags=["audio"])
logger = logging.getLogger(__name__)

# Initialize Groq client
groq_client = Groq(api_key=settings.groq_api_key)


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default="en")
):
    """
    Transcribe audio file to text using Groq Whisper API.
    
    Supports: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
    Max file size: 25MB (free tier), 100MB (dev tier)
    """
    try:
        # Validate file type
        allowed_types = [
            "audio/flac", "audio/mp3", "audio/mp4", "audio/mpeg", 
            "audio/mpga", "audio/m4a", "audio/ogg", "audio/wav", "audio/webm",
            "audio/x-m4a", "audio/x-wav"
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Supported: {', '.join(allowed_types)}"
            )
        
        # Read audio file
        audio_data = await file.read()
        
        # Check file size (25MB limit for free tier)
        max_size = 25 * 1024 * 1024  # 25MB in bytes
        if len(audio_data) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is 25MB, got {len(audio_data) / 1024 / 1024:.2f}MB"
            )
        
        logger.info(f"Transcribing audio file: {file.filename} ({len(audio_data) / 1024:.2f}KB)")
        
        # Create transcription using Groq Whisper
        transcription = groq_client.audio.transcriptions.create(
            file=(file.filename, audio_data),
            model="whisper-large-v3-turbo",
            language=language if language else None,  # Auto-detect if not specified
            response_format="json",
            temperature=0.0
        )
        
        logger.info(f"Successfully transcribed audio: {transcription.text[:100]}...")
        
        return {
            "text": transcription.text,
            "language": language,
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
