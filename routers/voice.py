"""
routers/voice.py — Voice / Speech-to-Text endpoints for JARVIS.

Endpoints:
  POST /voice/transcribe          — Upload audio → get transcript
  POST /voice/chat                — Upload audio → transcript + JARVIS reply
  GET  /voice/formats             — List supported audio formats

Flow for /voice/chat:
  1. Receive audio file upload
  2. Transcribe with Groq Whisper (fast, multilingual)
  3. Send transcript to JARVIS chat pipeline (Groq LLM + optional web search)
  4. Return both transcript and JARVIS response
"""

import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_db
from services.pipeline_service import run_voice_pipeline
from services.stt_service import (
    transcribe_and_detect,
    validate_audio_file,
    SUPPORTED_MIME_TYPES,
)
from services.memory_service import (
    get_or_create_conversation,
    get_conversation_history,
    save_message,
)
from services.groq_service import chat_completion
from services.search_service import web_search, extract_search_query
from core.personality import should_search_web
from core.context import build_messages
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["Voice / STT"])


# ── Response Schemas ──────────────────────────────────────────────────────────

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    language_code: str
    duration: Optional[float]
    char_count: int


class VoiceChatResponse(BaseModel):
    # STT result
    transcript: str
    transcript_language: str
    transcript_language_code: str
    audio_duration: Optional[float]
    # JARVIS reply
    reply: str
    conversation_id: str
    used_web_search: bool
    search_query: Optional[str]
    message_count: int


# ── POST /voice/transcribe ────────────────────────────────────────────────────

@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe audio to text",
    description=(
        "Upload an audio file and get back the transcribed text. "
        "Supports Hindi, English, Hinglish, and 99 other languages. "
        "Powered by Groq Whisper (whisper-large-v3-turbo)."
    ),
)
async def transcribe(
    file: UploadFile = File(..., description="Audio file (mp3, wav, webm, m4a, ogg, flac)"),
    language: Optional[str] = Form(
        None,
        description="Force a language code (e.g. 'hi' for Hindi, 'en' for English). "
                    "Leave empty for auto-detection."
    ),
):
    """
    Transcribe an uploaded audio file.

    - Auto-detects language if not specified
    - Optimised for Hindi, English, and Hinglish
    - Returns text, detected language, and audio duration
    """
    # Read file bytes
    audio_bytes = await file.read()
    content_type = file.content_type or "audio/webm"

    # Validate
    try:
        validate_audio_file(content_type, len(audio_bytes))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Transcribe
    try:
        result = await transcribe_and_detect(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.webm",
            content_type=content_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Transcription error: %s", e)
        raise HTTPException(status_code=502, detail=f"Transcription service error: {e}")

    return TranscriptionResponse(
        text=result["text"],
        language=result["language"],
        language_code=result["language_code"],
        duration=result["duration"],
        char_count=len(result["text"]),
    )


# ── POST /voice/chat ──────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=VoiceChatResponse,
    summary="Voice chat — audio in, JARVIS reply out",
    description=(
        "The full voice pipeline: upload audio → transcribe → JARVIS responds. "
        "Conversation memory and web search work exactly like the text /chat/ endpoint."
    ),
)
async def voice_chat(
    file: UploadFile = File(..., description="Audio file with Mr Aryan's voice"),
    conversation_id: Optional[str] = Form(
        None,
        description="Existing conversation ID for context continuity. Omit to start new."
    ),
    force_search: bool = Form(
        False,
        description="Force a web search regardless of content."
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Full voice pipeline:
    1. Receive audio upload
    2. Transcribe with Groq Whisper (fast, multilingual)
    3. Feed transcript through JARVIS chat (memory + optional web search)
    4. Return transcript + JARVIS reply
    """
    # ── 1. Read and validate audio ─────────────────────────────────────────
    audio_bytes = await file.read()
    content_type = file.content_type or "audio/webm"

    try:
        validate_audio_file(content_type, len(audio_bytes))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ── 2. Run Unified Pipeline (Optimized for Speed) ──────────────────────
    try:
        result = await run_voice_pipeline(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.webm",
            content_type=content_type,
            db=db,
            conversation_id=conversation_id,
            force_search=force_search,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Pipeline failure: %s", e)
        raise HTTPException(status_code=502, detail=f"JARVIS internal error: {e}")

    return VoiceChatResponse(
        transcript=result.transcript,
        transcript_language=result.transcript_language,
        transcript_language_code=result.transcript_language_code,
        audio_duration=result.audio_duration,
        reply=result.reply,
        conversation_id=result.conversation_id,
        used_web_search=result.used_web_search,
        search_query=result.search_query,
        message_count=result.message_count,
    )


# ── GET /voice/formats ────────────────────────────────────────────────────────

@router.get("/formats", summary="List supported audio formats")
async def supported_formats():
    """Return all supported audio formats and usage tips."""
    return {
        "supported_formats": list(set(SUPPORTED_MIME_TYPES.values())),
        "mime_types": list(SUPPORTED_MIME_TYPES.keys()),
        "max_file_size_mb": 25,
        "recommended": "webm (browser recording), mp3, wav, m4a",
        "languages": {
            "auto":    "Auto-detect (default, works well for Hinglish)",
            "hi":      "Hindi",
            "en":      "English",
            "hinglish": "Auto-detect with Hinglish bias (use 'auto')",
        },
        "model":      "whisper-large-v3-turbo (Groq)",
        "tip": (
            "For voice recording from a browser, use MediaRecorder API "
            "which outputs webm/opus — fully supported."
        ),
    }
