"""
routers/voice.py — Voice / Speech-to-Text endpoints for JARVIS.
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

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/voice",
    tags=["Voice / STT"]
)


# ─────────────────────────────────────────────────────────────
# RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    language_code: str
    duration: Optional[float]
    char_count: int


class VoiceChatResponse(BaseModel):
    transcript: str
    transcript_language: str
    transcript_language_code: str
    audio_duration: Optional[float]

    reply: str
    conversation_id: str

    used_web_search: bool
    search_query: Optional[str]

    message_count: int


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def normalize_language_code(language: Optional[str]) -> str:
    """
    Convert language names into short language codes.
    """

    if not language:
        return "en"

    language = language.lower()

    mapping = {
        "english": "en",
        "en": "en",

        "hindi": "hi",
        "hi": "hi",

        "hinglish": "hi-en",

        "urdu": "ur",
        "spanish": "es",
        "french": "fr",
    }

    return mapping.get(language, language)


# ─────────────────────────────────────────────────────────────
# /voice/transcribe
# ─────────────────────────────────────────────────────────────

@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
)
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):

    audio_bytes = await file.read()

    content_type = file.content_type or "audio/webm"

    # Validate audio
    try:
        validate_audio_file(
            content_type,
            len(audio_bytes)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # Transcribe
    try:
        result = await transcribe_and_detect(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.webm",
            content_type=content_type,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

    except Exception as e:
        logger.error("Transcription error: %s", e)

        raise HTTPException(
            status_code=502,
            detail=f"Transcription service error: {e}"
        )

    detected_language = result.get("language", "en")

    return TranscriptionResponse(
        text=result.get("text", ""),
        language=detected_language,
        language_code=normalize_language_code(detected_language),
        duration=result.get("duration"),
        char_count=len(result.get("text", "")),
    )


# ─────────────────────────────────────────────────────────────
# /voice/chat
# ─────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=VoiceChatResponse,
)
async def voice_chat(
    file: UploadFile = File(...),

    conversation_id: Optional[str] = Form(None),

    force_search: bool = Form(False),

    db: AsyncSession = Depends(get_db),
):

    audio_bytes = await file.read()

    content_type = file.content_type or "audio/webm"

    # Validate audio
    try:
        validate_audio_file(
            content_type,
            len(audio_bytes)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # Run full voice pipeline
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
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

    except Exception as e:
        logger.error("Pipeline failure: %s", e)

        raise HTTPException(
            status_code=502,
            detail=f"JARVIS internal error: {e}"
        )

    detected_language = getattr(
        result,
        "transcript_language",
        "en"
    )

    return VoiceChatResponse(
        transcript=result.transcript,

        transcript_language=detected_language,

        transcript_language_code=normalize_language_code(
            detected_language
        ),

        audio_duration=result.audio_duration,

        reply=result.reply,

        conversation_id=result.conversation_id,

        used_web_search=result.used_web_search,

        search_query=result.search_query,

        message_count=result.message_count,
    )


# ─────────────────────────────────────────────────────────────
# /voice/formats
# ─────────────────────────────────────────────────────────────

@router.get("/formats")
async def supported_formats():

    return {
        "supported_formats": list(
            set(SUPPORTED_MIME_TYPES.values())
        ),

        "mime_types": list(
            SUPPORTED_MIME_TYPES.keys()
        ),

        "max_file_size_mb": 25,

        "recommended": (
            "webm (browser recording), "
            "mp3, wav, m4a"
        ),

        "languages": {
            "auto": "Auto detect",
            "hi": "Hindi",
            "en": "English",
            "hinglish": "Auto detect Hinglish",
        },

        "model": "whisper-large-v3-turbo",

        "tip": (
            "Use MediaRecorder API "
            "for best browser compatibility."
        ),
    }
