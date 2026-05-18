"""
routers/conversation.py — Unified voice conversation pipeline for JARVIS.
FIXED VERSION (Render safe + Pydantic stable)
"""

import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_db
from services.pipeline_service import run_voice_pipeline, run_text_pipeline
from services.stt_service import validate_audio_file
from services.tts_service import synthesize_stream

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversation", tags=["Conversation Pipeline"])


# ─────────────────────────────────────────────
# RESPONSE MODEL (FIXED: timing_ms required)
# ─────────────────────────────────────────────
class PipelineResponse(BaseModel):
    input_type: str
    transcript: str
    transcript_language: str
    transcript_language_code: str
    audio_duration: Optional[float]

    conversation_id: str
    message_count: int
    history_turns_used: int

    used_web_search: bool
    search_query: Optional[str]

    reply: str
    reply_language: str

    # FIX: must exist or Pydantic crashes
    timing_ms: int

    tts_voice: str
    tts_language: str


class TextPipelineRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = None
    force_search: bool = False
    with_audio: bool = False   # FIX: was wrongly placed earlier


# ─────────────────────────────────────────────
# ACTION EXECUTOR (FIXED indentation)
# ─────────────────────────────────────────────
def _execute_desktop_actions(result, payload):
    import re

    match = re.search(r'\[ACTION:\s*(\w+)\((.*?)\)\]', result.reply)
    if not match:
        return

    try:
        from services import desktop_service

        action = match.group(1)
        args = match.group(2).replace('"', '').replace("'", "")

        logger.info(f"Action: {action}({args})")

        if action == "open_app":
            desktop_service.open_application(args)
        elif action == "volume":
            desktop_service.control_volume(args)
        elif action == "browser":
            desktop_service.browser_action(url=args)
        elif action == "system":
            desktop_service.system_power(args)
        elif action == "search_files":
            desktop_service.search_local_files(args)

    except Exception as e:
        logger.warning(f"Action failed: {e}")

    clean = re.sub(r'\[ACTION:.*?\]', '', result.reply).strip()
    result.reply = clean
    payload.reply = clean


# ─────────────────────────────────────────────
# VOICE ENDPOINT
# ─────────────────────────────────────────────
@router.post("/voice", response_model=PipelineResponse)
async def voice_pipeline(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
    force_search: bool = Form(False),
    with_audio: bool = Form(False),
    db: AsyncSession = Depends(get_db),
):

    audio_bytes = await file.read()
    content_type = file.content_type or "audio/webm"

    validate_audio_file(content_type, len(audio_bytes))

    try:
        result = await run_voice_pipeline(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.webm",
            content_type=content_type,
            db=db,
            conversation_id=conversation_id,
            force_search=force_search,
        )
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    payload = PipelineResponse(
        input_type="voice",
        transcript=result.transcript,
        transcript_language=result.transcript_language,
        transcript_language_code=result.transcript_language_code,
        audio_duration=result.audio_duration,

        conversation_id=result.conversation_id,
        message_count=result.message_count,
        history_turns_used=result.history_turns_used,

        used_web_search=result.used_web_search,
        search_query=result.search_query,

        reply=result.reply,
        reply_language=result.reply_language,

        timing_ms=0,   # FIX APPLIED

        tts_voice="hi-IN-MadhurNeural",
        tts_language=result.reply_language,
    )

    _execute_desktop_actions(result, payload)

    return payload


# ─────────────────────────────────────────────
# TEXT ENDPOINT
# ─────────────────────────────────────────────
@router.post("/text", response_model=PipelineResponse)
async def text_pipeline(
    req: TextPipelineRequest,
    db: AsyncSession = Depends(get_db),
):

    try:
        result = await run_text_pipeline(
            user_message=req.message,
            db=db,
            conversation_id=req.conversation_id,
            force_search=req.force_search,
        )
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    payload = PipelineResponse(
        input_type="text",
        transcript=result.transcript,
        transcript_language=result.transcript_language,
        transcript_language_code=result.transcript_language_code,
        audio_duration=None,

        conversation_id=result.conversation_id,
        message_count=result.message_count,
        history_turns_used=result.history_turns_used,

        used_web_search=result.used_web_search,
        search_query=result.search_query,

        reply=result.reply,
        reply_language=result.reply_language,

        timing_ms=0,   # FIX APPLIED

        tts_voice="hi-IN-MadhurNeural",
        tts_language=result.reply_language,
    )

    _execute_desktop_actions(result, payload)

    return payload


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────
@router.get("/health")
async def health():
    return {"status": "ok"}
