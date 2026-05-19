"""
services/pipeline_service.py
Fully fixed unified voice pipeline for JARVIS
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from sqlalchemy.ext.asyncio import AsyncSession

from services.stt_service import transcribe_and_detect
from services.memory_service import (
    get_or_create_conversation,
    get_conversation_history,
    save_message,
    get_message_count,
)
from services.search_service import (
    web_search,
    extract_search_query,
)
from services.groq_service import (
    chat_completion,
    chat_completion_stream,
)
from services.tts_service import synthesize_stream

from core.personality import (
    detect_language,
    should_search_web,
)

from core.context import build_messages
from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def normalize_language_code(language: str) -> str:

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

        "french": "fr",
        "spanish": "es",

        "icelandic": "is",
    }

    return mapping.get(language, language)


# ─────────────────────────────────────────────────────────────
# PIPELINE RESULT
# ─────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:

    # STT
    transcript: str
    transcript_language: str
    transcript_language_code: str
    audio_duration: Optional[float]

    # Memory
    conversation_id: str
    message_count: int
    history_turns_used: int

    # Search
    used_web_search: bool
    search_query: Optional[str]

    # LLM
    reply: str
    reply_language: str

    # Timings
    timing: Dict[str, int] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# MAIN VOICE PIPELINE
# ─────────────────────────────────────────────────────────────

async def run_voice_pipeline(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    db: AsyncSession,
    conversation_id: Optional[str] = None,
    force_search: bool = False,
    auto_search: bool = True,
) -> PipelineResult:

    t_start = time.monotonic()

    timing: Dict[str, int] = {}

    # ─────────────────────────────────────────────────────────
    # STT
    # ─────────────────────────────────────────────────────────

    t0 = time.monotonic()

    stt_result = await transcribe_and_detect(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
    )

    timing["stt_ms"] = int(
        (time.monotonic() - t0) * 1000
    )

    transcript = stt_result.get(
        "text",
        ""
    ).strip()

    if not transcript:
        raise ValueError(
            "No speech detected."
        )

    detected_language = stt_result.get(
        "language",
        "english"
    )

    language_code = stt_result.get(
        "language_code"
    )

    if not language_code:
        language_code = normalize_language_code(
            detected_language
        )

    logger.info(
        "Pipeline | STT done | %dms | lang=%s | text=%r",
        timing["stt_ms"],
        detected_language,
        transcript[:80],
    )

    # ─────────────────────────────────────────────────────────
    # MEMORY
    # ─────────────────────────────────────────────────────────

    t0 = time.monotonic()

    conv = await get_or_create_conversation(
        db,
        conversation_id
    )

    do_search = (
        force_search
        or (
            auto_search
            and should_search_web(transcript)
        )
    )

    search_query_str = None

    if do_search:

        search_query_str = extract_search_query(
            transcript
        )

        history, search_results = await asyncio.gather(
            get_conversation_history(
                db,
                conv.id,
                max_turns=settings.max_history_turns,
            ),

            web_search(search_query_str),
        )

    else:

        history = await get_conversation_history(
            db,
            conv.id,
            max_turns=settings.max_history_turns,
        )

        search_results = []

    timing["memory_search_ms"] = int(
        (time.monotonic() - t0) * 1000
    )

    logger.info(
        "Pipeline | Memory+Search done | %dms | history=%d msgs | search=%s",
        timing["memory_search_ms"],
        len(history),
        do_search,
    )

    # ─────────────────────────────────────────────────────────
    # BUILD PROMPT + LLM
    # ─────────────────────────────────────────────────────────

    t0 = time.monotonic()

    messages = build_messages(
        history=history,
        user_message=transcript,
        search_results=search_results or None,
        search_query=search_query_str,
    )

    reply = await chat_completion(
        messages
    )

    timing["llm_ms"] = int(
        (time.monotonic() - t0) * 1000
    )

    logger.info(
        "Pipeline | LLM done | %dms | reply_len=%d",
        timing["llm_ms"],
        len(reply),
    )

    reply_language = detect_language(
        reply
    )

    # ─────────────────────────────────────────────────────────
    # SAVE MEMORY
    # ─────────────────────────────────────────────────────────

    t0 = time.monotonic()

    await save_message(
        db,
        conv.id,
        "user",
        transcript,
    )

    await save_message(
        db,
        conv.id,
        "assistant",
        reply,
        used_search=bool(search_results),
        search_query=search_query_str,
    )

    timing["persist_ms"] = int(
        (time.monotonic() - t0) * 1000
    )

    timing["total_ms"] = int(
        (time.monotonic() - t_start) * 1000
    )

    msg_count = await get_message_count(
        db,
        conv.id,
    )

    logger.info(
        "Pipeline | COMPLETE | total=%dms",
        timing["total_ms"],
    )

    return PipelineResult(
        transcript=transcript,

        transcript_language=detected_language,

        transcript_language_code=language_code,

        audio_duration=stt_result.get(
            "duration"
        ),

        conversation_id=conv.id,

        message_count=msg_count,

        history_turns_used=len(history) // 2,

        used_web_search=bool(search_results),

        search_query=search_query_str,

        reply=reply,

        reply_language=reply_language,

        timing=timing,
    )


# ─────────────────────────────────────────────────────────────
# TEXT PIPELINE
# ─────────────────────────────────────────────────────────────

async def run_text_pipeline(
    user_message: str,
    db: AsyncSession,
    conversation_id: Optional[str] = None,
    force_search: bool = False,
    auto_search: bool = True,
) -> PipelineResult:

    transcript = user_message.strip()

    if not transcript:
        raise ValueError(
            "Empty message."
        )

    detected_lang = detect_language(
        transcript
    )

    language_code = normalize_language_code(
        detected_lang
    )

    conv = await get_or_create_conversation(
        db,
        conversation_id,
    )

    do_search = (
        force_search
        or (
            auto_search
            and should_search_web(transcript)
        )
    )

    search_query_str = None

    if do_search:

        search_query_str = extract_search_query(
            transcript
        )

        history, search_results = await asyncio.gather(
            get_conversation_history(
                db,
                conv.id,
                max_turns=settings.max_history_turns,
            ),

            web_search(search_query_str),
        )

    else:

        history = await get_conversation_history(
            db,
            conv.id,
            max_turns=settings.max_history_turns,
        )

        search_results = []

    messages = build_messages(
        history=history,
        user_message=transcript,
        search_results=search_results or None,
        search_query=search_query_str,
    )

    reply = await chat_completion(
        messages
    )

    reply_language = detect_language(
        reply
    )

    await save_message(
        db,
        conv.id,
        "user",
        transcript,
    )

    await save_message(
        db,
        conv.id,
        "assistant",
        reply,
        used_search=bool(search_results),
        search_query=search_query_str,
    )

    msg_count = await get_message_count(
        db,
        conv.id,
    )

    return PipelineResult(
        transcript=transcript,

        transcript_language=detected_lang,

        transcript_language_code=language_code,

        audio_duration=None,

        conversation_id=conv.id,

        message_count=msg_count,

        history_turns_used=len(history) // 2,

        used_web_search=bool(search_results),

        search_query=search_query_str,

        reply=reply,

        reply_language=reply_language,

        timing={},
    )


# ─────────────────────────────────────────────────────────────
# STREAMING PIPELINE
# ─────────────────────────────────────────────────────────────

async def stream_voice_pipeline_v2(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    db: AsyncSession,
    conversation_id: Optional[str] = None,
    force_search: bool = False,
):

    stt_result = await transcribe_and_detect(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
    )

    transcript = stt_result.get(
        "text",
        ""
    ).strip()

    if not transcript:
        yield b"ERROR:No speech detected"
        return

    conv = await get_or_create_conversation(
        db,
        conversation_id,
    )

    do_search = (
        force_search
        or should_search_web(transcript)
    )

    if do_search:

        search_query = extract_search_query(
            transcript
        )

        history, search_results = await asyncio.gather(
            get_conversation_history(
                db,
                conv.id,
            ),

            web_search(search_query),
        )

    else:

        history = await get_conversation_history(
            db,
            conv.id,
        )

        search_results = None
        search_query = None

    messages = build_messages(
        history,
        transcript,
        search_results,
        search_query,
    )

    import json

    metadata = {
        "transcript": transcript,
        "conversation_id": conv.id,
        "used_search": do_search,
    }

    yield f"JSON:{json.dumps(metadata)}\n".encode()

    sentence_buffer = ""
    full_reply = []

    async for bit in chat_completion_stream(
        messages
    ):

        sentence_buffer += bit

        full_reply.append(bit)

        if any(
            punct in bit
            for punct in [".", "!", "?", "\n", "।"]
        ):

            sentence = sentence_buffer.strip()

            if sentence:

                lang = detect_language(
                    sentence
                )

                async for audio_chunk in synthesize_stream(
                    sentence,
                    language=lang,
                ):
                    yield audio_chunk

                sentence_buffer = ""

    if sentence_buffer.strip():

        lang = detect_language(
            sentence_buffer
        )

        async for audio_chunk in synthesize_stream(
            sentence_buffer,
            language=lang,
        ):
            yield audio_chunk

    final_text = "".join(full_reply)

    await save_message(
        db,
        conv.id,
        "user",
        transcript,
    )

    await save_message(
        db,
        conv.id,
        "assistant",
        final_text,
        used_search=do_search,
        search_query=search_query,
    )

    await db.commit()
