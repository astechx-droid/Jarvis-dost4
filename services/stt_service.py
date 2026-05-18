"""
services/stt_service.py — Speech-to-Text service for JARVIS.

Uses Groq Whisper API (whisper-large-v3-turbo).
Fixed for Hinglish + Indian accent stability.
"""

import io
import logging
import httpx
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

_CLIENT_INSTANCE: Optional[httpx.AsyncClient] = None

def get_stt_client() -> httpx.AsyncClient:
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is None:
        _CLIENT_INSTANCE = httpx.AsyncClient(timeout=60.0)
    return _CLIENT_INSTANCE


GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> dict:

    if not audio_bytes:
        raise ValueError("Empty audio data")

    if len(audio_bytes) > 25 * 1024 * 1024:
        raise ValueError("Audio too large (max 25MB)")

    if prompt is None:
        prompt = (
            "User is speaking in Hinglish (Hindi + English mix) to an AI assistant named Jarvis. "
            "Handle Indian accent, code switching, and casual words like yaar, bhai, kya, hai, theek."
        )

    files = {
        "file": (f"audio.wav", io.BytesIO(audio_bytes), content_type or "audio/webm"),
    }

    # ✅ FIXED STABLE CONFIG FOR HINGLISH
    data = {
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",

        # 🔥 IMPORTANT FIX: DO NOT FORCE HINDI
        "language": "en",

        # 🧠 Better Hinglish understanding
        "prompt": prompt,

        # 🎯 Prevent gibberish output
        "temperature": 0.0,
    }

    client = get_stt_client()

    response = await client.post(
        GROQ_AUDIO_URL,
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        files=files,
        data=data,
    )

    if response.status_code != 200:
        raise ValueError(f"STT failed: {response.text}")

    result = response.json()

    text = result.get("text", "").strip()
    detected = result.get("language", "unknown")
    duration = result.get("duration")

    # 🔧 CLEAN OUTPUT (removes weird glitches)
    text = text.replace("Jarvis,", "Jarvis").strip()

    return {
        "text": text,
        "language": detected,
        "duration": duration,
        "segments": result.get("segments", []),
    }


async def transcribe_and_detect(audio_bytes: bytes, filename: str, content_type: str) -> dict:

    result = await transcribe_audio(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
        language=None,
    )

    text = result["text"]
    lang = result["language"]

    # Simple classification
    if any(word in text.lower() for word in ["kya", "hai", "yaar", "bhai", "nahi"]):
        final_lang = "hinglish"
    elif lang == "hi":
        final_lang = "hindi"
    else:
        final_lang = "english"

    return {
        "text": text,
        "language": final_lang,
        "language_code": lang,
        "duration": result["duration"],
    }
