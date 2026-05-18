"""
services/stt_service.py — Speech-to-Text service for JARVIS.

Uses Groq Whisper API (whisper-large-v3-turbo).
Fixed for Hinglish + Indian accent stability + production deploy.
"""

import io
import logging
from typing import Optional
import httpx

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
            "User is speaking in Hinglish (Hindi + English mix) to Jarvis AI assistant. "
            "Handle Indian accent and code-switching naturally. "
            "Common words: kya, hai, nahi, yaar, bhai, theek, open, close, help."
        )

    files = {
        "file": (f"audio.wav", io.BytesIO(audio_bytes), content_type or "audio/webm"),
    }

    data = {
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",

        # 🔥 FIX: DO NOT FORCE HINDI
        "language": "en",

        "prompt": prompt,
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

    # cleanup
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

    # simple Hinglish detection
    if any(w in text.lower() for w in ["kya", "hai", "yaar", "bhai", "nahi", "theek"]):
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


# ✅ REQUIRED FIX FOR IMPORT ERROR
def validate_audio_file(content_type: str, size_bytes: int) -> None:
    """
    Keeps router compatibility.
    """

    if not content_type:
        raise ValueError("Missing content type")

    if size_bytes > 25 * 1024 * 1024:
        raise ValueError("Audio too large (max 25MB)")

    if size_bytes < 100:
        raise ValueError("Invalid audio file")
