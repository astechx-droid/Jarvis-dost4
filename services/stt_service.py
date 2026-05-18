"""
services/stt_service.py — Speech-to-Text service for JARVIS
FIXED VERSION (Hinglish + pronunciation + Render safe)
"""

import io
import logging
from typing import Optional
import httpx

from config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# MIME TYPES (used by voice router)
# ─────────────────────────────────────────────
SUPPORTED_MIME_TYPES = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "mp4",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/webm": "webm",
    "video/webm": "webm",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
}

# ─────────────────────────────────────────────
# HTTP CLIENT (reuse connection)
# ─────────────────────────────────────────────
_CLIENT_INSTANCE: Optional[httpx.AsyncClient] = None


def get_stt_client() -> httpx.AsyncClient:
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is None:
        _CLIENT_INSTANCE = httpx.AsyncClient(timeout=60.0)
    return _CLIENT_INSTANCE


# ─────────────────────────────────────────────
# GROQ API
# ─────────────────────────────────────────────
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


# ─────────────────────────────────────────────
# SMART TEXT NORMALIZER (FIX pronunciation issues)
# ─────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """
    Fix common STT phonetic mistakes (Hinglish → correct English words)
    """
    import re

    corrections = {
        "komand": "command",
        "komputer": "computer",
        "aplikeshan": "application",
        "programing": "programming",
        "jarves": "jarvis",
        "kessidu": "what situation",
    }

    for wrong, correct in corrections.items():
        text = re.sub(rf"\b{wrong}\b", correct, text, flags=re.IGNORECASE)

    return text


# ─────────────────────────────────────────────
# MAIN TRANSCRIBE FUNCTION
# ─────────────────────────────────────────────
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
            "User speaks Hinglish (Hindi + English mix). "
            "Correct spelling of English words like command, computer, application. "
            "Do not hallucinate random languages."
        )

    files = {
        "file": (f"audio.wav", io.BytesIO(audio_bytes), content_type or "audio/webm"),
    }

    data = {
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",

        # IMPORTANT: do NOT force Hindi
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

    # ─────────────────────────────
    # FIX STEP: clean output text
    # ─────────────────────────────
    text = text.replace("Jarvis,", "Jarvis")
    text = normalize_text(text)

    return {
        "text": text,
        "language": detected,
        "duration": duration,
        "segments": result.get("segments", []),
    }


# ─────────────────────────────────────────────
# DETECTION WRAPPER
# ─────────────────────────────────────────────
async def transcribe_and_detect(audio_bytes: bytes, filename: str, content_type: str) -> dict:

    result = await transcribe_audio(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
        language=None,
    )

    text = result["text"]
    lang = result["language"]

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


# ─────────────────────────────────────────────
# REQUIRED FOR voice.py (DO NOT REMOVE)
# ─────────────────────────────────────────────
def validate_audio_file(content_type: str, size_bytes: int) -> None:
    if not content_type:
        raise ValueError("Missing content type")

    if size_bytes > 25 * 1024 * 1024:
        raise ValueError("Audio too large (max 25MB)")

    if size_bytes < 100:
        raise ValueError("Invalid audio file")
