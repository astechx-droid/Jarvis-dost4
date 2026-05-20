import io
import logging
from typing import Optional

import httpx
from config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Supported Audio Types
# ─────────────────────────────────────────────────────────────

SUPPORTED_MIME_TYPES = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "mp4",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
    "video/webm": "webm",
    "audio/ogg": "ogg",
}

# ─────────────────────────────────────────────────────────────

_CLIENT_INSTANCE: Optional[httpx.AsyncClient] = None

GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# ─────────────────────────────────────────────────────────────
# HTTP Client
# ─────────────────────────────────────────────────────────────


def get_stt_client() -> httpx.AsyncClient:
    global _CLIENT_INSTANCE

    if _CLIENT_INSTANCE is None:
        _CLIENT_INSTANCE = httpx.AsyncClient(
            timeout=60.0
        )

    return _CLIENT_INSTANCE


# ─────────────────────────────────────────────────────────────
# Text Cleanup
# ─────────────────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    import re

    if not text:
        return ""

    fixes = {
        "jarviss": "jarvis",
        "jarvez": "jarvis",
        "jarwis": "jarvis",
        "he jarvis": "hey jarvis",
        "hej jarvis": "hey jarvis",
        "jervis": "jarvis",
        "jarbis": "jarvis",
    }

    cleaned = text.strip()

    for wrong, correct in fixes.items():
        cleaned = re.sub(
            rf"\b{wrong}\b",
            correct,
            cleaned,
            flags=re.IGNORECASE,
        )

    return cleaned.strip()


# ─────────────────────────────────────────────────────────────
# Hinglish Detector
# ─────────────────────────────────────────────────────────────


def detect_language_from_text(text: str) -> tuple[str, str]:
    """
    Prevents Whisper from returning random languages.
    """

    if not text:
        return "english", "en"

    text_lower = text.lower()

    hinglish_words = [
        "kya",
        "kaise",
        "mera",
        "tum",
        "aap",
        "bhai",
        "jarvis",
        "hello",
        "hey",
        "namaste",
        "haan",
        "nahi",
        "kr",
        "kar",
        "bolo",
        "sun",
        "bata",
    ]

    matches = sum(
        word in text_lower
        for word in hinglish_words
    )

    if matches >= 2:
        return "hinglish", "hi"

    return "english", "en"


# ─────────────────────────────────────────────────────────────
# Main STT Function
# ─────────────────────────────────────────────────────────────


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
):
    """
    Transcribe audio using Groq Whisper.
    """

    if not audio_bytes:
        raise ValueError("Empty audio received")

    if len(audio_bytes) > 25 * 1024 * 1024:
        raise ValueError("Audio file too large")

    normalized_content_type = (
        content_type.split(";")[0].strip().lower()
        if content_type
        else "audio/webm"
    )

    files = {
        "file": (
            filename or "audio.webm",
            io.BytesIO(audio_bytes),
            normalized_content_type,
        )
    }

    data = {
        "model": "whisper-large-v3-turbo",

        # Force English base
        # prevents Icelandic nonsense
        "language": "en",

        "prompt": (
            "The user speaks Hinglish and English. "
            "Common words include: hey jarvis, bhai, kya, kaise. "
            "Do NOT translate words."
        ),

        "response_format": "verbose_json",
        "temperature": 0.0,
    }

    client = get_stt_client()

    try:
        response = await client.post(
            GROQ_AUDIO_URL,
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}"
            },
            files=files,
            data=data,
        )

    except Exception as e:
        logger.exception("Groq STT request failed")
        raise ValueError(f"STT request failed: {e}")

    if response.status_code != 200:
        logger.error(
            "Groq STT Error | status=%s | body=%s",
            response.status_code,
            response.text,
        )

        raise ValueError(
            f"Groq STT failed: {response.text}"
        )

    result = response.json()

    text = result.get("text", "").strip()

    text = normalize_text(text)

    detected_language, language_code = detect_language_from_text(text)

    logger.info(
        "STT Success | lang=%s | text=%s",
        detected_language,
        text[:80],
    )

    return {
        "text": text,
        "language": detected_language,
        "language_code": language_code,
        "duration": result.get("duration"),
        "segments": result.get("segments", []),
    }


# ─────────────────────────────────────────────────────────────
# Compatibility Wrapper
# ─────────────────────────────────────────────────────────────


async def transcribe_and_detect(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
):
    """
    Compatibility wrapper for old imports.
    """

    return await transcribe_audio(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
    )


# ─────────────────────────────────────────────────────────────
# Audio Validator
# ─────────────────────────────────────────────────────────────


def validate_audio_file(
    content_type: str,
    size_bytes: int,
):
    """
    Validate uploaded audio file.
    """

    if not content_type:
        raise ValueError("Missing content type")

    if size_bytes > 25 * 1024 * 1024:
        raise ValueError("Audio too large")

    if size_bytes < 100:
        raise ValueError("Invalid or empty audio")

    # FIX:
    # audio/webm;codecs=opus
    # -> audio/webm

    normalized_type = (
        content_type.split(";")[0]
        .strip()
        .lower()
    )

    if normalized_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported audio format: {content_type}"
        )

    return True
