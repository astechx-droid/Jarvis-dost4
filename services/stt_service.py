import io
import logging
from typing import Optional

import httpx
from config import settings

logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "mp4",
    "audio/x-m4a": "m4a",
    "audio/wav": "wav",
    "audio/webm": "webm",
    "video/webm": "webm",
}

_CLIENT_INSTANCE: Optional[httpx.AsyncClient] = None

GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def get_stt_client() -> httpx.AsyncClient:
    global _CLIENT_INSTANCE

    if _CLIENT_INSTANCE is None:
        _CLIENT_INSTANCE = httpx.AsyncClient(timeout=60.0)

    return _CLIENT_INSTANCE


def normalize_text(text: str) -> str:
    import re

    fixes = {
        "jarviss": "jarvis",
        "jarvez": "jarvis",
        "jarwis": "jarvis",
        "he jarvis": "hey jarvis",
        "hej jarvis": "hey jarvis",
    }

    for wrong, correct in fixes.items():
        text = re.sub(
            rf"\b{wrong}\b",
            correct,
            text,
            flags=re.IGNORECASE,
        )

    return text.strip()


def detect_language_from_text(text: str) -> tuple[str, str]:
    """
    Manual Hinglish detector.
    Prevents Whisper from returning nonsense like Icelandic.
    """

    text_lower = text.lower()

    hindi_words = [
        "kya", "kaise", "mera", "tum", "aap",
        "jarvis", "bhai", "hello", "hey",
        "namaste", "haan", "nahi", "kr", "kar"
    ]

    matches = sum(word in text_lower for word in hindi_words)

    if matches >= 2:
        return "hinglish", "hi"

    return "english", "en"


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
):
    if not audio_bytes:
        raise ValueError("Empty audio")

    if len(audio_bytes) > 25 * 1024 * 1024:
        raise ValueError("Audio too large")

    files = {
        "file": (
            filename or "audio.webm",
            io.BytesIO(audio_bytes),
            content_type or "audio/webm",
        )
    }

    data = {
        "model": "whisper-large-v3-turbo",

        # IMPORTANT
        "language": "en",

        "prompt": (
            "The speaker talks in Hinglish and English. "
            "Words like hey jarvis, bhai, kya, kaise "
            "must stay exactly as spoken."
        ),

        "response_format": "verbose_json",
        "temperature": 0.0,
    }

    client = get_stt_client()

    response = await client.post(
        GROQ_AUDIO_URL,
        headers={
            "Authorization": f"Bearer {settings.groq_api_key}"
        },
        files=files,
        data=data,
    )

    if response.status_code != 200:
        raise ValueError(response.text)

    result = response.json()

    text = result.get("text", "").strip()

    text = normalize_text(text)

    detected_language, language_code = detect_language_from_text(text)

    return {
        "text": text,
        "language": detected_language,
        "language_code": language_code,
        "duration": result.get("duration"),
        "segments": result.get("segments", []),
    }


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


def validate_audio_file(
    content_type: str,
    size_bytes: int,
):
    if not content_type:
        raise ValueError("Missing content type")

    if size_bytes > 25 * 1024 * 1024:
        raise ValueError("Audio too large")

    if size_bytes < 100:
        raise ValueError("Invalid audio")

    if content_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported audio format: {content_type}"
        )
