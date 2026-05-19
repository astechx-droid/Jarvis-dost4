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


def get_stt_client() -> httpx.AsyncClient:
    global _CLIENT_INSTANCE

    if _CLIENT_INSTANCE is None:
        _CLIENT_INSTANCE = httpx.AsyncClient(timeout=60.0)

    return _CLIENT_INSTANCE


GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def normalize_text(text: str) -> str:
    import re

    fixes = {
        "komand": "command",
        "komputer": "computer",
        "aplikeshan": "application",
    }

    for wrong, correct in fixes.items():
        text = re.sub(
            rf"\b{wrong}\b",
            correct,
            text,
            flags=re.IGNORECASE
        )

    return text


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
        raise ValueError("Audio too large")

    if prompt is None:
        prompt = (
            "User speaks Hinglish. "
            "Do not translate. "
            "Only transcribe."
        )

    files = {
        "file": (
            filename or "audio.wav",
            io.BytesIO(audio_bytes),
            content_type or "audio/webm",
        ),
    }

    data = {
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",
        "prompt": prompt,
        "temperature": 0.0,
    }

    if language:
        data["language"] = language

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

    return {
        "text": text,
        "language": result.get("language", "unknown"),
        "duration": result.get("duration"),
        "segments": result.get("segments", []),
    }


async def transcribe_and_detect(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
):
    """
    Compatibility wrapper for pipeline_service.py
    """

    result = await transcribe_audio(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
    )

    return result


def validate_audio_file(
    content_type: str,
    size_bytes: int,
) -> None:

    if not content_type:
        raise ValueError("Missing content type")

    if size_bytes > 25 * 1024 * 1024:
        raise ValueError("Too large")

    if size_bytes < 100:
        raise ValueError("Invalid audio")
