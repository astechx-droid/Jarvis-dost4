"""
services/groq_service.py — Groq API integration for JARVIS.

Handles:
  - Standard (non-streaming) chat completions
  - Streaming completions (for future WebSocket support)
  - Error handling and retries
"""

"""
services/groq_service.py
Optimized Groq AI integration for JARVIS
"""

import ssl
import logging
from typing import List, Dict, AsyncGenerator

import httpx

from groq import (
    AsyncGroq,
    APIError,
    RateLimitError,
    APIConnectionError,
)

from config import settings
from core.context import trim_history_for_context

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────────────────────

# SMART MODEL
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# FAST FALLBACK
FALLBACK_MODEL = "llama-3.1-8b-instant"

# ─────────────────────────────────────────────────────────────

_client = None


# ─────────────────────────────────────────────────────────────
# HTTP CLIENT
# ─────────────────────────────────────────────────────────────

def _make_http_client() -> httpx.AsyncClient:
    """
    Create stable HTTP client with proper SSL handling.
    """

    ssl_ctx = ssl.create_default_context()

    return httpx.AsyncClient(
        verify=ssl_ctx,
        timeout=60.0,
    )


# ─────────────────────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────────────────────

def get_groq_client() -> AsyncGroq:
    """
    Singleton Groq client.
    """

    global _client

    if _client is None:

        _client = AsyncGroq(
            api_key=settings.groq_api_key,
            http_client=_make_http_client(),
        )

    return _client


# ─────────────────────────────────────────────────────────────
# STANDARD CHAT COMPLETION
# ─────────────────────────────────────────────────────────────

async def chat_completion(
    messages: List[Dict],
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate a standard AI response.
    """

    client = get_groq_client()

    # Prevent oversized contexts
    messages = trim_history_for_context(messages)

    try:

        response = await client.chat.completions.create(
            model=model,

            messages=messages,

            temperature=0.7,

            max_tokens=1024,

            top_p=0.95,

            stream=False,

            timeout=20.0,
        )

        reply = (
            response.choices[0]
            .message
            .content
        )

        logger.info(
            "Groq response received | model=%s | tokens_used=%s",
            model,
            response.usage.total_tokens
            if response.usage
            else "unknown",
        )

        return (reply or "").strip()

    # ─────────────────────────────────────────────────────────
    # RATE LIMIT
    # ─────────────────────────────────────────────────────────

    except RateLimitError:

        logger.warning(
            "Groq rate limit hit on %s",
            model,
        )

        # fallback attempt
        if model != FALLBACK_MODEL:

            try:

                logger.info(
                    "Trying fallback model..."
                )

                return await chat_completion(
                    messages=messages,
                    model=FALLBACK_MODEL,
                )

            except Exception:
                pass

        raise ValueError(
            "Jarvis is busy right now. Please try again in a moment."
        )

    # ─────────────────────────────────────────────────────────
    # NETWORK ERROR
    # ─────────────────────────────────────────────────────────

    except APIConnectionError:

        logger.error(
            "Groq connection failed"
        )

        raise ValueError(
            "Could not connect to AI servers."
        )

    # ─────────────────────────────────────────────────────────
    # GENERIC API ERROR
    # ─────────────────────────────────────────────────────────

    except APIError as e:

        logger.error(
            "Groq API error: %s",
            e,
        )

        # fallback if smart model fails
        if model != FALLBACK_MODEL:

            try:

                logger.warning(
                    "Primary model failed. Using fallback."
                )

                return await chat_completion(
                    messages=messages,
                    model=FALLBACK_MODEL,
                )

            except Exception:
                pass

        raise ValueError(
            f"Groq API error: {e.message}"
        )

    except Exception as e:

        logger.exception(
            "Unexpected Groq error"
        )

        raise ValueError(
            f"Unexpected AI error: {e}"
        )


# ─────────────────────────────────────────────────────────────
# STREAMING CHAT COMPLETION
# ─────────────────────────────────────────────────────────────

async def chat_completion_stream(
    messages: List[Dict],
    model: str = DEFAULT_MODEL,
) -> AsyncGenerator[str, None]:
    """
    Stream AI response token-by-token.
    """

    client = get_groq_client()

    messages = trim_history_for_context(
        messages
    )

    try:

        stream = await client.chat.completions.create(
            model=model,

            messages=messages,

            temperature=0.75,

            max_tokens=1024,

            top_p=0.95,

            stream=True,
        )

        async for chunk in stream:

            try:

                delta = (
                    chunk.choices[0]
                    .delta
                    .content
                )

                if delta:
                    yield delta

            except Exception:
                continue

    except RateLimitError:

        yield (
            "[Jarvis is currently busy. "
            "Please try again shortly.]"
        )

    except APIConnectionError:

        yield (
            "[Connection problem detected.]"
        )

    except APIError as e:

        yield (
            f"[Groq API error: {e.message}]"
        )

    except Exception as e:

        logger.exception(
            "Streaming failed"
        )

        yield (
            f"[Streaming failed: {e}]"
        )
