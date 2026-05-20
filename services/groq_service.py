"""
services/groq_service.py
Ultra optimized Groq AI service for JARVIS
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
from core.model_router import choose_model

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────────────────────

FAST_MODEL = "llama-3.3-70b-versatile"

DEEP_MODEL = "deepseek-r1-distill-llama-70b"

FALLBACK_MODEL = "llama-3.1-8b-instant"

# ─────────────────────────────────────────────────────────────

_client = None


# ─────────────────────────────────────────────────────────────
# HTTP CLIENT
# ─────────────────────────────────────────────────────────────

def _make_http_client() -> httpx.AsyncClient:
    """
    Stable async HTTP client with SSL support.
    """

    ssl_ctx = ssl.create_default_context()

    return httpx.AsyncClient(
        verify=ssl_ctx,
        timeout=90.0,
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
# MODEL SELECTION
# ─────────────────────────────────────────────────────────────

def get_model(messages: List[Dict]) -> str:
    """
    Dynamically choose best model.
    """

    try:

        user_message = messages[-1]["content"]

        selected = choose_model(user_message)

        logger.info(
            "Model router selected: %s",
            selected,
        )

        return selected

    except Exception:

        return FAST_MODEL


# ─────────────────────────────────────────────────────────────
# STANDARD CHAT COMPLETION
# ─────────────────────────────────────────────────────────────

async def chat_completion(
    messages: List[Dict],
) -> str:
    """
    Generate AI response.
    """

    client = get_groq_client()

    messages = trim_history_for_context(messages)

    model = get_model(messages)

    try:

        response = await client.chat.completions.create(
            model=model,

            messages=messages,

            temperature=0.7,

            max_tokens=1024,

            top_p=0.95,

            stream=False,

            timeout=25.0,
        )

        reply = (
            response.choices[0]
            .message
            .content
        )

        logger.info(
            "Groq response | model=%s | tokens=%s",
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
            "Rate limit hit on %s",
            model,
        )

        try:

            logger.info(
                "Trying fallback model..."
            )

            response = await client.chat.completions.create(
                model=FALLBACK_MODEL,

                messages=messages,

                temperature=0.7,

                max_tokens=512,

                top_p=0.95,

                stream=False,
            )

            reply = (
                response.choices[0]
                .message
                .content
            )

            return (reply or "").strip()

        except Exception:

            raise ValueError(
                "Jarvis is currently overloaded. Please try again shortly."
            )

    # ─────────────────────────────────────────────────────────
    # NETWORK ERROR
    # ─────────────────────────────────────────────────────────

    except APIConnectionError:

        logger.error(
            "Groq connection error"
        )

        raise ValueError(
            "Unable to connect to AI servers."
        )

    # ─────────────────────────────────────────────────────────
    # API ERROR
    # ─────────────────────────────────────────────────────────

    except APIError as e:

        logger.error(
            "Groq API error: %s",
            e,
        )

        try:

            logger.warning(
                "Primary model failed. Using fallback."
            )

            response = await client.chat.completions.create(
                model=FALLBACK_MODEL,

                messages=messages,

                temperature=0.7,

                max_tokens=512,

                top_p=0.95,

                stream=False,
            )

            reply = (
                response.choices[0]
                .message
                .content
            )

            return (reply or "").strip()

        except Exception:

            raise ValueError(
                f"Groq API error: {e.message}"
            )

    except Exception as e:

        logger.exception(
            "Unexpected AI error"
        )

        raise ValueError(
            f"Unexpected AI error: {e}"
        )


# ─────────────────────────────────────────────────────────────
# STREAMING CHAT COMPLETION
# ─────────────────────────────────────────────────────────────

async def chat_completion_stream(
    messages: List[Dict],
) -> AsyncGenerator[str, None]:
    """
    Stream AI response token-by-token.
    """

    client = get_groq_client()

    messages = trim_history_for_context(messages)

    model = get_model(messages)

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
            "[Jarvis is busy right now. Please try again shortly.]"
        )

    except APIConnectionError:

        yield (
            "[Connection issue detected.]"
        )

    except APIError as e:

        yield (
            f"[Groq API error: {e.message}]"
        )

    except Exception as e:

        logger.exception(
            "Streaming failure"
        )

        yield (
            f"[Streaming failed: {e}]"
        )
