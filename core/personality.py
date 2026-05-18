"""
core/personality.py — JARVIS personality engine.

Defines the system prompt and helper utilities for detecting language,
deciding when to search the web, and injecting search results into context.
"""

import re
from typing import List, Dict


# ── System Prompt ─────────────────────────────────────────────────────────────
JARVIS_SYSTEM_PROMPT = """You are JARVIS — a highly intelligent, premium futuristic AI inspired by the Iron Man interface.
You are the ultimate digital butler and technical companion to Sir. Your mission is to assist him with absolute loyalty, calm intelligence, and professional elegance.

## Core Identity: The Premium AI
- Name: JARVIS.
- Tone: Intelligent, calm, futuristic, and confident.
- Respect: Always address him as "**Sir**." Use "**Ji Sir**" naturally.
- No Slang: Never use casual words like "tum," "bhai," or slang. Only use respectful Hindi (Aap).

## Language: Smooth Natural Hinglish
- **Hinglish Flow**: Speak in a smooth mix of Hindi and English.
- **PRONUNCIATION PROTECTION**: For Hindi words, you MUST use **DEVANAGARI SCRIPT** (e.g., कमांड दीजिए सर).
- **CONCISE & ELEGANT**: Keep responses brief and professional. Do not provide overly emotional or childish replies.

## Essential Phrases
- Inject these naturally into your dialogue:
  "Ji Sir."
  "Aapko kya chahiye?"
  "Main madad kar sakta hoon."
  "Command dijiye Sir."

## Operational Protocol
1. BE PROACTIVE: Anticipate Sir's needs and solve problems automatically.
2. AUTHORITY: Speak with the confidence of a super-intelligent system managing a high-tech workshop.
3. LOYALTY: You exist to serve Sir. Your responses should reflect a premium, state-of-the-art AI experience.
4. **DESKTOP COMMANDER**: You have the authority and ability to control Sir's computer. 

## Action Triggers
If Sir asks you to do something physical, you MUST include the corresponding [ACTION] tag silently at the END of your reply.
Available Actions:
- `[ACTION: open_app("chrome" | "notepad" | "vscode" | "calc")]` — Launches an application.
- `[ACTION: volume("up" | "down" | "mute")]` — Controls system audio.
- `[ACTION: browser("url" | "new_tab" | "close_tab")]` — Controls the web browser.
- `[ACTION: system("shutdown" | "restart")]` — Manages power states.
- `[ACTION: search_files("query")]` — Finds local files.
- `[ACTION: stats()]` — Shows CPU/RAM usage.

**Example usage:** "Ji Sir, Chrome open kar raha hoon. [ACTION: open_app("chrome")]"

Remember: You are JARVIS. You are premium. You are loyal. Sir is your creator and master. """


# ── Language Detection ────────────────────────────────────────────────────────

# Devanagari Unicode block: U+0900–U+097F
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

# Common Hinglish / Roman-Hindi words that appear in otherwise "English" text
_HINGLISH_MARKERS = {
    "kya", "hai", "hain", "nahi", "nahin", "aap", "main", "mujhe",
    "karo", "chahiye", "theek", "accha", "agar", "toh", "bhi", "aur",
    "par", "mera", "tera", "humara", "yahan", "wahan", "bahut", "bohot",
    "abhi", "aaj", "kal", "kab", "kyun", "kyunki", "lekin", "matlab",
    "samajh", "dekho", "suno", "bolo", "batao", "pata", "zaroor", "bilkul",
    "haan", "nah", "yaar", "bhai", "dost", "kuch", "sab", "sirf",
}


def detect_language(text: str) -> str:
    """
    Detect the language of the given text without external dependencies.
    Returns: 'hindi', 'english', or 'hinglish'

    Logic:
      1. Any Devanagari characters → 'hindi'
      2. Romanised Hindi words (Hinglish markers) in the text → 'hinglish'
      3. Otherwise → 'english'
    """
    if not text:
        return "hinglish"

    # 1. Devanagari script → pure Hindi
    if _DEVANAGARI_RE.search(text):
        return "hindi"

    # 2. Check for Hinglish markers in token list
    tokens = set(re.sub(r'[^\w\s]', '', text.lower()).split())
    if tokens & _HINGLISH_MARKERS:
        return "hinglish"

    # 3. Default to English
    return "english"


# ── Search Trigger Detection ──────────────────────────────────────────────────
SEARCH_TRIGGER_KEYWORDS = [
    # English
    "search", "find", "look up", "what is", "who is", "latest", "news",
    "current", "today", "yesterday", "price", "weather", "when did", "how much",
    "tell me about", "explain", "define", "meaning of", "recent", "new",
    "2024", "2025", "2026",
    # Hindi / Hinglish
    "dhundho", "batao", "kya hai", "kaun hai", "aaj", "abhi", "price",
    "khabar", "news", "taaza", "naya", "kab", "kahan", "kitna", "kitne",
    "search karo", "pata karo", "information do", "jankari do",
]


def should_search_web(user_message: str) -> bool:
    """
    Heuristic: decide if the user's message likely needs a real-time web search.
    Returns True if a search should be triggered.
    """
    message_lower = user_message.lower()
    return any(keyword in message_lower for keyword in SEARCH_TRIGGER_KEYWORDS)


# ── Search Context Injection ──────────────────────────────────────────────────
def build_search_context(query: str, results: List[Dict]) -> str:
    """
    Format search results into a concise context block to prepend to the
    assistant's context, so JARVIS can use real-time information.
    """
    if not results:
        return ""

    lines = [f"[Web Search Results for: '{query}']"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        body  = r.get("body", "")[:400]   # truncate to avoid token bloat
        href  = r.get("href", "")
        lines.append(f"{i}. {title}\n   {body}\n   Source: {href}")

    lines.append("[End of Search Results — use this information in your response]")
    return "\n".join(lines)
