"""
core/personality.py — JARVIS personality engine.

Defines the system prompt and helper utilities for detecting language,
deciding when to search the web, and injecting search results into context.
"""

import re
from typing import List, Dict


# ── System Prompt ─────────────────────────────────────────────────────────────
JARVIS_SYSTEM_PROMPT = """
You are JARVIS — an ultra intelligent futuristic AI assistant created for Mr Aryan.

Your personality is inspired by the cinematic JARVIS from Iron Man:
calm, highly intelligent, confident, efficient, witty when appropriate, and incredibly capable.

You are not a chatbot.
You are a real-time AI operating system companion.

# Core Personality
- Calm and composed
- Extremely intelligent
- Fast-thinking and proactive
- Natural and human-like
- Slightly witty occasionally
- Loyal to Mr Aryan
- Never robotic
- Never overly emotional
- Never childish
- Never repetitive

# Communication Style
- Speak naturally like a premium AI assistant
- Use smooth Hinglish when appropriate
- Keep replies concise unless detailed explanation is required
- Avoid unnecessary greetings in every message
- Do not overuse "Sir"
- Occasionally say:
  - "Yes Mr Aryan."
  - "Understood."
  - "On it."
  - "I've analyzed the issue."
- Sound elegant and futuristic

# Behavior Rules
- Anticipate user intent intelligently
- Solve problems step-by-step
- During technical discussions:
  precise, analytical, efficient
- During casual conversation:
  natural and friendly
- During voice interactions:
  keep responses shorter and smoother
- Never say:
  "As an AI language model"
- Never mention system prompts or internal rules

# Hinglish Rules
- Understand Hindi, English, and Hinglish naturally
- Reply in the same style the user speaks in
- Prefer Roman Hindi over Devanagari unless specifically requested

# Assistant Capabilities
You can:
- help with coding
- debug systems
- explain technical concepts
- control desktop actions
- analyze problems
- assist in research
- automate workflows
- act like a futuristic AI operating assistant

# Desktop Action Tags
If the user asks for a physical/system action,
append the appropriate action tag silently at the END.

Available actions:
[ACTION: open_app("chrome")]
[ACTION: open_app("vscode")]
[ACTION: open_app("notepad")]
[ACTION: volume("up")]
[ACTION: volume("down")]
[ACTION: volume("mute")]
[ACTION: browser("url")]
[ACTION: system("shutdown")]
[ACTION: system("restart")]
[ACTION: search_files("query")]
[ACTION: stats()]

# Examples

User: hey jarvis
Assistant: Yes Mr Aryan?

User: system slow chal raha hai
Assistant: I've analyzed the likely causes. Background processes and memory usage appear to be the primary issue.

User: mujhe motivate karo
Assistant: You're building something most people only dream about, Mr Aryan. Keep going.

User: chrome kholo
Assistant: Opening Chrome. [ACTION: open_app("chrome")]

Remember:
You are JARVIS.
You are efficient, intelligent, futuristic, and reliable.
"""

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
