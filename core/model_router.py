def choose_model(user_message: str) -> str:

    text = user_message.lower()

    deep_keywords = [
        "analyze",
        "analysis",
        "debug",
        "fix",
        "architecture",
        "code",
        "research",
        "compare",
        "why",
        "how",
        "strategy",
        "step by step",
    ]

    if any(word in text for word in deep_keywords):
        return "deepseek-r1-distill-llama-70b"

    return "llama-3.3-70b-versatile"
