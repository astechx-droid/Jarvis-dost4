# -*- coding: utf-8 -*-
import asyncio
import edge_tts

async def friendly_companion():
    # Simple, general Hindi/Hinglish for a friend-like feel
    text = "नमस्ते सर! कैसे हैं आप? मैंने आपके सारे सिस्टम्स चेक कर लिए हैं, सब ठीक है। वैसे, काफी देर से आप काम कर रहे हैं, क्या मैं आपके लिए कुछ नया सर्च करूँ या आप थोड़ा ब्रेक लेना चाहेंगे?"
    voice = "hi-IN-MadhurNeural"
    filename = "jarvis_friendly.mp3"
    
    print("Generating JARVIS (Friendly Companion)...")
    # Using 1.15x for a natural, conversational pace
    communicate = edge_tts.Communicate(text, voice, rate="+15%", pitch="+0Hz")
    await communicate.save(filename)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    asyncio.run(friendly_companion())
