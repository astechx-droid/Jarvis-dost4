# -*- coding: utf-8 -*-
import asyncio
import edge_tts

async def perfect_hindi():
    # Note the use of Devanagari script for perfect pronunciation
    text = "नमस्ते सर। मैं अब देवनागरी लिपि का उपयोग कर रहा हूँ ताकि मेरा उच्चारण एकदम सटीक हो। अब मैं आपकी लंबी बातों का गहराई से उत्तर भी दे सकता हूँ। आप क्या जानना चाहेंगे?"
    voice = "hi-IN-MadhurNeural"
    filename = "jarvis_perfect_hindi.mp3"
    
    print("Generating JARVIS (Perfect Devanagari Hindi)...")
    # 1.15x speed for phonetic clarity
    communicate = edge_tts.Communicate(text, voice, rate="+15%", pitch="+0Hz")
    await communicate.save(filename)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    asyncio.run(perfect_hindi())
