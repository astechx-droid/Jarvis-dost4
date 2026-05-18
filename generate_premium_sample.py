# -*- coding: utf-8 -*-
import asyncio
import edge_tts

async def premium_jarvis():
    # Sophisticated, professional Hinglish following the new protocol
    text = "Ji Sir. सिस्टम पूरी तरह तैयार है। Aapko kya chahiye? Main abhi madad kar sakta hoon. कमांड दीजिए सर।"
    voice = "hi-IN-MadhurNeural"
    filename = "jarvis_premium.mp3"
    
    print("Generating JARVIS (Premium Futuristic AI)...")
    # 1.15x for a precise, professional AI tempo
    communicate = edge_tts.Communicate(text, voice, rate="+15%", pitch="+0Hz")
    await communicate.save(filename)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    asyncio.run(premium_jarvis())
