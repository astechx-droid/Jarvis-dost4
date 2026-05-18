import os
import time
import wave
import pyaudio
import numpy as np
import requests
import threading
from openwakeword.model import Model
import tempfile
import io
from pydub import AudioSegment

# ── Configuration ─────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000/conversation/voice/stream"
WAKE_WORD_MODEL = "hey_jarvis"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

# ── VAD Config ────────────────────────────────────────────────────────────────
SILENCE_THRESHOLD = 700 
SILENCE_DURATION = 1.0  


class JarvisHardwareClient:
    def __init__(self):
        print("Initializing JARVIS Parallel Core...")
        self.oww_model = Model(wakeword_models=[WAKE_WORD_MODEL], inference_framework="onnx")
        self.pa = pyaudio.PyAudio()
        
        # Continuous Input Stream
        self.mic_stream = self.pa.open(
            format=FORMAT, channels=CHANNELS, rate=RATE,
            input=True, frames_per_buffer=CHUNK
        )
        
        self.stop_playback = threading.Event()
        self.is_speaking = False
        print("JARVIS Listening Module Online. You can interrupt me anytime, Sir.")

    def listen_continuously(self):
        while True:
            try:
                data = self.mic_stream.read(CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)
                
                # Check for Wake Word
                prediction = self.oww_model.predict(audio_np)
                
                for mdl, score in prediction.items():
                    if score > 0.45:  # Slightly more sensitive for easier interrupts
                        if self.is_speaking:
                            print("\n[INTERRUPT] Stopping playback immediately, Sir.")
                            self.stop_playback.set()
                        else:
                            print(f"\nWake word detected! (Score: {score:.2f})")
                        
                        # Trigger recording
                        self.record_and_send()
                        self.oww_model.reset()
                        break
            except Exception as e:
                print(f"Error in listener: {e}")

    def record_and_send(self):
        print("Listening for command...")
        frames = []
        silence_start = None
        
        while True:
            data = self.mic_stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            volume = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
            if volume < SILENCE_THRESHOLD:
                if silence_start is None: silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION: break
            else:
                silence_start = None
                
        # Non-blocking send to backend
        temp_path = os.path.join(tempfile.gettempdir(), f"cmd_{int(time.time())}.wav")
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(CHANNELS); wf.setsampwidth(self.pa.get_sample_size(FORMAT))
            wf.setframerate(RATE); wf.writeframes(b''.join(frames))
            
        thread = threading.Thread(target=self.send_to_backend, args=(temp_path,))
        thread.start()

    def send_to_backend(self, file_path):
        try:
            with open(file_path, "rb") as f:
                response = requests.post(BACKEND_URL, files={"file": f}, data={"with_audio": "true"}, stream=True)
                if response.status_code == 200:
                    self.play_response_stream(response)
        except Exception as e:
            print(f"Bridge connection error: {e}")

    def play_response_stream(self, response):
        """Streams audio sentence-by-sentence with instant kill support."""
        self.stop_playback.clear()
        self.is_speaking = True
        
        separator = b"\r\n\r\nAUDIO:"
        buffer = b""
        json_parsed = False
        
        # Audio Player Stream
        out_stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
        
        try:
            for chunk in response.iter_content(chunk_size=4096):
                if self.stop_playback.is_set():
                    break
                
                buffer += chunk
                if not json_parsed and separator in buffer:
                    _, audio_data = buffer.split(separator, 1)
                    buffer = audio_data
                    json_parsed = True
            
            if json_parsed and not self.stop_playback.is_set():
                # Convert MP3 bytes from buffer to Raw PCM for the stream
                audio_segment = AudioSegment.from_file(io.BytesIO(buffer), format="mp3")
                raw_data = audio_segment.raw_data
                
                # Play in small chunks for high granularity interruption
                chunk_len = 2048
                for i in range(0, len(raw_data), chunk_len):
                    if self.stop_playback.is_set():
                        break
                    out_stream.write(raw_data[i:i+chunk_len])
        
        finally:
            out_stream.stop_stream()
            out_stream.close()
            self.is_speaking = False
            self.stop_playback.clear()

if __name__ == "__main__":
    client = JarvisHardwareClient()
    client.listen_continuously()
