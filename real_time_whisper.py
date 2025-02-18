import whisper
import pyaudio
import wave
import numpy as np
import threading
import time
import os

# Load the Whisper model
model = whisper.load_model("base")  # Use "tiny", "base", "small", "medium", "large"

# Audio configuration
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1  # Mono
RATE = 16000  # 16kHz sample rate
CHUNK = 1024  # Buffer size

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Flag to control recording
recording = True
audio_lock = threading.Lock()  # Lock to ensure safe file access

def capture_audio():
    """Captures audio and continuously saves small chunks as WAV files."""
    global recording
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []

    while recording:
        data = stream.read(CHUNK)
        frames.append(data)

        if len(frames) >= int(RATE / CHUNK * 5):  # Save every ~5 seconds
            with audio_lock:
                with wave.open("temp.wav", "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(audio.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
            frames = []  # Reset frames buffer

    stream.stop_stream()
    stream.close()

def transcribe_audio():
    """Continuously transcribes the latest recorded audio."""
    while recording:
        time.sleep(5)  # Wait for chunks to be recorded
        if os.path.exists("temp.wav"):
            with audio_lock:  # Ensure the file is not being written while reading
                result = model.transcribe("temp.wav")
            print("Transcription:", result["text"])

# Start threads
capture_thread = threading.Thread(target=capture_audio)
transcribe_thread = threading.Thread(target=transcribe_audio)

capture_thread.start()
transcribe_thread.start()

try:
    while True:
        time.sleep(1)  # Keep the main thread alive
except KeyboardInterrupt:
    print("\nStopping...")
    recording = False  # Stop both threads

capture_thread.join()
transcribe_thread.join()

# Cleanup
audio.terminate()
print("Stopped.")
