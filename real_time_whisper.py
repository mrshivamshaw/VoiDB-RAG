import whisper
import pyaudio
import wave
import threading
import time
import os
import requests
import pvporcupine
import struct
from queue import Queue
import numpy as np
from dotenv import load_dotenv
load_dotenv()


class VoiceAssistant:
    def __init__(self, whisper_model="tiny", sample_rate=16000, chunk_size=1024, 
                 wake_word="computer", llm_model="deepseek-r1:1.5b", 
                 llm_endpoint="http://localhost:11434/api/generate"):
        # Configuration
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = sample_rate
        self.CHUNK = chunk_size
        self.whisper_model = whisper_model
        self.llm_model = llm_model
        self.llm_endpoint = llm_endpoint
        self.wake_word = wake_word
        
        # State variables
        self.recording = False
        self.running = True
        self.audio_queue = Queue()
        
        # Initialize components
        self.audio = pyaudio.PyAudio()
        print(f"Loading Whisper model '{whisper_model}'...")
        self.model = whisper.load_model(whisper_model)
        print("Whisper model loaded.")
        
        # Initialize Porcupine wake word detector
        self.porcupine = pvporcupine.create(
            access_key=os.getenv("PVPORCUPINE_ACCESS_KEY"), 
            keywords=[wake_word]
        )
        
    def start(self):
        """Start all threads and the voice assistant."""
        # Create and start threads
        threads = [
            threading.Thread(target=self.detect_wake_word, daemon=True),
            threading.Thread(target=self.process_audio_queue, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
            
        print(f"Voice assistant started. Listening for wake word '{self.wake_word}'...")
        
        try:
            while self.running:
                time.sleep(0.1)  # Reduced sleep time for more responsive shutdown
        except KeyboardInterrupt:
            self.stop()
        
        for thread in threads:
            thread.join(timeout=1.0)  # Add timeout to prevent hanging
            
        print("Voice assistant stopped.")
            
    def stop(self):
        """Stop all threads and cleanup resources."""
        print("\nStopping voice assistant...")
        self.running = False
        self.recording = False
        
        # Clean up resources
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()
        if hasattr(self, 'audio'):
            self.audio.terminate()
    
    def detect_wake_word(self):
        """Listen for wake word continuously."""
        wake_word_stream = self.audio.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=self.FORMAT,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )
        
        while self.running:
            try:
                pcm = wake_word_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm_unpacked = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                keyword_index = self.porcupine.process(pcm_unpacked)
                if keyword_index >= 0:
                    print(f"\nWake word '{self.wake_word}' detected! Listening...")
                    self.record_audio()
            except Exception as e:
                print(f"Error in wake word detection: {e}")
                time.sleep(0.5)  # Brief pause on error
                
        wake_word_stream.close()
                
    def record_audio(self):
        """Record audio for processing after wake word detection."""
        frames = []
        audio_stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        # Record for maximum 5 seconds (configurable)
        max_recording_time = 5  # seconds
        silence_threshold = 300  # Adjust based on your microphone sensitivity
        silence_duration = 0.5   # Stop after 0.5s of silence
        
        print("Recording...")
        recording_start = time.time()
        silence_start = None
        
        try:
            while self.running and (time.time() - recording_start) < max_recording_time:
                data = audio_stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Simple silence detection
                audio_data = np.frombuffer(data, dtype=np.int16)
                if np.abs(audio_data).mean() < silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration:
                        print("Silence detected, stopping recording")
                        break
                else:
                    silence_start = None
                    
            # Only process if we have enough audio
            if len(frames) > 5:  # At least 5 chunks to be meaningful
                audio_file = "temp.wav"
                with wave.open(audio_file, "wb") as wf:
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                    wf.setframerate(self.RATE)
                    wf.writeframes(b''.join(frames))
                
                self.audio_queue.put(audio_file)
                
        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            audio_stream.close()
    
    def process_audio_queue(self):
        """Process audio files in the queue."""
        while self.running:
            try:
                if not self.audio_queue.empty():
                    audio_file = self.audio_queue.get()
                    print("Transciption started...")
                    self.transcribe_and_respond(audio_file)
                else:
                    time.sleep(0.1)  # Short sleep when queue is empty
            except Exception as e:
                print(f"Error processing audio: {e}")
                time.sleep(0.5)
    
    def transcribe_and_respond(self, audio_file):
        """Transcribe audio and get response from LLM."""
        try:
            print("Transcribing audio...")
            result = self.model.transcribe(audio_file)
            transcription = result["text"].strip()
            
            if transcription:
                print(f"Transcription: {transcription}")
                self.get_llm_response(transcription)
            else:
                print("No speech detected.")
                
            # Clean up the temporary file
            try:
                os.remove(audio_file)
            except:
                pass
                
        except Exception as e:
            print(f"Error in transcription: {e}")
    
    def get_llm_response(self, prompt):
        """Get response from LLM API."""
        try:
            print(f"Sending to LLM: '{prompt}'")
            data = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.llm_endpoint, json=data)
            
            if response.status_code == 200:
                response_data = response.json()
                llm_response = response_data.get("response", "No response received")
                print(f"LLM Response: {llm_response}")
                # Here you could add text-to-speech output
            else:
                print(f"LLM API error: {response.status_code}, {response.text}")
                
        # except requests.exceptions.Timeout:
        #     print("LLM request timed out")
        except Exception as e:
            print(f"Error getting LLM response: {e}")


if __name__ == "__main__":
    assistant = VoiceAssistant(
        whisper_model="tiny",  # Use tiny model for faster processing
        wake_word="computer"
    )
    assistant.start()