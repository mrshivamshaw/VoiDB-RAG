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
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


class VoiceAssistant:
    def __init__(self, whisper_model="tiny", sample_rate=16000, chunk_size=1024, 
                 wake_word="computer", model="deepseek-r1:1.5b", 
                 llm_endpoint="http://localhost:11434/api/generate"):
        # Configuration
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = sample_rate
        self.CHUNK = chunk_size
        self.whisper_model = whisper_model
        self.llm_model = Ollama(model=model)
        self.llm_endpoint = llm_endpoint
        self.wake_word = wake_word
        self.output_parser=StrOutputParser()

        
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
    
    def get_llm_response(self, question):
        """Get response from LLM API for database-related queries only."""
        try:
            # More explicit and restrictive system prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a specialized database assistant that ONLY answers questions about:
    - SQL queries and syntax
    - Database design and architecture
    - Database management systems (MySQL, PostgreSQL, MongoDB, etc.)
    - Data modeling and normalization
    - Database performance and optimization
    - Database administration tasks

    For ANY question not directly related to databases, respond ONLY with:
    "I can only answer database-related questions. Please ask a question about databases or SQL."

    DO NOT answer questions about other topics, even if they seem related to programming or data.
    Be concise and direct in your answers to database questions."""),
                ("user", "Question: {question}")
            ])
            
            print(f"Sending to LLM: '{question}'")
            
            # Check if it's likely a database question before sending to the model
            is_db_related = self._is_database_question(question)
            
            if not is_db_related:
                print("I can only answer database-related questions. Please ask a question about databases or SQL.")
                return
            
            # If it seems database related, proceed with the LLM
            chain = prompt | self.llm_model | self.output_parser
            llm_response = chain.invoke({"question": question})
            print(f"LLM Response: {llm_response}")
            return llm_response
                
        except Exception as e:
            error_msg = f"Error getting LLM response: {e}"
            print(error_msg)
            return "Sorry, I encountered an error processing your database question."
        
    def _is_database_question(self, question):
        """Basic check if question is likely database related."""
        # List of database-related keywords
        db_keywords = [
            "database", "sql", "query", "table", "column", "row", "mysql", "postgresql", 
            "mongodb", "nosql", "schema", "index", "primary key", "foreign key", 
            "join", "select", "insert", "update", "delete", "where", "from", 
            "normalization", "transaction", "procedure", "function", "view", "trigger"
        ]
        
        # Convert to lowercase for case-insensitive matching
        question_lower = question.lower()
        
        # Check if any database keyword is in the question
        return any(keyword in question_lower for keyword in db_keywords)

if __name__ == "__main__":
    assistant = VoiceAssistant(
        whisper_model="tiny",  # Use tiny model for faster processing
        wake_word="computer"
    )
    assistant.start()