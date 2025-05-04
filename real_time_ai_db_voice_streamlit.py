# import streamlit as st
# import whisper
# import pyaudio
# import wave
# import threading  # CHANGED: Added threading import
# import time
# import os
# import requests
# import pvporcupine
# import struct
# from queue import Queue
# import numpy as np
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_community.llms import Ollama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import OpenAIEmbeddings
# import mysql.connector
# import pyttsx3
# from langchain_core.documents import Document
# from langchain_community.vectorstores.faiss import FAISS
# import numpy as np
# import faiss
# from langchain_community.docstore.in_memory import InMemoryDocstore
# import re
# import tempfile
# from io import BytesIO
# import sounddevice as sd
# import soundfile as sf
# from PIL import Image

# # Disable PyTorch module watching to prevent errors
# st._config.get_option('server.runOnSave')
# st._config.set_option('server.runOnSave', False)

# # Load environment variables
# load_dotenv()
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# class VoiceAssistant:
#     def __init__(self, whisper_model="tiny", sample_rate=16000, chunk_size=1024, 
#                  wake_word="computer", db_config=None, model="deepseek-r1:1.5b", 
#                  llm_endpoint="http://localhost:11434/api/generate"):
#         # Configuration
#         self.FORMAT = pyaudio.paInt16
#         self.CHANNELS = 1
#         self.RATE = sample_rate
#         self.CHUNK = chunk_size
#         self.whisper_model = whisper_model
#         self.llm_model = Ollama(model=model)
#         self.llm_endpoint = llm_endpoint
#         self.wake_word = wake_word
#         self.output_parser = StrOutputParser()
#         self.db_config = db_config
        
#         # State variables
#         self.recording = False
#         self.running = True
#         self.audio_queue = Queue()
        
#         # Initialize components
#         self.audio = pyaudio.PyAudio()
#         print(f"Loading Whisper model '{whisper_model}'...")
#         self.model = whisper.load_model(whisper_model)
#         print("Whisper model loaded.")
        
#         # Initialize Porcupine wake word detector if access key is available
#         if os.getenv("PVPORCUPINE_ACCESS_KEY"):
#             self.porcupine = pvporcupine.create(
#                 access_key=os.getenv("PVPORCUPINE_ACCESS_KEY"), 
#                 keywords=[wake_word]
#             )
#         else:
#             print("Warning: PVPORCUPINE_ACCESS_KEY not found. Wake word detection disabled.")
#             self.porcupine = None
        
#         # Initialize FAISS for schema-based retrieval
#         if self.db_config:
#             self.vector_db = self._setup_vector_db()
#         else:
#             print("Warning: No database configuration provided. Database functionality disabled.")
#             self.vector_db = None
        
#     def _setup_vector_db(self):
#         """Extracts database schema and stores it in a vector database."""
#         try:
#             schema_texts = self._extract_db_schema()
#             documents = [Document(page_content=text) for text in schema_texts]
#             embeddings = OpenAIEmbeddings()
#             embedded_docs = embeddings.embed_documents([doc.page_content for doc in documents])

#             # Normalize embeddings for cosine similarity
#             embedded_docs = [vec / np.linalg.norm(vec) for vec in embedded_docs]

#             # Create FAISS index with IndexFlatIP for cosine similarity
#             dimension = len(embedded_docs[0])
#             index = faiss.IndexFlatIP(dimension)
#             index.add(np.array(embedded_docs, dtype=np.float32))

#             # Create document store and mapping
#             docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
#             index_to_docstore_id = {i: str(i) for i in range(len(documents))}

#             # Instantiate FAISS with the correct arguments
#             return FAISS(
#                 embedding_function=embeddings,
#                 index=index,
#                 docstore=docstore,
#                 index_to_docstore_id=index_to_docstore_id
#             )
#         except Exception as e:
#             print(f"Error setting up vector database: {e}")
#             return None
    
#     def _extract_db_schema(self):
#         """Fetches schema information from MySQL."""
#         try:
#             conn = mysql.connector.connect(**self.db_config)
#             cursor = conn.cursor()
#             cursor.execute("SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = %s", (self.db_config['database'],))
#             schema_info = cursor.fetchall()
#             cursor.close()
#             conn.close()
            
#             schema_texts = []
#             table_structure = {}
#             for table, column, dtype in schema_info:
#                 if table not in table_structure:
#                     table_structure[table] = []
#                 table_structure[table].append(f"{column} ({dtype})")
            
#             for table, columns in table_structure.items():
#                 schema_texts.append(f"Table: {table}, Columns: {', '.join(columns)}")
            
#             return schema_texts
#         except Exception as e:
#             print(f"Error extracting database schema: {e}")
#             return []
    
#     def get_llm_response(self, question):
#         """Retrieve response from LLM or execute SQL query if applicable."""
#         if not self.vector_db:
#             return "Database functionality is not available."
            
#         retrieved_docs_with_scores = self.vector_db.similarity_search_with_score(question, k=1)
#         if not retrieved_docs_with_scores:
#             return "Query is out of schema context!"
        
#         doc, score = retrieved_docs_with_scores[0]
#         if score > 1.0:  # L2 distance threshold, lower is better (tune this value)
#             return "Query is not sufficiently related to the schema."
        
#         schema_text = doc.page_content
#         print(schema_text)
#         raw_response = self._generate_sql(question, schema_text)
#         sql_query = self._clean_sql_output(raw_response)
#         print(sql_query)
#         # sql_query = self._clean_sql_output("SELECT name FROM users")
        
#         if not self.is_likely_sql(sql_query):
#             return "Could not generate a valid SQL query."
#         print(sql_query)
#         sql_response = self._execute_sql(sql_query)
#         return self._generate_final_response(sql_response,question)
    
#     def _generate_final_response(self, sql_data, question):
#         """Generate a natural language response from SQL query results.
        
#         Args:
#             sql_data (str): The results from the SQL query execution
#             question (str): The original natural language question asked by the user
        
#         Returns:
#             str: A natural language response answering the user's question
#         """
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are a database assistant that generates clear, concise responses based on SQL query results.

#     Your task is to:
#     1. Analyze the SQL query results provided
#     2. Understand the user's original question
#     3. Format the response in a natural, conversational way
#     4. Highlight key insights from the data
#     5. Include specific numbers and facts from the results
#     6. Keep responses concise and directly relevant to the question

#     If the data shows "No results found" or contains an error message, explain what this means in simple terms.

#     Respond in a helpful, direct manner without unnecessary explanations or SQL terminology unless requested."""),
#             ("user", """Original question: {question}

#     SQL query results:
#     {schema}

#     Please provide a natural language response that answers my question based on these results.""")
#         ])
        
#         chain = prompt | self.llm_model | self.output_parser
#         response = chain.invoke({"question": question, "schema": sql_data})
#         return self._clean_response_output(response)
    
#     def _generate_sql(self, question, schema_text):
#         """Generate an SQL query from natural language using schema context."""
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are an expert SQL query generator. Based on the provided database schema, generate only the SQL query for the given user question. Do not include any explanations, additional text, or comments."),
#             ("user", "Schema: {schema}\nQuestion: {question}")
#         ])
        
#         chain = prompt | self.llm_model | self.output_parser
#         return chain.invoke({"question": question, "schema": schema_text})
    
#     def _clean_response_output(self, response):
#         """Remove any unwanted tags or explanations from the response."""
#         # Remove <think> tags and their contents
#         cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.IGNORECASE | re.DOTALL)
#         return cleaned
    
#     def _clean_sql_output(self, response):
#         """Remove any unwanted tags or explanations from the response."""
#         # Remove <think> tags and their contents
#         cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.IGNORECASE | re.DOTALL)
#         # Match any SQL command (a word followed by SQL-like content)
#         match = re.search(r"\b[A-Za-z]+\b.*", cleaned, re.IGNORECASE | re.DOTALL)
#         return match.group(0).strip() if match else cleaned.strip()
    
#     def is_likely_sql(self, query):
#         """Basic check to see if the query is likely an SQL statement."""
#         query = query.strip().upper()
#         return query.startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP'))

#     def _execute_sql(self, query):
#         """Execute SQL query and return results."""
#         try:
#             conn = mysql.connector.connect(**self.db_config)
#             cursor = conn.cursor()
#             cursor.execute(query)
#             results = cursor.fetchall()
            
#             if results:
#                 # Format results as a more readable string
#                 column_names = [i[0] for i in cursor.description]
#                 formatted_results = []
#                 formatted_results.append(" | ".join(column_names))
#                 formatted_results.append("-" * (len(" | ".join(column_names))))
                
#                 for row in results[:20]:  # Show up to 20 rows
#                     formatted_results.append(" | ".join(str(cell) for cell in row))
                
#                 result_text = "\n".join(formatted_results)
#                 cursor.close()
#                 conn.close()
#                 return result_text
#             else:
#                 cursor.close()
#                 conn.close()
#                 return "No results found."
#         except Exception as e:
#             return f"Error executing query: {e}"
    
#     def transcribe_audio(self, audio_file):
#         """Transcribe audio file using Whisper model."""
#         try:
#             result = self.model.transcribe(audio_file)
#             print(result["text"].strip())
#             return result["text"].strip()
#         except Exception as e:
#             print(f"Error in transcription: {e}")
#             return ""

# # ADDED: New function for threaded recording
# def record_audio_thread(duration, sample_rate, audio_queue):
#     """Record audio in a separate thread and put the data in a queue."""
#     audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
#     sd.wait()  # Wait for recording to complete
#     audio_queue.put(audio_data)

# def record_audio(duration, sample_rate):  # Note: This function is no longer used but left in code
#     """Record audio and return the data - moved outside of the Streamlit callback"""
#     audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
#     return audio_data

# def main():
#     # Set page config first
#     st.set_page_config(
#         page_title="Voice Database Assistant",
#         page_icon="🎤",
#         layout="wide"
#     )
    
#     st.title("Voice Database Assistant")
    
#     # Initialize session state variables
#     if 'assistant' not in st.session_state:
#         st.session_state.assistant = None
#     if 'recording' not in st.session_state:
#         st.session_state.recording = False
#     if 'audio_data' not in st.session_state:
#         st.session_state.audio_data = None
#     if 'response' not in st.session_state:
#         st.session_state.response = ""
#     if 'transcription' not in st.session_state:
#         st.session_state.transcription = ""
#     if 'processing_complete' not in st.session_state:
#         st.session_state.processing_complete = False
#     # ADDED: Initialize audio queue in session state
#     if 'audio_queue' not in st.session_state:
#         st.session_state.audio_queue = Queue()
    
#     # Sidebar for configuration
#     with st.sidebar:
#         st.header("Configuration")
        
#         # Database configuration
#         st.subheader("Database Settings")
#         db_host = st.text_input("Host", "localhost")
#         db_user = st.text_input("User", "root")
#         db_password = st.text_input("Password", "", type="password")
#         db_name = st.text_input("Database", "testing")
        
#         # Model configuration
#         st.subheader("Model Settings")
#         whisper_model = st.selectbox("Whisper Model", ["tiny", "base", "small", "medium"], index=0)
#         llm_model = st.text_input("LLM Model", "deepseek-r1:1.5b")
        
#         # Wake word configuration
#         st.subheader("Wake Word Settings")
#         wake_word = st.text_input("Wake Word", "computer")
        
#         # API Keys
#         st.subheader("API Keys")
#         pvporcupine_key = st.text_input("Picovoice Porcupine Key", os.getenv("PVPORCUPINE_ACCESS_KEY", ""), type="password")
#         openai_key = st.text_input("OpenAI API Key", os.getenv("OPENAI_API_KEY", ""), type="password")
        
#         # Save configuration and initialize assistant
#         if st.button("Save Configuration"):
#             os.environ["PVPORCUPINE_ACCESS_KEY"] = pvporcupine_key
#             os.environ["OPENAI_API_KEY"] = openai_key
            
#             # Initialize or update the assistant with new configuration
#             db_config = {
#                 "host": db_host,
#                 "user": db_user,
#                 "password": db_password,
#                 "database": db_name
#             }
            
#             with st.spinner("Initializing assistant..."):
#                 st.session_state.assistant = VoiceAssistant(
#                     whisper_model=whisper_model,
#                     wake_word=wake_word,
#                     db_config=db_config,
#                     model=llm_model
#                 )
            
#             st.success("Configuration saved and assistant initialized!")

#     # Main content area
#     tabs = st.tabs(["Voice Interaction", "Text Interaction", "Database Explorer"])
    
#     # Voice Interaction Tab
#     # Voice Interaction Tab
#     # Voice Interaction Tab
#     with tabs[0]:
#         st.header("Voice Interaction")
        
#         # Check if assistant is initialized
#         if st.session_state.assistant is None:
#             st.warning("Please save configuration in the sidebar first to initialize the assistant.")
#         else:
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.write("Press the button and speak your database query.")
                
#                 # Recording state management
#                 start_recording = st.button("🎤 Start Recording", disabled=st.session_state.recording)
                
#                 if start_recording:
#                     st.session_state.recording = True
#                     st.session_state.processing_complete = False
#                     st.session_state.transcription = ""
#                     st.session_state.response = ""
#                     st.rerun()
                
#                 if st.session_state.recording:
#                     stop_button = st.button("⏹️ Stop Recording")
#                     status_placeholder = st.empty()
#                     status_placeholder.warning("Recording... Speak your database query.")
#                     progress_placeholder = st.empty()
                    
#                     # Record audio
#                     duration = 5  # seconds
#                     sample_rate = 16000
                    
#                     # Initialize audio recording
#                     if st.session_state.audio_data is None:
#                         st.session_state.audio_data = sd.rec(
#                             int(duration * sample_rate), 
#                             samplerate=sample_rate, 
#                             channels=1, 
#                             dtype='int16'
#                         )
                        
#                         # Add progress bar for recording duration
#                         progress_bar = progress_placeholder.progress(0)
#                         for i in range(100):
#                             # Simulate progress during recording
#                             time.sleep(duration/100)
#                             progress_bar.progress(i + 1)
                        
#                         sd.wait()  # Wait for recording to complete
                    
#                     if stop_button or st.session_state.audio_data is not None:
#                         status_placeholder.info("Processing audio...")
                        
#                         # Save recording to temporary file
#                         with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
#                             temp_filename = tmpfile.name
#                             sf.write(temp_filename, st.session_state.audio_data, sample_rate)
                        
#                         # Transcribe audio
#                         with st.spinner("Transcribing..."):
#                             st.session_state.transcription = st.session_state.assistant.transcribe_audio(temp_filename)
                        
#                         if st.session_state.transcription:
#                             status_placeholder.success(f"Transcription: {st.session_state.transcription}")
                            
#                             # Process query
#                             with st.spinner("Processing query..."):
#                                 st.session_state.response = st.session_state.assistant.get_llm_response(st.session_state.transcription)
#                         else:
#                             status_placeholder.error("Could not transcribe audio. Please try again.")
                        
#                         # Clean up
#                         try:
#                             os.remove(temp_filename)
#                         except Exception as e:
#                             print(f"Error removing temp file: {e}")
                        
#                         # Reset state for next recording
#                         st.session_state.recording = False
#                         st.session_state.audio_data = None
#                         st.session_state.processing_complete = True
#                         st.rerun()
            
#             with col2:
#                 st.write("Query results will appear here:")
                
#                 if st.session_state.response:
#                     st.code(st.session_state.response)
        
#     # Text Interaction Tab
#     with tabs[1]:
#         st.header("Text Interaction")
        
#         # Check if assistant is initialized
#         if st.session_state.assistant is None:
#             st.warning("Please save configuration in the sidebar first to initialize the assistant.")
#         else:
#             user_query = st.text_area("Enter your database query:", height=150)
            
#             if st.button("Submit Query"):
#                 if user_query:
#                     with st.spinner("Processing query..."):
#                         response = st.session_state.assistant.get_llm_response(user_query)
#                         st.session_state.response = response
#                         st.code(response)
#                 else:
#                     st.warning("Please enter a query first.")
    
#     # Database Explorer Tab
#     with tabs[2]:
#         st.header("Database Explorer")
        
#         # Database connection
#         try:
#             db_config = {
#                 "host": db_host,
#                 "user": db_user,
#                 "password": db_password,
#                 "database": db_name
#             }
            
#             conn = mysql.connector.connect(**db_config)
#             cursor = conn.cursor()
            
#             # Get table list
#             cursor.execute("SHOW TABLES")
#             tables = [table[0] for table in cursor.fetchall()]
            
#             if tables:
#                 selected_table = st.selectbox("Select a table to explore:", tables)
                
#                 if selected_table:
#                     # Get table columns
#                     cursor.execute(f"DESCRIBE {selected_table}")
#                     columns = [col[0] for col in cursor.fetchall()]
                    
#                     # Display table preview
#                     cursor.execute(f"SELECT * FROM {selected_table} LIMIT 10")
#                     rows = cursor.fetchall()
                    
#                     if rows:
#                         st.write(f"Preview of table '{selected_table}':")
                        
#                         # Create DataFrame-like display
#                         st.write("| " + " | ".join(columns) + " |")
#                         st.write("| " + " | ".join(["---"] * len(columns)) + " |")
                        
#                         for row in rows:
#                             st.write("| " + " | ".join([str(cell) for cell in row]) + " |")
#                     else:
#                         st.info(f"Table '{selected_table}' is empty.")
                        
#                     # Custom SQL query option
#                     st.subheader("Run custom SQL query")
#                     custom_query = st.text_area("Enter SQL query:", f"SELECT * FROM {selected_table} LIMIT 100;")
                    
#                     if st.button("Run Query"):
#                         with st.spinner("Running query..."):
#                             try:
#                                 cursor.execute(custom_query)
#                                 results = cursor.fetchall()
                                
#                                 if results:
#                                     # Get column names
#                                     column_names = [i[0] for i in cursor.description]
                                    
#                                     # Display results
#                                     st.write("Query results:")
                                    
#                                     # Create DataFrame-like display
#                                     st.write("| " + " | ".join(column_names) + " |")
#                                     st.write("| " + " | ".join(["---"] * len(column_names)) + " |")
                                    
#                                     for row in results[:100]:  # Limit to 100 rows for display
#                                         st.write("| " + " | ".join([str(cell) for cell in row]) + " |")
#                                 else:
#                                     st.info("Query returned no results.")
#                             except Exception as e:
#                                 st.error(f"Error executing query: {e}")
#             else:
#                 st.warning(f"No tables found in database '{db_name}'.")
                
#             # Close resources
#             cursor.close()
#             conn.close()
            
#         except Exception as e:
#             st.error(f"Database connection error: {e}")
#             st.info("Please check your database configuration in the sidebar.")

#     # Footer
#     st.markdown("---")
#     st.markdown("Voice Database Assistant | Built with Streamlit")

# if __name__ == "__main__":
#     main()

from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
