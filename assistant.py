from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import numpy as np
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import re
import logging
import traceback
import json
import time
from typing import Dict, List, Any, Optional, Union
from langchain_core.documents import Document
from db_handler import get_db_connection, extract_schema, execute_query
from groq import GroqError, APIError, APIConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VoiceAssistantError(Exception):
    """Base exception class for VoiceAssistant errors"""
    pass

class DatabaseConnectionError(VoiceAssistantError):
    """Exception raised for database connection issues"""
    pass

class SchemaExtractionError(VoiceAssistantError):
    """Exception raised when schema extraction fails"""
    pass

class EmbeddingError(VoiceAssistantError):
    """Exception raised when document embedding fails"""
    pass

class TranscriptionError(VoiceAssistantError):
    """Exception raised when audio transcription fails"""
    pass

class QueryGenerationError(VoiceAssistantError):
    """Exception raised when SQL/MongoDB query generation fails"""
    pass

class QueryExecutionError(VoiceAssistantError):
    """Exception raised when query execution fails"""
    pass

class VoiceAssistant:
    def __init__(self, db_config, whisper_model, model, max_retries=3, retry_delay=1):
        """
        Initialize the VoiceAssistant with error handling and retry logic.
        
        Args:
            db_config: Database configuration dictionary
            whisper_model: Speech-to-text model for transcription
            model: LLM model for query generation and response formatting
            max_retries: Maximum number of retries for external API calls
            retry_delay: Delay between retries in seconds
        """
        self.whisper_model = whisper_model
        self.llm_model = model
        self.output_parser = StrOutputParser()
        self.db_config = db_config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.vector_db = None
        
        try:
            self.vector_db = self._setup_vector_db()
            logger.info("VoiceAssistant initialized successfully")
        except SchemaExtractionError as e:
            logger.error(f"Failed to extract schema: {str(e)}")
            # We'll initialize without vector_db and handle this in get_response
        except EmbeddingError as e:
            logger.error(f"Failed to create embeddings: {str(e)}")
            # We'll initialize without vector_db and handle this in get_response
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}\n{traceback.format_exc()}")
            # We'll initialize without vector_db and handle this in get_response

    def _setup_vector_db(self):
        """
        Set up the vector database with error handling.
        
        Returns:
            FAISS vector database
        
        Raises:
            SchemaExtractionError: If schema extraction fails
            EmbeddingError: If document embedding fails
        """
        try:
            # Extract schema with error handling
            schema_texts = self._extract_schema_with_retry()
            if not schema_texts:
                raise SchemaExtractionError("Failed to extract schema or schema is empty")
            
            # Create documents
            documents = [Document(page_content=text) for text in schema_texts]
            
            # Create embeddings with retry logic
            try:
                embeddings = OpenAIEmbeddings()
                embedded_docs = self._embed_documents_with_retry(embeddings, documents)
            except Exception as e:
                logger.error(f"Embedding error: {str(e)}")
                raise EmbeddingError(f"Error creating embeddings: {str(e)}")
            
            # Normalize embeddings
            embedded_docs = [vec / np.linalg.norm(vec) for vec in embedded_docs]
            
            # Create FAISS index
            dimension = len(embedded_docs[0])
            index = faiss.IndexFlatIP(dimension)
            index.add(np.array(embedded_docs, dtype=np.float32))
            
            # Create docstore
            docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
            index_to_docstore_id = {i: str(i) for i in range(len(documents))}
            
            logger.info(f"Vector database setup successful with {len(documents)} schema documents")
            return FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )
        except SchemaExtractionError:
            # Re-raise to be handled by caller
            raise
        except EmbeddingError:
            # Re-raise to be handled by caller
            raise
        except Exception as e:
            logger.error(f"Vector DB setup error: {str(e)}\n{traceback.format_exc()}")
            raise VoiceAssistantError(f"Error setting up vector database: {str(e)}")

    def _extract_schema_with_retry(self) -> List[str]:
        """
        Extract database schema with retry logic.
        
        Returns:
            List of schema text strings
        
        Raises:
            SchemaExtractionError: If schema extraction fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                return extract_schema(self.db_config)
            except Exception as e:
                logger.warning(f"Schema extraction attempt {attempt+1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Schema extraction failed after {self.max_retries} attempts")
                    raise SchemaExtractionError(f"Failed to extract schema after {self.max_retries} attempts: {str(e)}")

    def _embed_documents_with_retry(self, embeddings, documents) -> List[np.ndarray]:
        """
        Embed documents with retry logic.
        
        Args:
            embeddings: Embedding model
            documents: List of documents to embed
            
        Returns:
            List of embedded document vectors
            
        Raises:
            EmbeddingError: If embedding fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                return embeddings.embed_documents([doc.page_content for doc in documents])
            except Exception as e:
                logger.warning(f"Document embedding attempt {attempt+1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Document embedding failed after {self.max_retries} attempts")
                    raise EmbeddingError(f"Failed to embed documents after {self.max_retries} attempts: {str(e)}")

    def transcribe_audio(self, audio_file):
        """
        Transcribe audio file with error handling.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text
            
        Raises:
            TranscriptionError: If transcription fails
        """
        try:
            for attempt in range(self.max_retries):
                try:
                    result = self.whisper_model.transcribe(audio_file)
                    return result["text"].strip()
                except (GroqError, APIError, APIConnectionError) as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Transcription attempt {attempt+1} failed: {str(e)}")
                        time.sleep(self.retry_delay)
                    else:
                        raise TranscriptionError(f"Failed to transcribe after {self.max_retries} attempts: {str(e)}")
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}\n{traceback.format_exc()}")
            raise TranscriptionError(f"Error transcribing audio: {str(e)}")

    def get_response(self, question: str) -> str:
        """
        Get response to user question with comprehensive error handling.
        
        Args:
            question: User question or prompt
            
        Returns:
            Response text
        """
        try:
            # Validate input
            if not question or not isinstance(question, str):
                return "I need a valid question to assist you. Could you please try again?"
            
            # Check if vector_db was properly initialized
            if self.vector_db is None:
                return "I'm having trouble accessing the database schema. Please check your database configuration and try again."
            
            # Check for common greetings and pleasantries first
            greeting_patterns = [
                r'\b(hi|hello|hey|greetings|howdy)\b',
                r'\b(thank you|thanks)\b',
                r'\b(good morning|good afternoon|good evening)\b'
            ]
            
            for pattern in greeting_patterns:
                if re.search(pattern, question.lower()):
                    return "Hello! I'm your database assistant. How can I help with your database queries today?"
            
            # Perform similarity search with error handling
            try:
                retrieved_docs = self.vector_db.similarity_search_with_score(question, k=1)
                if not retrieved_docs or retrieved_docs[0][1] > 1.0:
                    return "I can only answer questions about your database schema. Your query appears to be out of context."
                
                schema_text = retrieved_docs[0][0].page_content
            except Exception as e:
                logger.error(f"Similarity search error: {str(e)}\n{traceback.format_exc()}")
                return "I'm having trouble finding relevant information in your database schema. Could you try a more specific question about your database?"
            
            # Generate query with error handling
            try:
                query = self._generate_query(question, schema_text)
                if not query:
                    return "I couldn't generate a valid database query from your question. Could you rephrase it to be more specific about the database information you're looking for?"
            except QueryGenerationError as e:
                logger.error(f"Query generation error: {str(e)}")
                return f"I encountered an error generating a database query: {str(e)}. Could you try rephrasing your question?"
            except Exception as e:
                logger.error(f"Unexpected query generation error: {str(e)}\n{traceback.format_exc()}")
                return "I had an unexpected issue generating a database query. Please try again with a clearer question about your database."
            
            # Execute query with error handling
            try:
                sql_response = execute_query(self.db_config, query)
            except Exception as e:
                logger.error(f"Query execution error: {str(e)}\n{traceback.format_exc()}")
                return f"I encountered an error executing the database query. The database returned: {str(e)}"
            
            # Generate final response with error handling
            try:
                return self._generate_final_response(sql_response, question)
            except Exception as e:
                logger.error(f"Response generation error: {str(e)}\n{traceback.format_exc()}")
                # Fallback to returning raw SQL response if formatting fails
                return f"Here are the raw results from your query (I had trouble formatting them nicely): {sql_response}"
                
        except Exception as e:
            logger.error(f"Unexpected error in get_response: {str(e)}\n{traceback.format_exc()}")
            return "I encountered an unexpected error while processing your question. Please try again or check your database connection."

    def _generate_query(self, question: str, schema_text: str) -> str:
        """
        Generate database query with error handling.
        
        Args:
            question: User question
            schema_text: Database schema text
            
        Returns:
            Generated query string
            
        Raises:
            QueryGenerationError: If query generation fails
        """
        try:
            # Select appropriate prompt based on database type
            if self.db_config['db_type'] in ['mysql', 'postgresql', 'sqlite', 'sqlserver']:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an expert SQL query generator that ONLY generates SQL for database-related questions.

        IMPORTANT: 
        - First, determine if the question is actually about querying the database or accessing data.
        - If the question is NOT about data retrieval or database operations (such as 'make a tea', 'what's the weather', etc.), respond ONLY with the exact text 'NOT_DB_QUERY'.
        - Do not attempt to generate SQL for non-database questions.
        - If the question is a database query, generate ONLY the SQL query with no explanations or comments.

        Follow the schema EXACTLY when writing queries - do not reference tables or columns that don't exist in the schema.

        Schema information:
        {schema}

        User question:
        {question}"""),
                    ("user", "")
                ])
            elif self.db_config['db_type'] == 'mongodb':
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an expert MongoDB query generator that ONLY generates queries for database-related questions.

        IMPORTANT: 
        - First, determine if the question is actually about querying the database or accessing data.
        - If the question is NOT about data retrieval or database operations (such as 'make a tea', 'what's the weather', etc.), respond ONLY with the exact text 'NOT_DB_QUERY'.
        - Do not attempt to generate MongoDB queries for non-database questions.
        - If the question is a database query, generate ONLY the MongoDB query in the format specified below with no explanations.

        Format for valid queries: {'collection': '...', 'operation': 'find', 'query': {...}, 'projection': {...}}

        Schema information:
        {schema}

        User question:
        {question}"""),
                    ("user", "")
                ])
            else:
                raise QueryGenerationError(f"Unsupported database type: {self.db_config['db_type']}")
            
            # Execute query generation with retry logic
            for attempt in range(self.max_retries):
                try:
                    chain = prompt | self.llm_model | self.output_parser
                    response = chain.invoke({"question": question, "schema": schema_text})
                    
                    # Check if the model indicated this is not a database query
                    if "NOT_DB_QUERY" in response:
                        logger.info("Model determined the query is not database-related")
                        return ""
                    
                    # Validate the query is not empty after cleaning
                    cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                    if not cleaned_response:
                        raise QueryGenerationError("Generated query is empty after cleaning")
                    
                    logger.info("Successfully generated database query")
                    return cleaned_response
                    
                except (GroqError, APIError, APIConnectionError) as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Query generation attempt {attempt+1} failed: {str(e)}")
                        time.sleep(self.retry_delay)
                    else:
                        raise QueryGenerationError(f"Failed to generate query after {self.max_retries} attempts: {str(e)}")
                        
        except QueryGenerationError:
            # Re-raise to be handled by caller
            raise
        except Exception as e:
            logger.error(f"Unexpected error in query generation: {str(e)}\n{traceback.format_exc()}")
            raise QueryGenerationError(f"Unexpected error generating query: {str(e)}")

    def _generate_final_response(self, sql_data: Union[str, Dict, List], question: str) -> str:
        """
        Generate final natural language response with error handling.
        
        Args:
            sql_data: SQL query result data
            question: Original user question
            
        Returns:
            Formatted response text
        """
        try:
            # Validate and prepare SQL data
            if isinstance(sql_data, (dict, list)):
                sql_response_str = json.dumps(sql_data, indent=2)
            else:
                sql_response_str = str(sql_data)
            
            # Create a more contextually aware prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful database assistant. Respond to the user's question based on the context:

        1. If the question is a greeting (like "hello", "hi", "good morning", "thank you", etc.), respond with a friendly greeting.

        2. If SQL data is provided, generate a natural language response that explains the data results clearly.

        3. If the question is about database schema and SQL data is available, answer based on the schema information.

        4. If the query is unrelated to databases or the available schema, politely inform the user that the question is out of context and you can only help with database-related queries.

        SQL Query Results: {sql_response}
        User Question: {question}"""),
                ("user", "")
            ])
            
            # Execute response generation with retry logic
            for attempt in range(self.max_retries):
                try:
                    chain = prompt | self.llm_model | self.output_parser
                    response = chain.invoke({"question": question, "sql_response": sql_response_str})
                    
                    # Clean and validate response
                    cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                    if not cleaned_response:
                        raise Exception("Generated response is empty after cleaning")
                    
                    logger.info("Successfully generated final response")
                    return cleaned_response
                    
                except (GroqError, APIError, APIConnectionError) as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Response generation attempt {attempt+1} failed: {str(e)}")
                        time.sleep(self.retry_delay)
                    else:
                        # On final failure, return a simple formatted response with the raw data
                        logger.error(f"Final response generation failed after {self.max_retries} attempts: {str(e)}")
                        return f"Here are the results from your database query (I had trouble creating a detailed explanation):\n\n{sql_response_str}"
                        
        except Exception as e:
            logger.error(f"Error generating final response: {str(e)}\n{traceback.format_exc()}")
            # Fallback to returning raw SQL response
            return f"Here are the raw results from your query (I encountered an error while formatting the response):\n\n{sql_data}"