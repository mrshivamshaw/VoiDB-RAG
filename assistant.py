import re
import logging
import traceback
import json
import time
from typing import Dict, List, Any, Optional, Union
from db_handler import get_db_connection, extract_schema, execute_query
from groq import GroqError, APIError, APIConnectionError
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import hashlib
load_dotenv()

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
    def __init__(self, db_config, groq_model, max_retries=3, retry_delay=1):
        """
        Initialize the VoiceAssistant with error handling and retry logic.
        
        Args:
            db_config: Database configuration dictionary
            groq_model: LLM model for query generation and response formatting
            max_retries: Maximum number of retries for external API calls
            retry_delay: Delay between retries in seconds
        """
        self.llm_model = groq_model
        self.db_config = db_config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.model_name = "llama3-8b-8192"
        self.pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        self.pinecone_index_name = os.environ.get('PINECONE_INDEX_NAME')

        # Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.index = self.pc.Index(self.pinecone_index_name)
            logger.info("Pinecone client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
        
        try:
            self._setup_pinecone_vectors()
            logger.info("VoiceAssistant initialized successfully with Pinecone")
        except SchemaExtractionError as e:
            logger.error(f"Failed to extract schema: {str(e)}")
        except EmbeddingError as e:
            logger.error(f"Failed to create embeddings: {str(e)}")
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}\n{traceback.format_exc()}")

    def _setup_pinecone_vectors(self):
        """
        Set up Pinecone vectors with schema data.
        
        Raises:
            SchemaExtractionError: If schema extraction fails
            EmbeddingError: If document embedding fails
        """
        try:
            # Extract schema with error handling
            schema_texts = self._extract_schema_with_retry()
            
            if not schema_texts:
                raise SchemaExtractionError("Failed to extract schema or schema is empty")
            
            # Combine all schema texts into a single string
            combined_schema_text = "\n".join(schema_texts)
            logger.info(f"Combined schema text length: {len(combined_schema_text)} characters")
            
            # Check if a vector for the combined schema already exists
            vector_id = self._generate_vector_id(combined_schema_text)
            result = self.index.fetch(ids=[vector_id])
            if result.vectors:
                logger.info("Schema vector already exists in Pinecone")
                return
            
            # Upsert the combined schema
            self._upsert_with_inference_api([combined_schema_text])
            logger.info("Successfully set up Pinecone vector for combined schema")
                
        except SchemaExtractionError:
            raise
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Pinecone setup error: {str(e)}\n{traceback.format_exc()}")
            raise VoiceAssistantError(f"Error setting up Pinecone vectors: {str(e)}")

    def _upsert_with_inference_api(self, schema_texts: List[str]):
        """
        Upsert a single document for the combined schema using Pinecone's Inference API.
        
        Args:
            schema_texts: List containing a single combined schema text
        """
        try:
            # Expect a single schema text (combined)
            if len(schema_texts) != 1:
                raise EmbeddingError("Expected a single combined schema text")
            
            combined_schema_text = schema_texts[0]
            print("combined ",combined_schema_text)
            # Generate embedding for the combined schema
            response = self.pc.inference.embed(
                model="llama-text-embed-v2",  # Match the model used in similarity search
                inputs=[combined_schema_text],
                parameters={"input_type": "passage"}
            )
            
            # Extract the embedding
            embedding = response.data[0]['values']
            logger.info(f"Generated embedding for combined schema: dimension {len(embedding)}")
            
            # Create vector dictionary
            vector_id = self._generate_vector_id(combined_schema_text)
            vector = {
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'text': combined_schema_text,
                    'type': 'schema'
                }
            }
            
            # Upsert the single vector
            self.index.upsert(vectors=[vector])
            logger.info("Successfully upserted combined schema vector")
            
        except Exception as e:
            logger.error(f"Upsert failed: {str(e)}")
            raise EmbeddingError(f"Failed to upsert combined schema vector: {str(e)}")
                # Don't raise - the assistant can still work without pre-stored vectors


    def _generate_vector_id(self, text: str) -> str:
        """Generate a unique ID for a vector based on text content."""
        return hashlib.md5(text.encode()).hexdigest()

    def _check_existing_vectors(self, schema_texts: List[str]) -> bool:
        """Check if vectors for the current schema already exist in Pinecone."""
        try:
            # Check if any of the schema text vectors exist
            for text in schema_texts[:1]:  # Check first one as sample
                vector_id = self._generate_vector_id(text)
                result = self.index.fetch(ids=[vector_id])
                if result.vectors:
                    return True
            return False
        except Exception as e:
            logger.warning(f"Error checking existing vectors: {str(e)}")
            return False

    def _similarity_search_pinecone(self, query: str, k: int = 1):
        """
        Perform similarity search using Pinecone Inference API.
        """
        try:
            # Generate embeddings for the query using Pinecone's Inference API
            response = self.pc.inference.embed(
                model="llama-text-embed-v2",  # Replace with your embedding model
                inputs=[query],
                parameters={
                    "input_type": "query"
                }
            )

            print(response)
            
            # Extract the embedding (list of floats)
            query_embedding = response.data[0]['values']
            
            # Perform similarity search with the embedding
            search_results = self.index.query(
                vector=query_embedding,  # Pass the embedding, not the raw text
                top_k=k,
                include_values=False,
                include_metadata=True,
                filter={'type': 'schema'}  # Only search schema documents
            )
            
            # Format results
            results = []
            matches = search_results.matches if hasattr(search_results, 'matches') else []
            
            for match in matches:
                doc_content = match.metadata.get('text', '') if match.metadata else ''
                similarity_score = match.score if hasattr(match, 'score') else 0.0
                
                class MockDocument:
                    def __init__(self, content):
                        self.page_content = content
                
                results.append((MockDocument(doc_content), similarity_score))
            
            logger.info(f"Found {len(results)} similar documents with scores: {[r[1] for r in results]}")
            return results
            
        except Exception as e:
            logger.error(f"Pinecone similarity search error: {str(e)}")
            # Return empty results instead of crashing
            return []
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
            

            # Check for common greetings and pleasantries first
            greeting_patterns = [
                r'\b(hi|hello|hey|greetings|howdy)\b',
                r'\b(thank you|thanks)\b',
                r'\b(good morning|good afternoon|good evening)\b'
            ]
            
            for pattern in greeting_patterns:
                if re.search(pattern, question.lower()):
                    return "Hello! I'm your database assistant. How can I help with your database queries today?"
            
            # Perform similarity search with Pinecone
            try:
                retrieved_docs = self._similarity_search_pinecone(question, k=1)
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
        Generate database query with error handling using Groq SDK.
        
        Args:
            question: User question
            schema_text: Combined database schema text
            
        Returns:
            Generated query string
            
        Raises:
            QueryGenerationError: If query generation fails
        """
        try:
            # Preprocess query to handle common grammatical issues
            question = question.lower().replace("student", "students").strip()
            logger.info(f"Preprocessed question: {question}")
            
            # Select appropriate system prompt
            if self.db_config['db_type'] in ['mysql', 'postgresql', 'sqlite', 'sqlserver']:
                system_prompt = """You are an expert SQL query generator that generates SQL for database-related questions.

        IMPORTANT:
        - The schema provided contains the ENTIRE database schema (all tables and columns).
        - Determine if the question is about querying the database (e.g., retrieving, filtering, or aggregating data).
        - If the question is NOT database-related (e.g., 'make tea', 'what's the weather'), respond ONLY with 'NOT_DB_QUERY'.
        - If the question is database-related, generate ONLY the SQL query with no explanations.
        - Use the schema EXACTLY, referencing only existing tables and columns.
        - For queries about 'all students' or similar, assume the 'students' table is relevant unless otherwise specified.

        Schema:
        {schema}

        User question:
        {question}"""
            elif self.db_config['db_type'] == 'mongodb':
                system_prompt = """You are an expert MongoDB query generator that generates queries for database-related questions.

        IMPORTANT:
        - The schema provided contains the ENTIRE database schema (all collections and fields).
        - Determine if the question is about querying the database (e.g., retrieving, filtering, or aggregating data).
        - If the question is NOT database-related (e.g., 'make tea', 'what's the weather'), respond ONLY with 'NOT_DB_QUERY'.
        - If the question is database-related, generate ONLY the MongoDB query in the format: {'collection': '...', 'operation': 'find', 'query': {...}, 'projection': {...}}.
        - Use the schema EXACTLY, referencing only existing collections and fields.

        Schema:
        {schema}

        User question:
        {question}"""
            else:
                raise QueryGenerationError(f"Unsupported database type: {self.db_config['db_type']}")
            
            formatted_prompt = system_prompt.format(schema=schema_text, question=question)
            
            # Execute query generation with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = self.llm_model.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": formatted_prompt},
                            {"role": "user", "content": "Generate the query based on the above question and schema."}
                        ],
                        temperature=0.1,
                        max_tokens=1000,
                        top_p=1,
                        stream=False
                    )
                    
                    generated_query = response.choices[0].message.content.strip()
                    logger.info(f"Generated query: {generated_query}")
                    
                    if "NOT_DB_QUERY" in generated_query:
                        logger.info("Model determined the query is not database-related")
                        return ""
                    
                    cleaned_response = re.sub(r"<think>.*?</think>", "", generated_query, flags=re.DOTALL).strip()
                    if not cleaned_response:
                        raise QueryGenerationError("Generated query is empty after cleaning")
                    
                    return cleaned_response
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Query generation attempt {attempt+1} failed: {str(e)}")
                        time.sleep(self.retry_delay)
                    else:
                        raise QueryGenerationError(f"Failed to generate query after {self.max_retries} attempts: {str(e)}")
                        
        except QueryGenerationError:
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
            prompt = """You are a helpful database assistant. Respond to the user's question based on the context:

        1. If the question is a greeting (like "hello", "hi", "good morning", "thank you", etc.), respond with a friendly greeting.

        2. If SQL data is provided, generate a natural language response that explains the data results clearly.

        3. If the question is about database schema and SQL data is available, answer based on the schema information.

        4. If the query is unrelated to databases or the available schema, politely inform the user that the question is out of context and you can only help with database-related queries.
        
        5. Give response in html and tailwindcss format.

        SQL Query Results: {sql_response}

        User Question: {question}
        """
            formatted_prompt = prompt.format(sql_response=sql_response_str, question=question)
            # Execute response generation with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = self.llm_model.chat.completions.create(
                        model=self.model_name,  # e.g., "llama3-8b-8192" or your preferred model
                        messages=[
                            {"role": "system", "content": formatted_prompt},
                            {"role": "user", "content": "Generate the query based on the above question and schema."}
                        ],
                        temperature=0.1,
                        max_tokens=1000,
                        top_p=1,
                        stream=False
                    )
                    
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