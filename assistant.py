import whisper
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import numpy as np
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import re
from langchain_core.documents import Document
from db_handler import get_db_connection, extract_schema, execute_query

class VoiceAssistant:
    def __init__(self, db_config, whisper_model, model):
        self.whisper_model = whisper_model
        self.llm_model = model
        self.output_parser = StrOutputParser()
        self.db_config = db_config
        self.vector_db = self._setup_vector_db()

    def _setup_vector_db(self):
        schema_texts = extract_schema(self.db_config)
        documents = [Document(page_content=text) for text in schema_texts]
        embeddings = OpenAIEmbeddings()
        embedded_docs = embeddings.embed_documents([doc.page_content for doc in documents])
        embedded_docs = [vec / np.linalg.norm(vec) for vec in embedded_docs]
        dimension = len(embedded_docs[0])
        index = faiss.IndexFlatIP(dimension)
        index.add(np.array(embedded_docs, dtype=np.float32))
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}
        return FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

    def transcribe_audio(self, audio_file):
        result = self.whisper_model.transcribe(audio_file)
        return result["text"].strip()

    def get_response(self, question):
        # Check for common greetings and pleasantries first
        greeting_patterns = [
            r'\b(hi|hello|hey|greetings|howdy)\b',
            r'\b(thank you|thanks)\b',
            r'\b(good morning|good afternoon|good evening)\b'
        ]
        
        for pattern in greeting_patterns:
            if re.search(pattern, question.lower()):
                return "Hello! I'm your database assistant. How can I help with your database queries today?"
        
        # Continue with normal processing for database queries
        retrieved_docs = self.vector_db.similarity_search_with_score(question, k=1)
        if not retrieved_docs or retrieved_docs[0][1] > 1.0:
            return "I can only answer questions about your database schema. Your query appears to be out of context."
        
        schema_text = retrieved_docs[0][0].page_content
        query = self._generate_query(question, schema_text)
        if not query:
            return "I couldn't generate a valid database query from your question. Could you rephrase it?"
        
        print("sql query", query)
        sql_response = execute_query(self.db_config, query)
        return self._generate_final_response(sql_response, question)

    def _generate_query(self, question, schema_text):
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
        
        chain = prompt | self.llm_model | self.output_parser
        response = chain.invoke({"question": question, "schema": schema_text})
        print("query generation response:", response)
        
        # Check if the model indicated this is not a database query
        if "NOT_DB_QUERY" in response:
            return ""
        
        return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    def _generate_final_response(self, sql_data, question):
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
        
        chain = prompt | self.llm_model | self.output_parser
        response = chain.invoke({"question": question, "sql_response": sql_data})
        return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()