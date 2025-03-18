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
        retrieved_docs = self.vector_db.similarity_search_with_score(question, k=1)
        if not retrieved_docs or retrieved_docs[0][1] > 1.0:
            return "Query is out of schema context!"
        schema_text = retrieved_docs[0][0].page_content
        query = self._generate_query(question, schema_text)
        if not query:
            return "Could not generate a valid query."
        sql_response = execute_query(self.db_config, query)
        return self._generate_final_response(sql_response, question)

    def _generate_query(self, question, schema_text):
        if self.db_config['db_type'] in ['mysql', 'postgresql', 'sqlite', 'sqlserver']:
            prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert SQL query generator. Based on the provided database schema, generate only the SQL query for the given user question. Do not include any explanations, additional text, or comments."),
            ("user", "Schema: {schema}\nQuestion: {question}")
            ])
        elif self.db_config['db_type'] == 'mongodb':
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Generate a MongoDB query in JSON format based on the schema: {schema}\nQuestion: {question}\nFormat: {'collection': '...', 'operation': 'find', 'query': {...}, 'projection': {...}}"),
                ("user", "")
            ])
        chain = prompt | self.llm_model | self.output_parser
        response = chain.invoke({"question": question, "schema": schema_text})
        # print(response)
        return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    def _generate_final_response(self, sql_data, question):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a natural language response based on query results: {schema}\nQuestion: {question}"),
            ("user", "")
        ])
        chain = prompt | self.llm_model | self.output_parser
        response = chain.invoke({"question": question, "schema": sql_data})
        return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()