from sqlalchemy import create_engine, text
from pymongo import MongoClient
import mysql.connector
import json
from cryptography.fernet import Fernet
import os

# # Encryption key (generate once and store securely)
# key = Fernet.generate_key()  # In production, store this securely
# print("Encryption key:", key.decode())


key = os.getenv("FERNET_KEY")
if not key:
    raise ValueError("FERNET_KEY environment variable not set")
    
# Convert the string back to bytes if stored as string
if isinstance(key, str):
    key = key.encode()
cipher = Fernet(key)

def encrypt_password(password: str) -> str:
    return cipher.encrypt(password.encode()).decode()

def decrypt_password(encrypted_password: str) -> str:
    return cipher.decrypt(encrypted_password.encode()).decode()

def get_db_connection(db_config):
    db_type = db_config['db_type']
    password = decrypt_password(db_config['encrypted_password'])
    
    if db_type in ['mysql', 'postgresql', 'sqlite', 'sqlserver']:
        if db_type == 'mysql':
            url = f"mysql+mysqlconnector://{db_config['username']}:{password}@{db_config['host']}:{db_config['port']}/{db_config['database_name']}"
        elif db_type == 'postgresql':
            url = f"postgresql+psycopg2://{db_config['username']}:{password}@{db_config['host']}:{db_config['port']}/{db_config['database_name']}"
        elif db_type == 'sqlite':
            url = f"sqlite:///{db_config['database_name']}"
        elif db_type == 'sqlserver':
            url = f"mssql+pyodbc://{db_config['username']}:{password}@{db_config['host']}:{db_config['port']}/{db_config['database_name']}?driver=ODBC+Driver+17+for+SQL+Server"
        return create_engine(url)
    elif db_type == 'mongodb':
        return MongoClient(f"mongodb://{db_config['username']}:{password}@{db_config['host']}:{db_config['port']}")[db_config['database_name']]
    else:
        raise ValueError("Unsupported database type")

def extract_schema(db_config):
    db_type = db_config['db_type']
    if db_type in ['mysql', 'postgresql', 'sqlserver']:
        engine = get_db_connection(db_config)
        with engine.connect() as conn:
            if db_type == 'mysql' or db_type == 'sqlserver':
                result = conn.execute(text("SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = :db"), {"db": db_config['database_name']})
            elif db_type == 'postgresql':
                result = conn.execute(text("SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public'"))
            schema_info = result.fetchall()
        table_structure = {}
        for table, column, dtype in schema_info:
            if table not in table_structure:
                table_structure[table] = []
            table_structure[table].append(f"{column} ({dtype})")
        return [f"Table: {table}, Columns: {', '.join(columns)}" for table, columns in table_structure.items()]
    elif db_type == 'sqlite':
        engine = get_db_connection(db_config)
        with engine.connect() as conn:
            tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
            schema_texts = []
            for (table,) in tables:
                info = conn.execute(text(f"PRAGMA table_info('{table}')")).fetchall()
                columns = [f"{col[1]} ({col[2]})" for col in info]
                schema_texts.append(f"Table: {table}, Columns: {', '.join(columns)}")
        return schema_texts
    elif db_type == 'mongodb':
        schema_json = json.loads(db_config.get('schema_json', '{}'))
        return [f"Collection: {coll}, Fields: {', '.join(fields)}" for coll, fields in schema_json.items()]
    return []

def execute_query(db_config, query):
    db_type = db_config['db_type']
    if db_type in ['mysql', 'postgresql', 'sqlite', 'sqlserver']:
        engine = get_db_connection(db_config)
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
            if result:
                column_names = result[0]._fields if db_type != 'sqlite' else [desc[0] for desc in conn.execute(text(query)).cursor.description]
                formatted = [" | ".join(column_names), "-" * len(" | ".join(column_names))]
                formatted.extend(" | ".join(str(cell) for cell in row) for row in result[:20])
                return "\n".join(formatted)
            return "No results found."
    elif db_type == 'mongodb':
        db = get_db_connection(db_config)
        query_json = json.loads(query)
        collection = db[query_json['collection']]
        if query_json['operation'] == 'find':
            results = list(collection.find(query_json.get('query', {}), query_json.get('projection', {})))
            return "\n".join([str(doc) for doc in results[:20]]) if results else "No results found."
    return "Invalid query or database type."