from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError, OperationalError
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ConnectionFailure, ServerSelectionTimeoutError
import mysql.connector
import psycopg2
from mysql.connector import Error as MySQLError
import json
from cryptography.fernet import Fernet, InvalidToken
import os
import logging
from typing import Dict, List, Any, Union, Optional
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_utils")

# Fernet key handling
def get_encryption_key() -> bytes:
    """Get the Fernet encryption key from environment variables."""
    try:
        key = os.getenv("FERNET_KEY")
        if not key:
            logger.critical("FERNET_KEY environment variable not set")
            raise ValueError("FERNET_KEY environment variable not set")
            
        # Convert the string back to bytes if stored as string
        if isinstance(key, str):
            key = key.encode()
            
        # Validate the key is a valid Fernet key
        if len(key) != 32 and not key.startswith(b'dGVzdA=='):
            logger.warning("The provided key appears to be an invalid Fernet key")
            
        return key
    except Exception as e:
        logger.critical(f"Failed to retrieve encryption key: {str(e)}")
        raise

try:
    cipher = Fernet(get_encryption_key())
except Exception as e:
    logger.critical(f"Failed to initialize encryption: {str(e)}")
    raise

def encrypt_password(password: str) -> str:
    """Encrypt a password using Fernet symmetric encryption."""
    try:
        if not password:
            logger.warning("Attempted to encrypt empty password")
            return ""
        return cipher.encrypt(password.encode()).decode()
    except Exception as e:
        logger.error(f"Password encryption failed: {str(e)}")
        raise ValueError("Failed to encrypt password") from e

def decrypt_password(encrypted_password: str) -> str:
    """Decrypt an encrypted password."""
    try:
        if not encrypted_password:
            logger.warning("Attempted to decrypt empty password")
            return ""
        return cipher.decrypt(encrypted_password.encode()).decode()
    except InvalidToken:
        logger.error("Invalid token - password might have been encrypted with different key")
        raise ValueError("Invalid encrypted password - may have been encrypted with a different key")
    except Exception as e:
        logger.error(f"Password decryption failed: {str(e)}")
        raise ValueError("Failed to decrypt password") from e
    
def direct_db_connect():
    """
    Connect directly to the application database using a NeonTech PostgreSQL URI.
    """
    try:
        # Get the database URI from environment
        db_uri = os.environ.get("DB_URI")
        
        if not db_uri:
            logger.critical("DATABASE_URL environment variable not set")
            raise ValueError("DATABASE_URL environment variable not set")
            
        logger.info("Connecting to PostgreSQL database using NeonTech URI")
        
        # Connect to PostgreSQL using the URI
        connection = psycopg2.connect(db_uri)
        
        # Set autocommit to False to enable transaction control
        connection.autocommit = False
        
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}\n{traceback.format_exc()}")
        raise

def validate_db_config(db_config: Dict[str, Any]) -> None:
    """Validate that the database configuration has all required fields."""
    if not isinstance(db_config, dict):
        raise TypeError("Database configuration must be a dictionary")
        
    required_fields = ['db_type']
    if not all(field in db_config for field in required_fields):
        missing = [field for field in required_fields if field not in db_config]
        raise ValueError(f"Missing required database configuration fields: {', '.join(missing)}")
    
    # Check specific required fields based on db_type
    db_type = db_config['db_type']
    
    # For all except SQLite
    if db_type != 'sqlite':
        required_db_fields = ['username', 'encrypted_password', 'host', 'port']
        
        if not all(field in db_config for field in required_db_fields):
            missing = [field for field in required_db_fields if field not in db_config]
            raise ValueError(f"Missing required fields for {db_type}: {', '.join(missing)}")
    
    # Database name is required for all types
    if 'database_name' not in db_config:
        raise ValueError("Missing required field: database_name")

def get_db_connection(db_config: Dict[str, Any]) -> Any:
    """
    Create a database connection based on the provided configuration.
    
    Args:
        db_config: A dictionary containing database connection details
        
    Returns:
        A database connection object appropriate for the database type
        
    Raises:
        ValueError: If configuration is invalid or database type is unsupported
        ConnectionError: If connection to the database fails
    """
    try:
        # Validate the configuration
        validate_db_config(db_config)
        
        db_type = db_config['db_type']
        
        # Only decrypt password if we need it (not for SQLite)
        password = ""
        if db_type != 'sqlite' and 'encrypted_password' in db_config:
            try:
                password = decrypt_password(db_config['encrypted_password'])
            except ValueError as e:
                logger.error(f"Failed to decrypt database password: {str(e)}")
                raise ConnectionError(f"Authentication error: {str(e)}")
        
        # Create appropriate connection based on database type
        if db_type in ['mysql', 'postgresql', 'sqlite', 'sqlserver']:
            try:
                if db_type == 'mysql':
                    url = f"mysql+mysqlconnector://{db_config['username']}:{password}@{db_config['host']}:{db_config['port']}/{db_config['database_name']}"
                elif db_type == 'postgresql':
                    url = f"postgresql+psycopg2://{db_config['username']}:{password}@{db_config['host']}:{db_config['port']}/{db_config['database_name']}"
                elif db_type == 'sqlite':
                    url = f"sqlite:///{db_config['database_name']}"
                elif db_type == 'sqlserver':
                    url = f"mssql+pyodbc://{db_config['username']}:{password}@{db_config['host']}:{db_config['port']}/{db_config['database_name']}?driver=ODBC+Driver+17+for+SQL+Server"
                
                logger.info(f"Connecting to {db_type} database: {db_config['database_name']}")
                engine = create_engine(url, echo=False, pool_pre_ping=True)
                
                # Test the connection
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    
                return engine
            except OperationalError as e:
                logger.error(f"Database connection operational error: {str(e)}")
                raise ConnectionError(f"Failed to connect to {db_type} database: {str(e)}")
            except SQLAlchemyError as e:
                logger.error(f"SQLAlchemy error: {str(e)}")
                raise ConnectionError(f"Database error: {str(e)}")
                
        elif db_type == 'mongodb':
            try:
                connection_string = f"mongodb://{db_config['username']}:{password}@{db_config['host']}:{db_config['port']}"
                logger.info(f"Connecting to MongoDB: {db_config['database_name']}")
                client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
                
                # Test the connection
                client.admin.command('ping')
                
                return client[db_config['database_name']]
            except ServerSelectionTimeoutError as e:
                logger.error(f"MongoDB server selection timeout: {str(e)}")
                raise ConnectionError(f"Failed to connect to MongoDB server: {str(e)}")
            except ConnectionFailure as e:
                logger.error(f"MongoDB connection failure: {str(e)}")
                raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")
            except PyMongoError as e:
                logger.error(f"MongoDB error: {str(e)}")
                raise ConnectionError(f"MongoDB error: {str(e)}")
        else:
            logger.error(f"Unsupported database type: {db_type}")
            raise ValueError(f"Unsupported database type: {db_type}")
    except ValueError as e:
        # Re-raise validation errors
        logger.error(f"Invalid database configuration: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_db_connection: {str(e)}\n{traceback.format_exc()}")
        raise ConnectionError(f"Failed to establish database connection: {str(e)}")

def extract_schema(db_config: Dict[str, Any]) -> List[str]:
    """
    Extract database schema information.
    
    Args:
        db_config: Database configuration dictionary
        
    Returns:
        A list of strings containing schema information
        
    Raises:
        ValueError: If schema extraction fails
    """
    try:
        validate_db_config(db_config)
        db_type = db_config['db_type']
        
        logger.info(f"Extracting schema for {db_type} database: {db_config['database_name']}")
        
        if db_type in ['mysql', 'postgresql', 'sqlserver']:
            try:
                engine = get_db_connection(db_config)
                with engine.connect() as conn:
                    if db_type == 'mysql' or db_type == 'sqlserver':
                        result = conn.execute(text("SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = :db"), {"db": db_config['database_name']})
                    elif db_type == 'postgresql':
                        result = conn.execute(text("SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public'"))
                    
                    schema_info = result.fetchall()
                    
                if not schema_info:
                    logger.warning(f"No schema information found for {db_config['database_name']}")
                    return ["No tables found in database."]
                    
                table_structure = {}
                for table, column, dtype in schema_info:
                    if table not in table_structure:
                        table_structure[table] = []
                    table_structure[table].append(f"{column} ({dtype})")
                    
                return [f"Table: {table}, Columns: {', '.join(columns)}" for table, columns in table_structure.items()]
                
            except ProgrammingError as e:
                logger.error(f"SQL programming error during schema extraction: {str(e)}")
                raise ValueError(f"Error accessing schema information: {str(e)}")
            except SQLAlchemyError as e:
                logger.error(f"SQLAlchemy error during schema extraction: {str(e)}")
                raise ValueError(f"Database error during schema extraction: {str(e)}")
                
        elif db_type == 'sqlite':
            try:
                engine = get_db_connection(db_config)
                with engine.connect() as conn:
                    tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
                    
                    if not tables:
                        logger.warning("No tables found in SQLite database")
                        return ["No tables found in database."]
                        
                    schema_texts = []
                    for (table,) in tables:
                        info = conn.execute(text(f"PRAGMA table_info('{table}')")).fetchall()
                        columns = [f"{col[1]} ({col[2]})" for col in info]
                        schema_texts.append(f"Table: {table}, Columns: {', '.join(columns)}")
                        
                return schema_texts
                
            except SQLAlchemyError as e:
                logger.error(f"SQLite error during schema extraction: {str(e)}")
                raise ValueError(f"SQLite error: {str(e)}")
                
        elif db_type == 'mongodb':
            try:
                if 'schema_json' not in db_config or not db_config['schema_json']:
                    logger.warning("No schema_json provided for MongoDB")
                    # For MongoDB, we could list collections if no schema is provided
                    db = get_db_connection(db_config)
                    collections = db.list_collection_names()
                    if not collections:
                        return ["No collections found in database."]
                    return [f"Collection: {coll}" for coll in collections]
                
                schema_json = json.loads(db_config['schema_json'])
                if not schema_json:
                    logger.warning("Empty schema_json for MongoDB")
                    return ["No collections defined in schema."]
                
                return [f"Collection: {coll}, Fields: {', '.join(fields)}" for coll, fields in schema_json.items()]
                
            except json.JSONDecodeError:
                logger.error("Invalid JSON in schema_json")
                raise ValueError("Invalid JSON format in schema_json")
            except PyMongoError as e:
                logger.error(f"MongoDB error during schema extraction: {str(e)}")
                raise ValueError(f"MongoDB error: {str(e)}")
                
        logger.error(f"Unsupported database type for schema extraction: {db_type}")
        return []
        
    except ConnectionError as e:
        logger.error(f"Connection error during schema extraction: {str(e)}")
        raise ValueError(f"Failed to connect to database: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in extract_schema: {str(e)}\n{traceback.format_exc()}")
        raise ValueError(f"Failed to extract schema: {str(e)}")

def execute_query(db_config: Dict[str, Any], query: str) -> str:
    """
    Execute a query against the specified database.
    
    Args:
        db_config: Database configuration dictionary
        query: SQL query string or MongoDB query JSON
        
    Returns:
        String containing formatted query results
        
    Raises:
        ValueError: If query execution fails
    """
    try:
        validate_db_config(db_config)
        db_type = db_config['db_type']
        
        if not query:
            logger.warning("Empty query provided")
            raise ValueError("Query cannot be empty")
            
        logger.info(f"Executing query on {db_type} database: {db_config['database_name']}")
        
        if db_type in ['mysql', 'postgresql', 'sqlite', 'sqlserver']:
            try:
                engine = get_db_connection(db_config)
                with engine.connect() as conn:
                    # Check if query is a SELECT query to prevent data modification
                    if not query.strip().lower().startswith(('select', 'show', 'describe', 'explain')):
                        logger.warning(f"Non-SELECT query attempted: {query[:50]}...")
                        raise ValueError("Only SELECT, SHOW, DESCRIBE, and EXPLAIN queries are allowed")
                    
                    result_proxy = conn.execute(text(query))
                    result = result_proxy.fetchall()
                    
                    if not result:
                        logger.info("Query returned no results")
                        return "No results found."
                    
                    # Get column names
                    if db_type != 'sqlite':
                        column_names = result[0]._fields
                    else:
                        column_names = [desc[0] for desc in result_proxy.cursor.description]
                    
                    # Format results
                    header = " | ".join(column_names)
                    separator = "-" * len(header)
                    
                    formatted = [header, separator]
                    for row in result[:20]:  # Limit to 20 rows
                        # Convert each cell to string, handle None values
                        row_str = " | ".join(str(cell) if cell is not None else "NULL" for cell in row)
                        formatted.append(row_str)
                    
                    # Add a note if results were truncated
                    if len(result) > 20:
                        formatted.append(f"\n... {len(result) - 20} more rows (showing first 20 of {len(result)} total)")
                    
                    return "\n".join(formatted)
                
            except ProgrammingError as e:
                logger.error(f"SQL programming error: {str(e)}")
                raise ValueError(f"Query error: {str(e)}")
            except OperationalError as e:
                logger.error(f"SQL operational error: {str(e)}")
                raise ValueError(f"Database operational error: {str(e)}")
            except SQLAlchemyError as e:
                logger.error(f"SQLAlchemy error: {str(e)}")
                raise ValueError(f"Database error: {str(e)}")
                
        elif db_type == 'mongodb':
            try:
                # Parse the MongoDB query from JSON
                try:
                    query_json = json.loads(query)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON in MongoDB query")
                    raise ValueError("Invalid JSON format in MongoDB query")
                
                # Validate query structure
                if 'collection' not in query_json:
                    logger.error("Missing 'collection' in MongoDB query")
                    raise ValueError("MongoDB query must specify a 'collection'")
                    
                if 'operation' not in query_json:
                    logger.error("Missing 'operation' in MongoDB query")
                    raise ValueError("MongoDB query must specify an 'operation'")
                
                # Get database connection
                db = get_db_connection(db_config)
                collection = db[query_json['collection']]
                
                # Execute appropriate operation
                if query_json['operation'] == 'find':
                    filter_query = query_json.get('query', {})
                    projection = query_json.get('projection', {})
                    limit = query_json.get('limit', 20)  # Default limit to 20
                    
                    logger.info(f"Executing MongoDB find on {query_json['collection']}")
                    results = list(collection.find(filter_query, projection).limit(limit))
                    
                    if not results:
                        return "No results found."
                    
                    # Format results
                    formatted_results = [str(doc) for doc in results]
                    
                    # Add count information
                    result_count = len(results)
                    total_count = collection.count_documents(filter_query)
                    
                    if total_count > result_count:
                        formatted_results.append(f"\n... showing {result_count} of {total_count} total documents")
                    
                    return "\n".join(formatted_results)
                else:
                    logger.error(f"Unsupported MongoDB operation: {query_json['operation']}")
                    raise ValueError(f"Unsupported MongoDB operation: {query_json['operation']}")
                
            except PyMongoError as e:
                logger.error(f"MongoDB error: {str(e)}")
                raise ValueError(f"MongoDB error: {str(e)}")
                
        logger.error(f"Unsupported database type for query execution: {db_type}")
        raise ValueError(f"Unsupported database type: {db_type}")
        
    except ConnectionError as e:
        logger.error(f"Connection error during query execution: {str(e)}")
        raise ValueError(f"Failed to connect to database: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in execute_query: {str(e)}\n{traceback.format_exc()}")
        raise ValueError(f"Failed to execute query: {str(e)}")