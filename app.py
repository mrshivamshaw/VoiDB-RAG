from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request, status
from fastapi.responses import JSONResponse

from pydantic import BaseModel, ValidationError
import os
import uuid
import logging
from typing import Optional
import tempfile
import traceback
import soundfile as sf
from auth import get_current_user, verify_password, get_password_hash, create_access_token
from db_handler import encrypt_password, direct_db_connect
from assistant import VoiceAssistant
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv()
app = FastAPI()

# List of allowed origins (frontend URLs)
origins = [
    "http://localhost:3000",  # React frontend (or any other frontend running locally)
    "https://voi-db.vercel.app",  # Vercel frontend
    "https://aiquerynet.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows only specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Custom exception handler for the entire application
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

# Custom exception handler for validation errors
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors"""
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid input data", "errors": exc.errors()}
    )


# Load models on startup with robust error handling
@app.on_event("startup")
async def startup_event():
    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            logger.critical("GROQ_API_KEY environment variable is not set")
            raise ValueError("GROQ_API_KEY environment variable must be set")
        
        # Initialize Groq client
        try:
            app.state.whisper_model = Groq(api_key=groq_api_key)
            logger.info("Groq Whisper model initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize Groq Whisper model: {str(e)}")
            raise
        
        # Initialize the Groq LLM model
        try:
            app.state.llm_model = ChatGroq(
                api_key=groq_api_key,
                model_name="llama3-70b-8192"  # Use a model that's available in Groq
            )
            logger.info("Groq LLM model initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize Groq LLM model: {str(e)}")
            raise
            
    except Exception as e:
        logger.critical(f"Application startup failed: {str(e)}\n{traceback.format_exc()}")
        # Re-raise to prevent app from starting with missing dependencies
        raise

# Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down")
    # Add any cleanup code here if needed

class UserCreate(BaseModel):
    username: str
    password: str

class DBConfigCreate(BaseModel):
    db_type: str
    host: str
    port: int
    username: str
    password: str
    database_name: str
    db_schema_json: Optional[str] = None  # Renamed from schema_json to db_schema_json

class TextQueryRequest(BaseModel):
    query: str


@app.post("/signup")
def signup(user: UserCreate):
    try:
        # Check if username is valid length
        if len(user.username) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Username must be at least 3 characters long"
            )
            
        # Check if password meets requirements
        if len(user.password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Password must be at least 8 characters long"
            )
            
        # Get database connection
        db_conn = direct_db_connect()
        cursor = db_conn.cursor()
        
        # Check if username already exists
        cursor.execute("SELECT id FROM users WHERE username = %s", (user.username,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Username already exists"
            )
            
        # Create new user
        hashed_password = get_password_hash(user.password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
            (user.username, hashed_password)
        )
        db_conn.commit()
        
        logger.info(f"New user created: {user.username}")
        return {"message": "User created successfully"}
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during signup: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later."
        )
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'db_conn' in locals():
            db_conn.close()

@app.post("/login")
def login(user: UserCreate):
    try:
        # Get database connection
        db_conn = direct_db_connect()
        cursor = db_conn.cursor()
        
        # Get user
        cursor.execute("SELECT id, password_hash FROM users WHERE username = %s", (user.username,))
        db_user = cursor.fetchone()
        
        if not db_user or not verify_password(user.password, db_user[1]):  # db_user[1] is the password_hash
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid username or password"
            )
            
        token = create_access_token({"sub": user.username})
        logger.info(f"User logged in: {user.username}")
        return {"access_token": token, "token_type": "bearer"}
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Login error for user {user.username}: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login. Please try again later."
        )
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'db_conn' in locals():
            db_conn.close()

@app.post("/db-config")
def save_db_config(config: DBConfigCreate, current_user: str = Depends(get_current_user)):
    try:
        # Get database connection
        db_conn = direct_db_connect()
        cursor = db_conn.cursor()
        
        # Get user ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (current_user,))
        user_result = cursor.fetchone()
        if not user_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="User not found"
            )
        user_id = user_result[0]
            
        # Check if configuration already exists
        cursor.execute("SELECT id FROM db_configs WHERE user_id = %s", (user_id,))
        existing_config = cursor.fetchone()
        
        encrypted_password = encrypt_password(config.password)
        
        if existing_config:
            # Update existing configuration
            cursor.execute("""
                UPDATE db_configs 
                SET db_type = %s, host = %s, port = %s, username = %s, 
                    encrypted_password = %s, database_name = %s, db_schema_json = %s
                WHERE user_id = %s
            """, (
                config.db_type, config.host, config.port, config.username,
                encrypted_password, config.database_name, config.db_schema_json,
                user_id
            ))
            message = "Database configuration updated"
        else:
            # Create new configuration
            cursor.execute("""
                INSERT INTO db_configs 
                (user_id, db_type, host, port, username, encrypted_password, database_name, db_schema_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id, config.db_type, config.host, config.port, config.username,
                encrypted_password, config.database_name, config.db_schema_json
            ))
            message = "Database configuration saved"
            
        db_conn.commit()
        
        logger.info(f"DB config saved/updated for user: {current_user}")
        return {"message": message}
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error saving DB config: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while saving database configuration"
        )
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'db_conn' in locals():
            db_conn.close()

@app.post("/voice-query")
async def voice_query(audio: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    temp_file_path = None
    try:
        # Get database connection
        db_conn = direct_db_connect()
        cursor = db_conn.cursor()
        
        # Get user ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (current_user,))
        user_result = cursor.fetchone()
        if not user_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        user_id = user_result[0]
            
        # Get DB config
        cursor.execute("""
            SELECT db_type, host, port, username, encrypted_password, database_name, db_schema_json
            FROM db_configs WHERE user_id = %s
        """, (user_id,))
        db_config_data = cursor.fetchone()
        
        if not db_config_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No database configuration found. Please set up your database first."
            )
            
        db_config = {
            "db_type": db_config_data[0],
            "host": db_config_data[1],
            "port": db_config_data[2],
            "username": db_config_data[3],
            "encrypted_password": db_config_data[4],
            "database_name": db_config_data[5],
            "db_schema_json": db_config_data[6]
        }
        
        # Validate file type and size
        if not audio.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported audio format. Please upload .wav, .mp3, .flac, or .ogg files."
            )
            
        # Create a unique temp file path
        temp_file_path = os.path.join(tempfile.gettempdir(), f"voice_query_{uuid.uuid4().hex}.wav")
        
        # Save the uploaded file
        content = await audio.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio file received"
            )
            
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        # Process the audio file
        try:
            with open(temp_file_path, "rb") as file:
                # Try to transcribe the audio
                transcription = app.state.whisper_model.audio.transcriptions.create(
                    file=(temp_file_path, file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                )
                
            if not hasattr(transcription, 'text') or not transcription.text:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Could not transcribe audio. Please ensure the audio contains clear speech."
                )
                
            transcription_text = transcription.text
            
            # Process the transcription
            assistant = VoiceAssistant(db_config, app.state.whisper_model, app.state.llm_model)
            response = assistant.get_response(transcription_text)
            
            logger.info(f"Successfully processed voice query for user: {current_user}")
            return {"transcription": transcription_text, "response": response}
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error processing audio file"
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in voice query: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your voice query"
        )
    
    finally:
        # Clean up temporary file
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")
            
        if 'cursor' in locals():
            cursor.close()
        if 'db_conn' in locals():
            db_conn.close()

@app.post("/text-query")
def text_query(request: TextQueryRequest, current_user: str = Depends(get_current_user)):
    try:
        # Validate query
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        # Get database connection
        db_conn = direct_db_connect()
        cursor = db_conn.cursor()
        
        # Get user ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (current_user,))
        user_result = cursor.fetchone()
        if not user_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        user_id = user_result[0]
            
        # Get DB config
        cursor.execute("""
            SELECT db_type, host, port, username, encrypted_password, database_name, db_schema_json
            FROM db_configs WHERE user_id = %s
        """, (user_id,))
        db_config_data = cursor.fetchone()
        
        if not db_config_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No database configuration found. Please set up your database first."
            )
            
        db_config = {
            "db_type": db_config_data[0],
            "host": db_config_data[1],
            "port": db_config_data[2],
            "username": db_config_data[3],
            "encrypted_password": db_config_data[4],
            "database_name": db_config_data[5],
            "db_schema_json": db_config_data[6]
        }
        
        # Process the query
        try:
            assistant = VoiceAssistant(db_config, app.state.whisper_model, app.state.llm_model)
            response = assistant.get_response(request.query)
            
            logger.info(f"Successfully processed text query for user: {current_user}")
            return {"response": response}
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error processing your query"
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in text query: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your query"
        )
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'db_conn' in locals():
            db_conn.close()

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint to verify service is running"""
    try:
        # Check if models are loaded
        if not hasattr(app.state, "whisper_model") or not hasattr(app.state, "llm_model"):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "unhealthy", "detail": "AI models not initialized"}
            )
            
        # Test database connection
        try:
            db_conn = direct_db_connect()
            cursor = db_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            db_conn.close()
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "unhealthy", "detail": f"Database connection failed: {str(e)}"}
            )
            
        return {"status": "healthy", "version": "1.0.0"}
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "detail": str(e)}
        )

@app.get("/")
def home():
    return {"status": "OK", "message": "API is running"}