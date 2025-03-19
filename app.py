from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from pydantic import BaseModel, ValidationError
import os
import uuid
import logging
from typing import Optional, Dict, Any
import tempfile
import traceback
import soundfile as sf
from models import User, DBConfig, engine
from auth import get_current_user, verify_password, get_password_hash, create_access_token
from db_handler import encrypt_password, extract_schema
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
    "https://voi-db.vercel.app/",  # Vercel frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows only specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Database session handling with error catching
SessionLocal = sessionmaker(bind=engine)

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

# Custom exception handler for database errors
@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """Handle SQLAlchemy database errors"""
    logger.error(f"Database error: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "A database error occurred. Please try again later."}
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

# Database dependency with error handling
def get_db():
    db = SessionLocal()
    try:
        yield db
    except OperationalError as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection error. Please try again later."
        )
    finally:
        db.close()

@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
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
            
        # Check if username already exists
        db_user = db.query(User).filter(User.username == user.username).first()
        if db_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Username already exists"
            )
            
        # Create new user
        hashed_password = get_password_hash(user.password)
        new_user = User(username=user.username, password_hash=hashed_password)
        db.add(new_user)
        db.commit()
        
        logger.info(f"New user created: {user.username}")
        return {"message": "User created successfully"}
        
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Database integrity error during signup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists or database constraint violated"
        )
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error during signup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred. Please try again later."
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error during signup: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later."
        )

@app.post("/login")
def login(user: UserCreate, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.username == user.username).first()
        if not db_user or not verify_password(user.password, db_user.password_hash):
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

@app.post("/db-config")
def save_db_config(config: DBConfigCreate, current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.username == current_user).first()
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="User not found"
            )
            
        # Check if configuration already exists and update it
        existing_config = db.query(DBConfig).filter(DBConfig.user_id == db_user.id).first()
        
        encrypted_password = encrypt_password(config.password)
        db_config = {
            "db_type": config.db_type,
            "host": config.host,
            "port": config.port,
            "username": config.username,
            "encrypted_password": encrypted_password,
            "database_name": config.database_name,
            "db_schema_json": config.db_schema_json
        }
        
        if existing_config:
            # Update existing configuration
            for key, value in db_config.items():
                setattr(existing_config, key, value)
            message = "Database configuration updated"
        else:
            # Create new configuration
            new_config = DBConfig(user_id=db_user.id, **db_config)
            db.add(new_config)
            message = "Database configuration saved"
            
        db.commit()
        
        logger.info(f"DB config saved/updated for user: {current_user}")
        return {"message": message}
        
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error while saving DB config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred while saving configuration"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving DB config: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while saving database configuration"
        )

@app.post("/voice-query")
async def voice_query(audio: UploadFile = File(...), current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    temp_file_path = None
    try:
        # Validate user and configuration
        db_user = db.query(User).filter(User.username == current_user).first()
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
            
        db_config = db.query(DBConfig).filter(DBConfig.user_id == db_user.id).first()
        if not db_config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No database configuration found. Please set up your database first."
            )
        
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
            assistant = VoiceAssistant(db_config.__dict__, app.state.whisper_model, app.state.llm_model)
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

@app.post("/text-query")
def text_query(request: TextQueryRequest, current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        # Validate query
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
            
        # Validate user and configuration
        db_user = db.query(User).filter(User.username == current_user).first()
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
            
        db_config = db.query(DBConfig).filter(DBConfig.user_id == db_user.id).first()
        if not db_config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No database configuration found. Please set up your database first."
            )
        
        # Process the query
        try:
            assistant = VoiceAssistant(db_config.__dict__, app.state.whisper_model, app.state.llm_model)
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
            
        return {"status": "healthy", "version": "1.0.0"}
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "detail": str(e)}
        )