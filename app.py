from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session, sessionmaker
from pydantic import BaseModel
import os
import uuid
import whisper
from typing import Optional
from langchain_community.llms import Ollama
import tempfile
import soundfile as sf
from models import User, DBConfig, engine
from auth import get_current_user, verify_password, get_password_hash, create_access_token
from db_handler import encrypt_password, extract_schema
from assistant import VoiceAssistant
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = FastAPI()
SessionLocal = sessionmaker(bind=engine)

# Load models on startup
@app.on_event("startup")
async def startup_event():
    app.state.whisper_model = whisper.load_model("tiny")  # Load Whisper model once
    app.state.llm_model = Ollama(model="deepseek-r1:1.5b")  # Load LLM once

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
    schema_json: Optional[str] = None  # Optional for MongoDB
    
class TextQueryRequest(BaseModel):
    query: str
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, password_hash=hashed_password)
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully"}

@app.post("/login")
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/db-config")
def save_db_config(config: DBConfigCreate, current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == current_user).first()
    encrypted_password = encrypt_password(config.password)
    db_config = {
        "db_type": config.db_type,
        "host": config.host,
        "port": config.port,
        "username": config.username,
        "encrypted_password": encrypted_password,
        "database_name": config.database_name,
        "schema_json": config.schema_json
    }
    new_config = DBConfig(user_id=db_user.id, **db_config)
    db.add(new_config)
    db.commit()
    db.refresh(new_config)
    return {"message": "Database configuration saved"}

@app.post("/voice-query")
async def voice_query(audio: UploadFile = File(...), current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == current_user).first()
    db_config = db.query(DBConfig).filter(DBConfig.user_id == db_user.id).first()
    if not db_config:
        raise HTTPException(status_code=400, detail="No database configuration found")
    
    # Create a unique temp file path
    temp_file_path = os.path.join(tempfile.gettempdir(), f"voice_query_{uuid.uuid4().hex}.wav")
    
    try:
        # Save the uploaded file directly
        content = await audio.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        # Process the audio file
        assistant = VoiceAssistant(db_config.__dict__, app.state.whisper_model, app.state.llm_model)
        transcription = assistant.transcribe_audio(temp_file_path)
        response = assistant.get_response(transcription)
        
        return {"transcription": transcription, "response": response}
    
    finally:
        # Try to remove the file in the finally block
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception as e:
            # Just log the error but don't fail the request
            print(f"Error removing temporary file: {e}")

@app.post("/text-query")
def text_query(request: TextQueryRequest, current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == current_user).first()
    db_config = db.query(DBConfig).filter(DBConfig.user_id == db_user.id).first()
    if not db_config:
        raise HTTPException(status_code=400, detail="No database configuration found")
    assistant = VoiceAssistant(db_config.__dict__, app.state.whisper_model, app.state.llm_model)
    response = assistant.get_response(request.query)
    return {"response": response}