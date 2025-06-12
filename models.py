from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import os
from dotenv import load_dotenv
load_dotenv()
DATABASE_URL = os.environ.get("DB_URI")

# Create engine with proper configuration for Neon
engine = create_engine(
    DATABASE_URL,
    # These settings help with Neon's serverless nature
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
)
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    db_configs = relationship("DBConfig", back_populates="user")

class DBConfig(Base):
    __tablename__ = 'db_configs'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    db_type = Column(String, nullable=False)  # e.g., 'mysql', 'postgresql', 'mongodb'
    host = Column(String, nullable=False)
    port = Column(Integer, nullable=False)
    username = Column(String, nullable=False)
    encrypted_password = Column(String, nullable=False)
    database_name = Column(String, nullable=False)
    db_schema_json = Column(String, nullable=True)  # For MongoDB, optional JSON schema
    faiss_index_path = Column(String , nullable=True) 
    user = relationship("User", back_populates="db_configs")