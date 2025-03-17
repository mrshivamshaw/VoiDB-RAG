from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

engine = create_engine('sqlite:///app.db')
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
    db_string = Column(String, nullable=False)
    port = Column(Integer, nullable=False)
    username = Column(String, nullable=False)
    encrypted_password = Column(String, nullable=False)
    database_name = Column(String, nullable=False)
    schema_json = Column(String)  # For MongoDB, optional JSON schema
    faiss_index_path = Column(String)  # Path to FAISS index file
    user = relationship("User", back_populates="db_configs")

Base.metadata.create_all(engine)