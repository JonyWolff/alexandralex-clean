# app/models.py
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Text, Date
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from .database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    phone = Column(String)
    plan = Column(String, default="TRIAL")  # TRIAL, BASICO, PROFISSIONAL, PREMIUM
    trial_ends_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    
    # Relacionamentos
    condominios = relationship("Condominio", back_populates="sindico")
    queries = relationship("Query", back_populates="user")
    limits = relationship("UserLimit", back_populates="user", uselist=False)
    documents = relationship("Document", back_populates="uploader")

class Condominio(Base):
    __tablename__ = "condominios"
    
    id = Column(Integer, primary_key=True, index=True)
    sindico_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, nullable=False)
    address = Column(Text)
    units = Column(Integer)
    cnpj = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relacionamentos
    sindico = relationship("User", back_populates="condominios")
    documents = relationship("Document", back_populates="condominio")
    queries = relationship("Query", back_populates="condominio")

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String)  # pdf, txt
    file_size = Column(Integer)
    description = Column(String)
    processed = Column(Boolean, default=False)
    chunks_count = Column(Integer, default=0)
    processing_error = Column(String, nullable=True)
    
    condo_id = Column(Integer, ForeignKey("condominios.id"))
    uploaded_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relacionamentos
    condominio = relationship("Condominio", back_populates="documents")
    uploader = relationship("User", back_populates="documents")

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    category = Column(String)  # LEI_FEDERAL, LEI_ESTADUAL, NORMA_ABNT, JURISPRUDENCIA
    title = Column(String)
    description = Column(Text)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    is_public = Column(Boolean, default=True)
    chunks_count = Column(Integer)
    embeddings_count = Column(Integer)

class Query(Base):
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    condominio_id = Column(Integer, ForeignKey("condominios.id"), nullable=True)
    question = Column(Text)
    answer = Column(Text)
    query_type = Column(String)  # PRIVATE_DOC, PUBLIC_KNOWLEDGE, ALEXANDRA
    tokens_used = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relacionamentos
    user = relationship("User", back_populates="queries")
    condominio = relationship("Condominio", back_populates="queries")

class UserLimit(Base):
    __tablename__ = "user_limits"
    
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    
    # Limites de condomínios
    condominiums_used = Column(Integer, default=0)
    condominiums_limit = Column(Integer, default=1)
    
    # Limites de uploads
    uploads_this_month = Column(Integer, default=0)
    uploads_limit = Column(Integer, default=10)
    
    # Limites de consultas
    queries_today = Column(Integer, default=0)
    queries_limit = Column(Integer, default=20)
    
    # Datas de reset
    last_upload_reset = Column(Date, default=datetime.utcnow)
    last_query_reset = Column(Date, default=datetime.utcnow)
    
    # Relacionamento
    user = relationship("User", back_populates="limits")

# Planos e seus limites
PLAN_LIMITS = {
    "TRIAL": {
        "condominiums": 1,
        "uploads_month": 5,
        "queries_day": 10,
        "duration_days": 10,
        "price": 0
    },
    "BASICO": {
        "condominiums": 1,
        "uploads_month": 10,
        "queries_day": 20,
        "price": 99
    },
    "PROFISSIONAL": {
        "condominiums": 5,
        "uploads_month": 50,
        "queries_day": 100,
        "price": 199
    },
    "PREMIUM": {
        "condominiums": 15,
        "uploads_month": 200,
        "queries_day": 500,
        "price": 299
    }
}