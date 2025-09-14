# app/auth.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
import os
from dotenv import load_dotenv

from .database import get_db
from .models import User, UserLimit, PLAN_LIMITS

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-please-change-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        return False
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(request: Request, db: Session = Depends(get_db)):
    # Tentar pegar token do cookie primeiro
    token = request.cookies.get("access_token")
    
    # Se não houver cookie, tentar header Authorization
    if not token:
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Não autenticado",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido"
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido"
        )
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuário não encontrado"
        )
    
    # Verificar se trial expirou
    if user.plan == "TRIAL" and user.trial_ends_at:
        if datetime.utcnow() > user.trial_ends_at:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Período de teste expirado. Por favor, assine um plano."
            )
    
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Usuário inativo")
    return current_user

def create_user(db: Session, email: str, password: str, full_name: str, phone: str = None):
    # Verificar se usuário já existe
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email já cadastrado"
        )
    
    # Criar usuário
    user = User(
        email=email,
        password_hash=get_password_hash(password),
        full_name=full_name,
        phone=phone,
        plan="TRIAL",
        trial_ends_at=datetime.utcnow() + timedelta(days=10)
    )
    db.add(user)
    db.commit()
    
    # Criar limites do usuário
    plan_limits = PLAN_LIMITS["TRIAL"]
    user_limits = UserLimit(
        user_id=user.id,
        condominiums_limit=plan_limits["condominiums"],
        uploads_limit=plan_limits["uploads_month"],
        queries_limit=plan_limits["queries_day"]
    )
    db.add(user_limits)
    db.commit()
    
    db.refresh(user)
    return user

def check_user_limits(user: User, db: Session, limit_type: str):
    """Verifica se usuário está dentro dos limites do plano"""
    limits = db.query(UserLimit).filter(UserLimit.user_id == user.id).first()
    
    if not limits:
        # Criar limites se não existirem
        plan_limits = PLAN_LIMITS.get(user.plan, PLAN_LIMITS["BASICO"])
        limits = UserLimit(
            user_id=user.id,
            condominiums_limit=plan_limits["condominiums"],
            uploads_limit=plan_limits["uploads_month"],
            queries_limit=plan_limits["queries_day"]
        )
        db.add(limits)
        db.commit()
    
    # Reset diário de queries
    today = datetime.utcnow().date()
    if limits.last_query_reset < today:
        limits.queries_today = 0
        limits.last_query_reset = today
        db.commit()
    
    # Reset mensal de uploads
    if limits.last_upload_reset.month != today.month:
        limits.uploads_this_month = 0
        limits.last_upload_reset = today
        db.commit()
    
    # Verificar limite específico
    if limit_type == "condominium":
        if limits.condominiums_used >= limits.condominiums_limit:
            raise HTTPException(
                status_code=403,
                detail=f"Limite de condomínios atingido ({limits.condominiums_limit}). Faça upgrade do plano."
            )
    
    elif limit_type == "upload":
        if limits.uploads_this_month >= limits.uploads_limit:
            raise HTTPException(
                status_code=403,
                detail=f"Limite mensal de uploads atingido ({limits.uploads_limit}). Faça upgrade do plano."
            )
    
    elif limit_type == "query":
        if limits.queries_today >= limits.queries_limit:
            raise HTTPException(
                status_code=403,
                detail=f"Limite diário de consultas atingido ({limits.queries_limit}). Tente novamente amanhã ou faça upgrade."
            )
    
    return limits