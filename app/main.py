# app/main.py
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func
from passlib.context import CryptContext
from jose import JWTError, jwt
from dotenv import load_dotenv

# Imports locais
from .database import get_db, engine
from .models import Base, User, Condominio, Document, Query, KnowledgeBase
from .auth import (
    authenticate_user, 
    create_access_token, 
    get_current_user,
    get_current_active_user,
    pwd_context
)
from .upload import process_upload, get_document_list
from .rag_system import get_or_create_rag
from .alexandra import alexandra_chat

# Criar tabelas
Base.metadata.create_all(bind=engine)

# Carregar variáveis de ambiente
load_dotenv()

# Configuração
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Inicializar FastAPI
app = FastAPI(title="AlexandraLex API")

# Servir arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------------ Páginas HTML ------------------------

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Página inicial - redireciona para login"""
    with open("static/login.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Dashboard principal"""
    try:
        with open("static/dashboard.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard em construção</h1>", status_code=200)

@app.get("/register", response_class=HTMLResponse)
async def register_page():
    """Página de registro"""
    with open("static/register.html", "r", encoding="utf-8") as f:
        return f.read()

# ------------------------ Autenticação ------------------------

@app.post("/api/register")
async def register(request: Request, db: Session = Depends(get_db)):
    """Registrar novo usuário"""
    data = await request.json()
    
    # Verificar se usuário já existe
    existing_user = db.query(User).filter(User.email == data["email"]).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email já cadastrado")
    
    # Criar novo usuário
    hashed_password = pwd_context.hash(data["password"])
    new_user = User(
        email=data["email"],
        name=data.get("name", data["email"].split("@")[0]),
        hashed_password=hashed_password,
        phone=data.get("phone")
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Criar token
    access_token = create_access_token(data={"sub": new_user.email})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": new_user.id,
            "email": new_user.email,
        }
    }

@app.post("/api/login")
async def login(request: Request, db: Session = Depends(get_db)):
    """Login de usuário"""
    data = await request.json()
    
    user = authenticate_user(db, data["email"], data["password"])
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Email ou senha incorretos"
        )
    
    # Criar token
    access_token = create_access_token(data={"sub": user.email})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
        }
    }

@app.get("/api/me")
async def get_me(current_user: User = Depends(get_current_active_user)):
    """Retorna informações do usuário atual"""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "is_active": current_user.is_active
    }

# ------------------------ Condomínios ------------------------

@app.get("/api/condominiums")
async def get_condominiums(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lista os condomínios do usuário"""
    condominiums = db.query(Condominio).filter(
        Condominio.sindico_id == current_user.id,
        Condominio.is_active == True
    ).all()
    
    return [
        {
            "id": c.id,
            "name": c.name,
            "address": c.address,
            "units": c.units,
            "created_at": c.created_at.isoformat() if c.created_at else None
        }
        for c in condominiums
    ]

@app.post("/api/condominiums")
async def create_condominium(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Cria um novo condomínio"""
    data = await request.json()
    
    # Verificar limite de condomínios (exemplo: máximo 5)
    count = db.query(func.count(Condominio.id)).filter(
        Condominio.sindico_id == current_user.id,
        Condominio.is_active == True
    ).scalar()
    
    if count >= 5:
        raise HTTPException(
            status_code=400,
            detail="Limite de condomínios atingido (máximo: 5)"
        )
    
    # Criar condomínio
    condominium = Condominio(
        name=data["name"],
        address=data.get("address", ""),
        units=data.get("units", 0),
        sindico_id=current_user.id
    )
    
    db.add(condominium)
    db.commit()
    db.refresh(condominium)
    
    return {
        "id": condominium.id,
        "name": condominium.name,
        "message": "Condomínio criado com sucesso!"
    }

# ------------------------ Documentos ------------------------

@app.get("/api/documents")
async def get_documents(
    condominium_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lista documentos de um condomínio"""
    documents = get_document_list(condominium_id, current_user.id, db)
    return documents

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    condominium_id: int = Form(...),
    title: Optional[str] = Form(None),
    category: Optional[str] = Form("geral"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload de documento para um condomínio"""
    result = await process_upload(
        file=file,
        condo_id=condominium_id,
        sindico_id=current_user.id,
        db=db
    )
    return result

# ------------------------ Query (Busca Inteligente) ------------------------

@app.post("/api/query")
async def query_documents(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Busca inteligente nos documentos"""
    # Parse do body
    raw_body = await request.body()
    print(f"RAW BODY: {raw_body}")
    
    try:
        data = await request.json()
    except:
        data = json.loads(raw_body)
    
    print(f"JSON DATA: {data}")
    
    query = data.get("question", "")
    condominium_id = data.get("condominium_id")
    
    print(f"DEBUG: Query recebida: {query}")
    print(f"DEBUG: Condomínio ID: {condominium_id}")
    print(f"DEBUG: User ID: {current_user.id}")
    
    if not query or not condominium_id:
        return {
            "answer": "Por favor, forneça uma pergunta e selecione um condomínio.",
            "sources": []
        }
    
    # Verificar permissão
    condominium = db.query(Condominio).filter(
        Condominio.id == condominium_id,
        Condominio.sindico_id == current_user.id
    ).first()
    
    if not condominium:
        raise HTTPException(status_code=403, detail="Sem permissão para este condomínio")
    
    try:
        # Chamar o sistema RAG
        rag = get_or_create_rag()
        result = rag.query(
            query=query,
            sindico_id=current_user.id,
            condo_id=condominium_id
        )
        
        print(f"DEBUG: Resultado do RAG: {result}")
        
        # Salvar query no histórico
        query_record = Query(
            user_id=current_user.id,
            condominio_id=condominium_id,
            question=query,
            answer=result.get("answer", ""),
            query_type="PRIVATE_DOC",
            tokens_used=0
        )
        db.add(query_record)
        db.commit()

        # Se não encontrou resultados
        if not result.get("success", False):
            print(f"DEBUG: RAG falhou - {result.get('error', 'Sem erro específico')}")
            return {
                "answer": f"Recebi sua pergunta sobre: '{query}'. O sistema está processando os documentos. Em breve teremos respostas mais precisas.",
                "sources": []
            }
        
        # Sucesso
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0)
        }
        
    except Exception as e:
        print(f"ERRO na query: {str(e)}")
        return {
            "answer": "Desculpe, ocorreu um erro ao processar sua pergunta. Por favor, tente novamente.",
            "sources": [],
            "error": str(e)
        }

# ------------------------ Dra. Alexandra ------------------------

@app.post("/api/alexandra")
async def chat_alexandra(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Chat com a Dra. Alexandra"""
    data = await request.json()
    
    result = await alexandra_chat(
        question=data.get("message", ""),
        context={},
        user_id=current_user.id,
        db=db
    )
    
    return {"answer": result.get("answer", "Desculpe, não consegui processar.")}

# ------------------------ Health ------------------------

@app.get("/api/health")
async def health_check():
    """Verificação de saúde da API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Health check alternativo"""
    return {"status": "ok"}

# ------------------------ Teste ------------------------

@app.get("/api/test")
async def test():
    """Endpoint de teste"""
    return {
        "message": "API funcionando!",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)