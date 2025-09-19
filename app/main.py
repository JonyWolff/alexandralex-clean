# app/main.py
import os
import json
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func
from dotenv import load_dotenv

# Imports locais
from .database import get_db, engine
from .models import Base, User, Condominio, Document, Query, KnowledgeBase
from .auth import (
    authenticate_user, 
    create_access_token, 
    get_current_active_user,
    pwd_context
)
from .upload import process_upload, get_document_list
from .rag_system import get_or_create_rag
from .alexandra import alexandra_chat
from .knowledge_base import router as knowledge_router

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

# Router da Base de Conhecimento
app.include_router(knowledge_router)

# Servir arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------------ Páginas HTML ------------------------

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/login.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    try:
        with open("static/dashboard.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard em construção</h1>", status_code=200)

@app.get("/register", response_class=HTMLResponse)
async def register_page():
    with open("static/register.html", "r", encoding="utf-8") as f:
        return f.read()

# ------------------------ Autenticação ------------------------

@app.post("/api/register")
async def register(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    
    existing_user = db.query(User).filter(User.email == data["email"]).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email já cadastrado")
    
    hashed_password = pwd_context.hash(data["password"])
    new_user = User(
        email=data["email"],
        full_name=data.get("name", data["email"].split("@")[0]),
        password_hash=hashed_password,
        phone=data.get("phone")
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = create_access_token(data={"sub": new_user.email})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": new_user.id, "email": new_user.email}
    }

@app.post("/api/login")
async def login(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    
    user = authenticate_user(db, data["email"], data["password"])
    if not user:
        raise HTTPException(status_code=401, detail="Email ou senha incorretos")
    
    access_token = create_access_token(data={"sub": user.email})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email}
    }

@app.get("/api/me")
async def get_me(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Retorna informações do usuário atual + estatísticas básicas"""
    condos_count = db.query(func.count(Condominio.id)).filter(
        Condominio.sindico_id == current_user.id,
        Condominio.is_active == True
    ).scalar()

    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "plan": current_user.plan,
        "is_active": current_user.is_active,
        "stats": {
    "condominiums_count": condos_count or 0,
    "condominiums_limit": 10,  # Força 10 para teste
    "uploads_this_month": 0,
    "uploads_limit": 20,
    "queries_today": 0,
    "queries_limit": 20
}
    }

# ------------------------ Condomínios ------------------------

@app.get("/api/condominiums")
async def get_condominiums(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
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
            "cnpj": c.cnpj,
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
    data = await request.json()
    
    count = db.query(func.count(Condominio.id)).filter(
        Condominio.sindico_id == current_user.id,
        Condominio.is_active == True
    ).scalar()
    
    if count >= 10:  # limite simples
        raise HTTPException(status_code=400, detail="Limite de condomínios atingido (máximo: 10)")
    
    condominium = Condominio(
        name=data["name"],
        address=data.get("address", ""),
        units=data.get("units", 0),
        cnpj=data.get("cnpj", ""),
        sindico_id=current_user.id
    )
    
    db.add(condominium)
    db.commit()
    db.refresh(condominium)
    
    return {
        "id": condominium.id,
        "name": condominium.name,
        "address": condominium.address,
        "units": condominium.units,
        "cnpj": condominium.cnpj,
        "created_at": condominium.created_at.isoformat() if condominium.created_at else None,
        "message": "Condomínio criado com sucesso!"
    }

# ------------------------ Documentos ------------------------

@app.get("/api/documents")
async def get_documents(
    condominium_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return get_document_list(condominium_id, current_user.id, db)
@app.delete("/api/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Deleta um documento do PostgreSQL e Pinecone
    """
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.uploaded_by == current_user.id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=404, 
            detail="Documento não encontrado ou sem permissão"
        )
    
    try:
        # Deletar do Pinecone (se configurado)
        if os.getenv('PINECONE_API_KEY'):
            from pinecone import Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index = pc.Index('alexandralex')
            
            # USAR condo_id (CORRETO!)
            namespace = f"user_{current_user.id}_cond_{doc.condo_id}"
            
            try:
                index.delete(
                    filter={"filename": doc.filename},
                    namespace=namespace
                )
            except Exception as e:
                print(f"Aviso Pinecone: {e}")
        
        # Deletar do PostgreSQL
        db.delete(doc)
        db.commit()
        
        return {"message": "Documento deletado com sucesso", "id": document_id}
        
    except Exception as e:
        db.rollback()
        print(f"Erro ao deletar documento: {e}")
        raise HTTPException(status_code=500, detail="Erro ao processar solicitação")
    
@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    condominium_id: int = Form(...),
    title: Optional[str] = Form(None),
    category: Optional[str] = Form("geral"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return await process_upload(
        file=file,
        condo_id=condominium_id,
        sindico_id=current_user.id,
        db=db
    )

# ------------------------ Query ------------------------

@app.post("/api/query")
async def query_documents(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    raw_body = await request.body()
    try:
        data = await request.json()
    except:
        data = json.loads(raw_body)
    
    query = data.get("question", "")
    condominium_id = data.get("condominium_id")
    
    if not query or not condominium_id:
        return {"answer": "Por favor, forneça uma pergunta e selecione um condomínio.", "sources": []}
    
    condominium = db.query(Condominio).filter(
        Condominio.id == condominium_id,
        Condominio.sindico_id == current_user.id
    ).first()
    
    if not condominium:
        raise HTTPException(status_code=403, detail="Sem permissão para este condomínio")
    
    try:
        rag = get_or_create_rag()
        result = rag.query(query=query, sindico_id=current_user.id, condo_id=condominium_id)
        
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

        if not result.get("success", False):
            return {"answer": f"Recebi sua pergunta: '{query}'. O sistema está processando os documentos.", "sources": []}
        
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0)
        }
        
    except Exception as e:
        return {"answer": "Erro ao processar a pergunta.", "sources": [], "error": str(e)}

# ------------------------ Dra. Alexandra ------------------------

@app.post("/api/alexandra")
async def chat_alexandra(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
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
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health():
    return {"status": "ok"}

# ------------------------ Teste ------------------------

@app.get("/api/test")
async def test():
    return {"message": "API funcionando!", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)