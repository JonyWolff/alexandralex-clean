from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import os

from .database import get_db
from .auth import get_current_user

# Router com prefixo e tags (vai aparecer no Swagger agrupado)
router = APIRouter(
    prefix="/api/knowledge",
    tags=["Knowledge Base"]
)

# Lista de admins (adicione os emails autorizados)
ADMIN_EMAILS = ["juliawolff6@gmail.com", "outro_admin@exemplo.com"]

def verify_admin(current_user=Depends(get_current_user)):
    """Verifica se o usuário é admin"""
    if current_user["email"] not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Apenas administradores podem acessar")
    return current_user

# 🚀 Endpoint de teste
@router.get("/test")
async def test_knowledge():
    """Teste básico para verificar integração do router"""
    return {"message": "✅ Base de Conhecimento funcionando!"}
# ========== ADICIONE ISSO NO FINAL DO ARQUIVO ==========

from .models import KnowledgeBase
import tempfile

@router.post("/upload")
async def upload_knowledge_document(
    file: UploadFile = File(...),
    category: str = Form("lei_federal"),
    description: str = Form(""),
    db: Session = Depends(get_db),
    admin = Depends(verify_admin)
):
    """Upload de documento para Base de Conhecimento (apenas admins)"""
    
    try:
        # Por enquanto, só salvamos no banco
        kb_doc = KnowledgeBase(
            filename=file.filename,
            category=category,
            description=description,
            processed=False
        )
        db.add(kb_doc)
        db.commit()
        
        return {
            "success": True,
            "message": f"Documento {file.filename} salvo",
            "category": category
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/list")
async def list_knowledge_documents(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Listar documentos da base"""
    docs = db.query(KnowledgeBase).all()
    return [{"id": doc.id, "filename": doc.filename} for doc in docs]