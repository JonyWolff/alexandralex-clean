from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import tempfile

from .database import get_db
from .auth import get_current_user
from .models import KnowledgeBase

# Router com prefixo e tags (vai aparecer no Swagger agrupado)
router = APIRouter(
    prefix="/api/knowledge",
    tags=["Knowledge Base"]
)

# Lista de admins (adicione os emails autorizados)
ADMIN_EMAILS = ["juliawolff6@gmail.com", "jony.wolff@gmail.com"]

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


# ========== NOVA FUNÇÃO DE UPLOAD ==========
@router.post("/upload")
async def upload_knowledge_document(
    file: UploadFile = File(...),
    category: str = Form("lei_federal"),
    description: str = Form(""),
    db: Session = Depends(get_db),
    admin = Depends(verify_admin)
):
    """Upload de documento para Base de Conhecimento com processamento Pinecone"""
    
    from .rag_system import get_or_create_rag
    
    try:
        # Salvar arquivo temporário
        content = await file.read()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(content)
        temp_file.close()
        
        # Processar com RAG no namespace especial
        rag = get_or_create_rag()
        result = await rag.process_document_to_pinecone(
            file_path=temp_file.name,
            sindico_id=0,  # 0 = Base de Conhecimento Geral
            condo_id=0,    # 0 = Base de Conhecimento Geral
            doc_id=f"kb_{category}_{file.filename}"
        )
        
        # Salvar no banco
        kb_doc = KnowledgeBase(
            filename=file.filename,
            category=category,
            description=description,
            processed=True,
            chunks_count=result.get('chunks_created', 0)
        )
        db.add(kb_doc)
        db.commit()
        
        # Limpar arquivo temporário
        os.unlink(temp_file.name)
        
        return {
            "success": True,
            "message": f"Documento processado: {result.get('chunks_created', 0)} chunks criados",
            "category": category
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# ========== LISTAGEM DE DOCUMENTOS ==========
@router.get("/list")
async def list_knowledge_documents(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Listar documentos da base"""
    docs = db.query(KnowledgeBase).all()
    return [{"id": doc.id, "filename": doc.filename} for doc in docs]


# ========== NOVA FUNÇÃO DE BUSCA ==========
@router.post("/search")
async def search_knowledge_base(
    query: str = Form(...),
    condo_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Busca híbrida: Base de Conhecimento + Documentos do Condomínio"""
    
    from .rag_system import get_or_create_rag
    
    rag = get_or_create_rag()
    results = []
    
    # Buscar na Base de Conhecimento Geral
    kb_results = await rag.search_documents(
        query=query,
        namespace="user_0_cond_0"  # namespace da base geral
    )
    
    # Se forneceu condo_id, buscar também nos docs do condomínio
    if condo_id:
        sindico_id = current_user['id']
        condo_results = await rag.search_documents(
            query=query,
            namespace=f"user_{sindico_id}_cond_{condo_id}"
        )
        results.extend(condo_results)
    
    # Combinar e ranquear resultados
    results.extend(kb_results)
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return {
        "query": query,
        "results": results[:10],  # Top 10 resultados
        "sources": {
            "knowledge_base": len(kb_results),
            "condominium": len(results) - len(kb_results)
        }
    }