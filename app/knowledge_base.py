from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import tempfile
from datetime import datetime

from .database import get_db
from .auth import get_current_user
from .models import KnowledgeBase

# Router com prefixo e tags
router = APIRouter(
    prefix="/api/knowledge",
    tags=["Knowledge Base"]
)

# Lista de admins
ADMIN_EMAILS = ["juliawolff6@gmail.com", "jony.wolff@gmail.com"]

def verify_admin(current_user=Depends(get_current_user)):
    """Verifica se o usuário é admin"""
    if current_user["email"] not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Apenas administradores podem acessar")
    return current_user

# Endpoint de teste
@router.get("/test")
async def test_knowledge():
    """Teste básico para verificar integração do router"""
    return {"message": "✅ Base de Conhecimento funcionando!"}

# ========== UPLOAD DE DOCUMENTO ==========
@router.post("/upload")
async def upload_knowledge_document(
    file: UploadFile = File(...),
    category: str = Form("modelo_documento"),
    description: str = Form(""),
    db: Session = Depends(get_db),
    admin = None
):
    """Upload de documento para Base de Conhecimento com processamento Pinecone"""
    
    from .rag_system import get_or_create_rag
    
    # Inicializar variáveis
    temp_file_path = None
    
    try:
        # Ler conteúdo do arquivo
        content = await file.read()
        
        # Criar arquivo temporário se for PDF
        if file.filename.lower().endswith('.pdf'):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_file.write(content)
            temp_file.close()
            temp_file_path = temp_file.name
        
        # Obter instância do RAG
        rag = get_or_create_rag()
        
        # Gerar ID limpo (apenas ASCII)
        doc_id = f"kb_{category}_{int(datetime.now().timestamp())}"
        
        # Processar documento
        print(f"DEBUG: Processando arquivo {file.filename}")
        print(f"DEBUG: Categoria: {category}")
        print(f"DEBUG: Doc ID: {doc_id}")
        
        # Chamar o método correto (sem await pois não é assíncrono)
        process_result = rag.process_pdf_content(
            pdf_content=content,
            sindico_id=0,  # 0 = Base de Conhecimento Geral
            condo_id=0,    # 0 = Base de Conhecimento Geral
            doc_id=doc_id,
            metadata={
                "filename": file.filename,
                "category": category,
                "description": description,
                "upload_date": datetime.now().isoformat()
            }
        )
        
        # Verificar se o processamento foi bem sucedido
        if not process_result.get('success', False):
            error_msg = process_result.get('error', 'Erro desconhecido no processamento')
            print(f"ERRO no processamento: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Erro ao processar documento: {error_msg}")
        
        # Salvar registro no banco de dados
        kb_doc = KnowledgeBase(
            filename=file.filename,
            category=category,
            description=description,
            processed=True,
            chunks_count=process_result.get('chunks_created', 0)
        )
        db.add(kb_doc)
        db.commit()
        
        # Preparar resposta de sucesso
        response = {
            "success": True,
            "message": f"✅ Documento processado com sucesso!",
            "details": {
                "filename": file.filename,
                "category": category,
                "chunks_created": process_result.get('chunks_created', 0),
                "embeddings_created": process_result.get('embeddings_created', 0),
                "doc_id": doc_id
            }
        }
        
        print(f"SUCESSO: {response}")
        return response
        
    except HTTPException:
        # Re-lançar exceções HTTP
        raise
        
    except Exception as e:
        # Log do erro
        print(f"ERRO GERAL: {str(e)}")
        import traceback
        print(f"TRACEBACK: {traceback.format_exc()}")
        
        # Retornar erro formatado
        return {
            "success": False,
            "error": str(e),
            "message": f"❌ Erro ao processar documento: {str(e)}"
        }
        
    finally:
        # Limpar arquivo temporário se existir
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"DEBUG: Arquivo temporário removido: {temp_file_path}")
            except:
                pass

# ========== LISTAGEM DE DOCUMENTOS ==========
@router.get("/list")
async def list_knowledge_documents(
    db: Session = Depends(get_db),
    current_user = None
):
    """Listar documentos da base de conhecimento"""
    try:
        docs = db.query(KnowledgeBase).all()
        return {
            "success": True,
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "category": doc.category,
                    "description": doc.description,
                    "processed": doc.processed,
                    "chunks_count": doc.chunks_count,
                    "created_at": doc.created_at.isoformat() if hasattr(doc, 'created_at') and doc.created_at else None
                }
                for doc in docs
            ],
            "total": len(docs)
        }
    except Exception as e:
        print(f"Erro ao listar documentos: {e}")
        return {"success": False, "error": str(e), "documents": []}

# ========== BUSCA NA BASE ==========
@router.post("/search")
async def search_knowledge_base(
    query: str = Form(...),
    condo_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_user = None
):
    """Busca híbrida: Base de Conhecimento + Documentos do Condomínio"""
    
    from .rag_system import get_or_create_rag
    
    try:
        rag = get_or_create_rag()
        results = []
        
        # Buscar na Base de Conhecimento Geral
        print(f"DEBUG: Buscando '{query}' na base de conhecimento")
        kb_results = rag.search_documents(
            query=query,
            namespace="user_0_cond_0"  # namespace da base geral
        )
        
        # Se forneceu condo_id, buscar também nos docs do condomínio
        if condo_id and current_user:
            sindico_id = current_user.get('id', 0)
            print(f"DEBUG: Buscando também no condomínio {condo_id}")
            condo_results = rag.search_documents(
                query=query,
                namespace=f"user_{sindico_id}_cond_{condo_id}"
            )
            results.extend(condo_results)
        
        # Combinar e ranquear resultados
        results.extend(kb_results)
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return {
            "success": True,
            "query": query,
            "results": results[:10],  # Top 10 resultados
            "sources": {
                "knowledge_base": len(kb_results),
                "condominium": len(results) - len(kb_results)
            },
            "total_results": len(results)
        }
        
    except Exception as e:
        print(f"Erro na busca: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "results": []
        }

# DEBUG - Endpoint para verificar status
@router.get("/debug-user")
async def debug_user(current_user = None):
    """Debug: verificar usuário e permissões"""
    return {
        "user": current_user,
        "is_admin": current_user.get("email") in ADMIN_EMAILS if current_user else False,
        "admin_emails": ADMIN_EMAILS
    }

# Status da base de conhecimento
@router.get("/status")
async def knowledge_base_status(db: Session = Depends(get_db)):
    """Verificar status da base de conhecimento"""
    try:
        doc_count = db.query(KnowledgeBase).count()
        processed_count = db.query(KnowledgeBase).filter(KnowledgeBase.processed == True).count()
        
        return {
            "success": True,
            "total_documents": doc_count,
            "processed_documents": processed_count,
            "pending_documents": doc_count - processed_count,
            "status": "operational"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "status": "error"
        }