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
# NÃO MUDEI NADA AQUI - MANTIVE EXATAMENTE COMO ESTÁ FUNCIONANDO
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

# ========== LISTAGEM MELHORADA ==========
@router.get("/list")
async def list_knowledge_documents(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)  # MUDANÇA: adicionar autenticação
):
    """Listar documentos da base de conhecimento"""
    try:
        docs = db.query(KnowledgeBase).order_by(KnowledgeBase.id.desc()).all()
        
        documents_list = []
        for doc in docs:
            doc_info = {
                "id": doc.id,
                "filename": doc.filename,
                "category": doc.category if hasattr(doc, 'category') else 'geral',
                "description": doc.description if hasattr(doc, 'description') else '',
                "chunks_count": doc.chunks_count if hasattr(doc, 'chunks_count') else 0
            }
            documents_list.append(doc_info)
        
        return {
            "success": True,
            "documents": documents_list,
            "total": len(documents_list)
        }
        
    except Exception as e:
        print(f"Erro ao listar documentos: {e}")
        return {"success": False, "error": str(e), "documents": []}

# ========== BUSCA MELHORADA ==========
@router.post("/search")
async def search_knowledge_base(
    query: str = Form(...),
    condo_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)  # MUDANÇA: adicionar autenticação
):
    """Busca na Base de Conhecimento"""
    
    from .rag_system import get_or_create_rag
    
    try:
        rag = get_or_create_rag()
        
        print(f"DEBUG SEARCH: Buscando '{query}' na base de conhecimento")
        
        # Usar query_documents em vez de search_documents
        search_results = rag.query_documents(
            query=query,
            sindico_id=0,  # 0 = Base Geral
            condo_id=0,    # 0 = Base Geral
            k=10           # Top 10 resultados
        )
        
        # Formatar resultados
        formatted_results = []
        
        if search_results and 'matches' in search_results:
            for match in search_results['matches']:
                formatted_results.append({
                    'id': match.get('id', ''),
                    'score': match.get('score', 0),
                    'text': match.get('metadata', {}).get('text', ''),
                    'metadata': {
                        'filename': match.get('metadata', {}).get('filename', 'Documento'),
                        'category': match.get('metadata', {}).get('category', 'geral'),
                        'chunk_index': match.get('metadata', {}).get('chunk_index', 0)
                    }
                })
        
        print(f"DEBUG SEARCH: Encontrados {len(formatted_results)} resultados")
        
        return {
            "success": True,
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }
        
    except Exception as e:
        print(f"ERRO na busca: {e}")
        import traceback
        print(f"TRACEBACK: {traceback.format_exc()}")
        
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
        
        return {
            "success": True,
            "total_documents": doc_count,
            "status": "operational"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "status": "error"
        }