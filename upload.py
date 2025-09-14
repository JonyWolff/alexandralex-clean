# app/upload.py
import os
import re
from typing import Dict, Any, List, Tuple
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from unicodedata import normalize

from .models import Document, Condominio
from .rag_system import get_or_create_rag

def validate_filename(filename: str) -> Tuple[bool, str]:
    """Valida se o nome do arquivo é compatível com Pinecone"""
    # Caracteres problemáticos conhecidos
    invalid_chars = {
        'ç': 'c', 'ã': 'a', 'õ': 'o', 'á': 'a', 'é': 'e', 
        'í': 'i', 'ó': 'o', 'ú': 'u', 'â': 'a', 'ê': 'e', 
        'ô': 'o', 'à': 'a', 'ñ': 'n', 'ü': 'u', 'Ç': 'C',
        'Ã': 'A', 'Õ': 'O', 'Á': 'A', 'É': 'E', 'Í': 'I',
        'Ó': 'O', 'Ú': 'U', 'Â': 'A', 'Ê': 'E', 'Ô': 'O'
    }
    
    # Verificar se tem caracteres especiais
    found_invalid = []
    for char in invalid_chars.keys():
        if char in filename:
            found_invalid.append(char)
    
    if found_invalid:
        # Criar sugestão de nome
        suggestion = filename
        for char, replacement in invalid_chars.items():
            suggestion = suggestion.replace(char, replacement)
        
        chars_list = ', '.join(f"'{c}'" for c in found_invalid)
        return False, f"⚠️ Nome do arquivo contém caracteres não permitidos ({chars_list}). Sugestão: {suggestion}"
    
    # Verificar outros símbolos (permitir apenas letras, números, ponto, hífen, underscore e espaço)
    if not re.match(r'^[a-zA-Z0-9._\- ]+$', filename):
        return False, "⚠️ Use apenas letras (sem acentos), números, espaços e os símbolos: . - _"
    
    return True, "OK"

def clean_filename_for_id(filename: str) -> str:
    """Remove caracteres especiais do nome do arquivo para usar como ID interno"""
    # Remover acentos
    filename = normalize('NFKD', filename)
    filename = filename.encode('ASCII', 'ignore').decode('ASCII')
    # Substituir espaços e caracteres não ASCII por underscore
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    # Remover underscores múltiplos
    filename = re.sub(r'_+', '_', filename)
    # Remover underscore do início e fim
    filename = filename.strip('_')
    return filename.lower()

async def process_upload(
    file: UploadFile,
    condo_id: int,
    sindico_id: int,
    db: Session
) -> Dict[str, Any]:
    """Processa upload de arquivo e indexa no Pinecone via métodos do RAG"""
    
    print(f"DEBUG UPLOAD: Iniciando upload de {file.filename}")
    print(f"DEBUG UPLOAD: Tipo do arquivo: {file.content_type}")
    
    # VALIDAR NOME DO ARQUIVO
    is_valid, error_msg = validate_filename(file.filename)
    if not is_valid:
        print(f"DEBUG UPLOAD: Nome inválido - {error_msg}")
        raise HTTPException(
            status_code=400,
            detail=error_msg
        )
    
    # Validar tipo de arquivo
    allowed_types = {".pdf", ".txt"}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    print(f"DEBUG UPLOAD: Extensão detectada: {file_extension}")
    
    if file_extension not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado: {file_extension}. Use apenas PDF ou TXT."
        )
    
    # VERIFICAR LIMITE DE DOCUMENTOS (20 por condomínio)
    doc_count = db.query(Document).filter(
        Document.condo_id == condo_id,
        Document.processed == True
    ).count()
    
    print(f"DEBUG UPLOAD: Documentos existentes no condomínio: {doc_count}/20")
    
    if doc_count >= 20:
        raise HTTPException(
            status_code=400,
            detail="⚠️ Limite máximo de 20 documentos por condomínio atingido. Delete documentos antigos antes de adicionar novos."
        )
    
    # Validar tamanho (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    file_content = await file.read()
    
    print(f"DEBUG UPLOAD: Tamanho do arquivo: {len(file_content)} bytes")
    
    if len(file_content) > max_size:
        raise HTTPException(
            status_code=400,
            detail="Arquivo muito grande. Tamanho máximo permitido: 10MB"
        )
    
    # Verificar se o condomínio existe e pertence ao síndico
    condominium = db.query(Condominio).filter(
        Condominio.id == condo_id,
        Condominio.sindico_id == sindico_id,
        Condominio.is_active == True
    ).first()
    
    if not condominium:
        raise HTTPException(
            status_code=404,
            detail="Condomínio não encontrado ou sem permissão"
        )
    
    try:
        rag = get_or_create_rag()
        
        # ID do documento - CORRIGIDO para remover caracteres especiais
        clean_name = clean_filename_for_id(file.filename)
        generated_doc_id = f"{clean_name}_s{sindico_id}_c{condo_id}_{int(datetime.now().timestamp())}"
        
        print(f"DEBUG UPLOAD: Nome limpo: {clean_name}")
        print(f"DEBUG UPLOAD: Doc ID gerado: {generated_doc_id}")
        
        # Namespace
        namespace = f"user_{sindico_id}_cond_{condo_id}"
        
        metadata = {
            "filename": file.filename,  # Nome original com acentos
            "title": file.filename,
            "category": "upload",
            "upload_date": datetime.now().isoformat(),
            "sindico_id": str(sindico_id),
            "condo_id": str(condo_id),
            "namespace": namespace
        }

        # Processamento via métodos do RAG
        if file_extension == ".pdf":
            print(f"DEBUG UPLOAD: Processando PDF...")
            result = rag.process_pdf_content(
                pdf_content=file_content,
                sindico_id=sindico_id,
                condo_id=condo_id,
                doc_id=generated_doc_id,
                metadata=metadata
            )
        else:  # .txt
            print(f"DEBUG UPLOAD: Processando TXT...")
            text = file_content.decode("utf-8", errors="ignore")
            print(f"DEBUG UPLOAD: Texto extraído (primeiros 200 chars): {text[:200]}")
            result = rag.process_txt_content(
                txt_content=text,
                sindico_id=sindico_id,
                condo_id=condo_id,
                doc_id=generated_doc_id,
                metadata=metadata
            )
        
        print(f"DEBUG UPLOAD: Resultado do processamento: {result}")
        
        # Normalização dos retornos
        chunks_created = int(result.get("chunks_created", result.get("chunks_count", 0)))
        embeddings_created = int(result.get("embeddings_created", chunks_created))

        if chunks_created == 0:
            raise HTTPException(
                status_code=400, 
                detail="Documento vazio ou sem texto extraível"
            )

        # Salvar no banco de dados
        doc = Document(
            filename=file.filename,
            file_path=f"pinecone:{generated_doc_id}",
            file_type=file_extension[1:],  # remove o ponto
            file_size=len(file_content),
            processed=True,
            chunks_count=chunks_created,
            condo_id=condo_id,
            uploaded_by=sindico_id
        )
        db.add(doc)
        db.commit()
        
        print(f"DEBUG UPLOAD: Documento salvo no banco com sucesso! ID: {doc.id}")
        
        return {
            "filename": file.filename,
            "condominium_id": condo_id,
            "condominium_name": condominium.name,
            "chunks_created": chunks_created,
            "embeddings_created": embeddings_created,
            "doc_id": generated_doc_id,
            "file_size": len(file_content),
            "file_type": file_extension[1:],
            "message": f"✅ Documento processado com sucesso! ({doc_count + 1}/20 documentos)",
            "success": True
        }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERRO UPLOAD: {str(e)}")
        if 'doc' in locals():
            doc.processed = False
            doc.processing_error = str(e)
            db.commit()
        
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar upload: {str(e)}"
        )

def get_document_list(
    condo_id: int,
    sindico_id: int,
    db: Session
) -> List[Dict[str, Any]]:
    """Retorna lista de documentos de um condomínio"""
    
    # Verificar permissão
    condominium = db.query(Condominio).filter(
        Condominio.id == condo_id,
        Condominio.sindico_id == sindico_id,
        Condominio.is_active == True
    ).first()
    
    if not condominium:
        return []
    
    # Buscar documentos
    documents = db.query(Document).filter(
        Document.condo_id == condo_id,
        Document.processed == True
    ).order_by(Document.created_at.desc()).all()
    
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "file_size": doc.file_size,
            "chunks_count": doc.chunks_count,
            "uploaded_at": doc.created_at.isoformat() if doc.created_at else None,
            "processed": doc.processed,
            "processing_error": doc.processing_error,
            "description": doc.description
        }
        for doc in documents
    ]

def delete_document(
    doc_id: int,
    sindico_id: int,
    db: Session
) -> bool:
    """Remove um documento do sistema"""
    
    # Buscar documento
    document = db.query(Document).filter(
        Document.id == doc_id
    ).first()
    
    if not document:
        return False
    
    # Verificar permissão (através do condomínio)
    condominium = db.query(Condominio).filter(
        Condominio.id == document.condo_id,
        Condominio.sindico_id == sindico_id
    ).first()
    
    if not condominium:
        return False
    
    try:
        # Por enquanto, apenas marcar como deletado no banco
        db.delete(document)
        db.commit()
        print(f"DEBUG DELETE: Documento {doc_id} removido com sucesso")
        return True
        
    except Exception as e:
        print(f"ERRO DELETE: {str(e)}")
        return False