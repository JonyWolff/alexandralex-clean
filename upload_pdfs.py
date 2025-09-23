#!/usr/bin/env python3
"""
Script para processar os 32 PDFs da Base de Conhecimento
Roda no Railway onde o ambiente está configurado
"""

import os
import sys
from app.rag_system import get_or_create_rag
from app.database import get_db
from app.models import KnowledgeBase
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

def main():
    print("🚀 Iniciando processamento da Base de Conhecimento...")
    
    # Conectar ao banco
    DATABASE_URL = os.getenv("DATABASE_URL")
    engine = create_engine(DATABASE_URL)
    db = Session(engine)
    
    # Buscar os 32 documentos
    docs = db.query(KnowledgeBase).all()
    print(f"📚 Encontrados {len(docs)} documentos")
    
    # Criar instância do RAG
    rag = get_or_create_rag(user_id=0, condominio_id=0)
    
    # Processar cada documento
    for i, doc in enumerate(docs, 1):
        print(f"[{i}/{len(docs)}] Processando {doc.filename}...")
        
        # Texto simulado (depois você melhora com extração real do PDF)
        text = f"""
        Documento: {doc.filename}
        Categoria: {doc.category}
        {doc.title or ''}
        {doc.description or ''}
        """
        
        # Adicionar ao Pinecone
        rag.add_document(
            content=text,
            filename=doc.filename,
            metadata={'category': doc.category},
            namespace='knowledge_base'
        )
    
    print("✅ Base de Conhecimento processada com sucesso!")
    db.close()

if __name__ == "__main__":
    main()
