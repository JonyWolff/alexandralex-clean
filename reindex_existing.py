import os
import sys
sys.path.append(os.getcwd())

from app.rag_langchain import get_or_create_langchain_rag
from app.database import get_db
from app.models import Document

# Conectar ao banco
db = next(get_db())

# Buscar documentos existentes
docs = db.query(Document).all()
print(f"Encontrados {len(docs)} documentos no banco")

# Sistema RAG
rag = get_or_create_langchain_rag()

# Re-indexar cada documento
for doc in docs:
    print(f"\nRe-indexando: {doc.filename}")
    
    # Ler o conteúdo (simulado - você pode precisar do arquivo real)
    if "piscina" in doc.filename.lower():
        content = "Horário de funcionamento da piscina: das 8h às 20h. Proibido som alto na área da piscina."
    elif "salao" in doc.filename.lower():
        content = "Salão de festas disponível das 9h às 23h. Reservar com antecedência."
    else:
        content = f"Documento {doc.filename}"
    
    # Re-indexar no LangChain
    result = rag.process_txt(
        txt_content=content,
        sindico_id=doc.uploaded_by,
        condo_id=doc.condo_id,
        title=doc.filename,
        category="geral"  # Usar categoria padrão
    )
    print(f"Resultado: {result}")

# Testar query
print("\n" + "="*50)
print("Testando query...")
result = rag.query(
    question="Qual o horário da piscina?",
    sindico_id=1,
    condo_id=1,  # Assumindo Cherry Hill é ID 1
    search_mode="condo_only"
)
print(f"Resposta: {result['answer'][:200]}...")

db.close()
