import os
from app.rag_system import get_or_create_rag

# Criar instância do RAG
rag = get_or_create_rag()

# Fazer uma busca simples
test_queries = [
    "condomínio",
    "cobrança", 
    "quotas",
    "CONCEITO DE CONDOMÍNIO LATO SENSU"
]

for query in test_queries:
    print(f"\n{'='*50}")
    print(f"Testando: {query}")
    print('='*50)
    
    result = rag.query(
        query=query,
        sindico_id=0,
        condo_id=0
    )
    
    print(f"Success: {result.get('success')}")
    print(f"Answer: {result.get('answer', 'N/A')[:200] if result.get('answer') else 'N/A'}...")
    print(f"Sources: {result.get('sources')}")
    print(f"Confidence: {result.get('confidence')}")
