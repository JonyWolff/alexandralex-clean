import os
print("Teste da Base de Conhecimento")
print("Verificando se conseguimos acessar o sistema...")

try:
    from app.rag_system import get_or_create_rag
    print("✓ Sistema RAG encontrado")
except:
    print("✗ Erro ao acessar sistema RAG")
