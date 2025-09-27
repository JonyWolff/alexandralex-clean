from app.rag_langchain import get_or_create_langchain_rag

rag = get_or_create_langchain_rag()

# Testar com diferentes perguntas
perguntas = [
    "piscina",
    "horário piscina",
    "8h 20h",
    "funcionamento"
]

for pergunta in perguntas:
    print(f"\nTestando: '{pergunta}'")
    result = rag.query(
        question=pergunta,
        sindico_id=1,
        condo_id=1,
        search_mode="condo_only",
        k=5  # Buscar mais documentos
    )
    
    # Mostrar apenas início da resposta
    resposta = result.get('answer', 'Sem resposta')
    if "Fonte:" in resposta:
        resposta = resposta.split('\n\n')[1] if '\n\n' in resposta else resposta
    
    print(f"Resposta: {resposta[:150]}...")
    
    # Verificar se encontrou documentos
    sources = result.get('sources', [])
    print(f"Fontes encontradas: {len(sources)}")
