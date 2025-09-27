from app.rag_langchain import get_or_create_langchain_rag

rag = get_or_create_langchain_rag()

# Primeiro, adicionar um documento bem claro
print("1. Adicionando documento...")
result = rag.process_txt(
    txt_content="REGULAMENTO DA PISCINA: O horário de funcionamento da piscina é das 8h às 20h todos os dias.",
    sindico_id=1,
    condo_id=1,
    title="regulamento_piscina.txt",
    category="regras"
)
print(f"   Upload: {result}")

# Agora testar a busca
print("\n2. Testando busca...")
result = rag.query(
    question="Qual o horário da piscina?",
    sindico_id=1,
    condo_id=1,
    search_mode="condo_only",
    k=5
)

print(f"\n3. Resposta completa:")
print(result['answer'])
