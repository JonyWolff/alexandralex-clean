from app.rag_langchain import get_or_create_langchain_rag

rag = get_or_create_langchain_rag()

# Adicionar documento mais completo
documento_completo = """
REGULAMENTO DA PISCINA - CONDOMÍNIO CHERRY HILL

Art. 1 - HORÁRIO DE FUNCIONAMENTO
O horário de funcionamento da piscina será das 8h às 20h, todos os dias.

Art. 2 - REGRAS GERAIS
- Proibido som alto na área da piscina
- Obrigatório o uso de trajes de banho adequados
- Crianças menores de 12 anos devem estar acompanhadas

Art. 3 - MANUTENÇÃO
A piscina será fechada às terças-feiras para manutenção.
"""

# Indexar documento completo
result = rag.process_txt(
    txt_content=documento_completo,
    sindico_id=1,
    condo_id=1,
    title="Regulamento_Piscina_Completo.txt",
    category="regulamento"
)
print(f"Indexação: {result}")

# Testar query novamente
result = rag.query(
    question="Qual o horário de funcionamento da piscina?",
    sindico_id=1,
    condo_id=1,
    search_mode="condo_only"
)
print(f"\nResposta: {result['answer']}")
