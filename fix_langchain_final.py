import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

load_dotenv()

# Limpar e recriar corretamente
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("alexandralex")

namespace = "user_1_cond_1"

# Limpar namespace
print(f"Limpando namespace {namespace}...")
index.delete(delete_all=True, namespace=namespace)

# Criar embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)

# Criar documentos do jeito que o LangChain espera
documents = [
    Document(
        page_content="REGULAMENTO DA PISCINA: O horário de funcionamento da piscina é das 8h às 20h todos os dias.",
        metadata={"source": "regulamento_piscina.txt", "tipo": "regras"}
    ),
    Document(
        page_content="SALÃO DE FESTAS: O salão está disponível das 9h às 23h. Reservar com 48h de antecedência.",
        metadata={"source": "salao_festas.txt", "tipo": "regras"}
    )
]

# Criar vectorstore e adicionar documentos do jeito correto
print(f"Adicionando documentos...")
vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name="alexandralex",
    namespace=namespace
)

# Testar busca
print("\nTestando busca...")
docs = vectorstore.similarity_search("horário piscina", k=3)
print(f"Documentos encontrados: {len(docs)}")
for doc in docs:
    print(f"- {doc.page_content[:100]}...")

# Agora testar com o sistema completo
print("\n" + "="*50)
from app.rag_langchain import get_or_create_langchain_rag

rag = get_or_create_langchain_rag()
result = rag.query(
    "Qual o horário da piscina?",
    sindico_id=1,
    condo_id=1,
    search_mode="condo_only"
)

print("RESPOSTA FINAL:")
print(result['answer'])
