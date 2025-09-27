import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

load_dotenv()

# Configurar
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("alexandralex")

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)

namespace = "user_1_cond_1"

# Limpar e recriar o namespace
print(f"Limpando namespace {namespace}...")
index.delete(delete_all=True, namespace=namespace)

# Criar documentos do LangChain corretamente
documents = [
    Document(
        page_content="REGULAMENTO DA PISCINA - Art. 1: O horário de funcionamento da piscina é das 8h às 20h todos os dias.",
        metadata={"source": "regulamento_piscina.txt", "tipo": "regras"}
    ),
    Document(
        page_content="SALÃO DE FESTAS - O salão de festas está disponível das 9h às 23h. Necessário reservar com antecedência.",
        metadata={"source": "salao_festas.txt", "tipo": "regras"}
    ),
    Document(
        page_content="REGRAS GERAIS - Proibido som alto na área da piscina. Crianças devem estar acompanhadas.",
        metadata={"source": "regras_gerais.txt", "tipo": "regras"}
    )
]

# Criar vectorstore e adicionar documentos
print(f"Adicionando documentos ao namespace {namespace}...")
vectorstore = PineconeVectorStore.from_documents(
    documents,
    embeddings,
    index_name="alexandralex",
    namespace=namespace
)

# Testar busca
print("\nTestando busca...")
results = vectorstore.similarity_search("horário da piscina", k=3)

for i, doc in enumerate(results):
    print(f"\nResultado {i+1}:")
    print(f"Conteúdo: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")

# Testar com query completa
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.3
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

print("\n" + "="*50)
print("Testando pergunta completa...")
response = qa_chain.invoke({"query": "Qual o horário de funcionamento da piscina?"})
print(f"Resposta: {response['result']}")
