import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Configurar
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("alexandralex")

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)

# Criar vectorstore corretamente
namespace = "user_1_cond_1"
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
    namespace=namespace
)

# Testar busca diretamente
docs = vectorstore.similarity_search("horário piscina", k=5)
print(f"Documentos encontrados: {len(docs)}")

for i, doc in enumerate(docs):
    print(f"\nDoc {i+1}:")
    print(f"Conteúdo: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")

# Se não encontrar, vamos verificar o problema
if len(docs) == 0:
    print("\n⚠️ Nenhum documento encontrado pelo LangChain!")
    print("Verificando índice diretamente...")
    
    # Buscar direto no Pinecone
    query_vec = embeddings.embed_query("piscina")
    results = index.query(
        vector=query_vec,
        top_k=5,
        namespace=namespace,
        include_metadata=True
    )
    
    print(f"Pinecone retornou {len(results['matches'])} resultados")
    
    # Pode ser problema de metadata key
    if results['matches']:
        print("\nPrimeiro resultado do Pinecone:")
        print(f"Metadata keys: {results['matches'][0].get('metadata', {}).keys()}")
