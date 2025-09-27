import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

load_dotenv()

# Verificar o que está no Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("alexandralex")

# Verificar namespace
namespace = "user_1_cond_1"
stats = index.describe_index_stats()
print(f"Vectors em {namespace}: {stats['namespaces'].get(namespace, {}).get('vector_count', 0)}")

# Buscar diretamente
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)

query_vec = embeddings.embed_query("horário piscina")
results = index.query(
    vector=query_vec,
    top_k=5,
    namespace=namespace,
    include_metadata=True
)

print(f"\nDocumentos no Pinecone:")
for match in results['matches']:
    print(f"- Score: {match['score']:.3f}")
    if 'text' in match.get('metadata', {}):
        print(f"  Texto: {match['metadata']['text'][:100]}...")
    else:
        print(f"  Metadata keys: {match.get('metadata', {}).keys()}")
    print()

# Agora vamos testar o LangChain diretamente
from langchain_pinecone import PineconeVectorStore

print("\nTestando PineconeVectorStore:")
vectorstore = PineconeVectorStore(
    index_name="alexandralex",
    embedding=embeddings,
    text_key="text",
    namespace=namespace
)

# Buscar documentos
docs = vectorstore.similarity_search("horário piscina", k=5)
print(f"Documentos encontrados pelo LangChain: {len(docs)}")

if len(docs) == 0:
    print("\n❌ LangChain não está encontrando documentos!")
    print("Possível problema: metadata 'text' vs 'page_content'")
    
    # Vamos verificar o primeiro match do Pinecone
    if results['matches']:
        first_match = results['matches'][0]
        print(f"\nMetadata do Pinecone: {first_match.get('metadata', {}).keys()}")
        
        # O problema pode ser que o LangChain espera 'page_content' mas temos 'text'
        if 'page_content' in first_match.get('metadata', {}):
            print("Tem 'page_content' na metadata")
        if 'text' in first_match.get('metadata', {}):
            print("Tem 'text' na metadata")
