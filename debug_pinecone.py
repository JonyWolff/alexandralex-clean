import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Conectar ao Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("alexandralex")

# Verificar namespace user_1_cond_1
namespace = "user_1_cond_1"

# Buscar estatísticas
stats = index.describe_index_stats()
print(f"Estatísticas do índice: {stats}")

# Fazer query direto no Pinecone
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)

# Criar embedding da pergunta
query_embedding = embeddings.embed_query("horário piscina funcionamento")

# Buscar no Pinecone
results = index.query(
    vector=query_embedding,
    top_k=5,
    namespace=namespace,
    include_metadata=True
)

print(f"\nResultados encontrados no namespace {namespace}:")
for match in results['matches']:
    print(f"Score: {match['score']:.3f}")
    if 'metadata' in match and 'text' in match['metadata']:
        print(f"Texto: {match['metadata']['text'][:200]}...")
    print("-" * 50)

# Se não encontrar nada, listar todos os namespaces
if not results['matches']:
    print("\nNenhum resultado encontrado. Verificando namespaces...")
    if 'namespaces' in stats:
        for ns, info in stats['namespaces'].items():
            print(f"Namespace: {ns}, Vectors: {info['vector_count']}")
