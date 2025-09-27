import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# Verificar conexão
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("alexandralex")

# Ver estatísticas
stats = index.describe_index_stats()
print("Namespaces no Pinecone:")
for ns, info in stats.get('namespaces', {}).items():
    print(f"  {ns}: {info['vector_count']} vectors")

# Busca simples
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Criar embedding
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input="piscina"
)
query_vec = response.data[0].embedding

# Buscar
results = index.query(
    vector=query_vec,
    top_k=3,
    namespace="user_1_cond_1",
    include_metadata=True
)

print(f"\nResultados encontrados: {len(results['matches'])}")
for match in results['matches']:
    if 'text' in match.get('metadata', {}):
        print(f"- {match['metadata']['text'][:100]}...")
