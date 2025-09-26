from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_6VFPHB_D2TwrH3vHwD4WPYNibbGAM7n8SsMhVHU897gegwzP9WqTf3TBER8EY1csN6X7AV")
index = pc.Index("alexandralex")

# Verificar namespace da base de conhecimento
stats = index.describe_index_stats()
print("\n=== ESTATÍSTICAS DO PINECONE ===")
print(f"Total de vetores: {stats.total_vector_count}")

# Ver namespaces
for ns, info in stats.namespaces.items():
    print(f"\nNamespace: {ns}")
    print(f"  Vetores: {info['vector_count']}")

# Verificar especificamente user_0_cond_0 (Base de Conhecimento)
namespace = "user_0_cond_0"
sample = index.query(
    vector=[0.1] * 1536,  # Vector dummy
    top_k=5,
    namespace=namespace,
    include_metadata=True
)

print(f"\n=== DOCUMENTOS NA BASE DE CONHECIMENTO ({namespace}) ===")
for match in sample.matches:
    if match.metadata:
        print(f"- {match.metadata.get('filename', 'sem nome')}: {match.metadata.get('category', 'sem categoria')}")
