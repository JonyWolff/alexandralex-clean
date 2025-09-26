from pinecone import Pinecone
import time

pc = Pinecone(api_key="pcsk_6VFPHB_D2TwrH3vHwD4WPYNibbGAM7n8SsMhVHU897gegwzP9WqTf3TBER8EY1csN6X7AV")
index = pc.Index("alexandralex")

namespace = "user_0_cond_0"
print(f"Deletando TODOS os vetores do namespace {namespace}...")

# Deletar tudo
index.delete(delete_all=True, namespace=namespace)
time.sleep(2)

# Verificar
stats = index.describe_index_stats()
if namespace in stats.namespaces:
    print(f"Ainda tem {stats.namespaces[namespace]['vector_count']} vetores")
else:
    print("✅ Namespace completamente limpo!")
