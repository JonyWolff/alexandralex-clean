from pinecone import Pinecone

# Sua chave do Pinecone
pc = Pinecone(api_key="pcsk_6VFPHB_D2TwrH3vHwD4WPYNibbGAM7n8SsMhVHU897gegwzP9WqTf3TBER8EY1csN6X7AV")
index = pc.Index("alexandralex")

# Deletar todos os vetores
index.delete(delete_all=True)
print("✅ Pinecone limpo! Pronto para novos documentos.")
