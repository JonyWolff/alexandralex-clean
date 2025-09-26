#!/usr/bin/env python3
"""Debug script para verificar o estado do Pinecone"""

from pinecone import Pinecone
from datetime import datetime

# Configurar Pinecone
API_KEY = "pcsk_6VFPHB_D2TwrH3vHwD4WPYNibbGAM7n8SsMhVHU897gegwzP9WqTf3TBER8EY1csN6X7AV"
pc = Pinecone(api_key=API_KEY)
index = pc.Index("alexandralex")

print("=" * 60)
print("DEBUG PINECONE - ANÁLISE COMPLETA")
print(f"Data/Hora: {datetime.now()}")
print("=" * 60)

# 1. Ver TODOS os namespaces
print("\n1. TODOS OS NAMESPACES E CONTAGENS:")
print("-" * 40)
stats = index.describe_index_stats()
total_vectors = 0
for ns, data in stats.namespaces.items():
    count = data.vector_count
    print(f"  {ns}: {count} vetores")
    total_vectors += count
print(f"\nTOTAL GERAL: {total_vectors} vetores")

# 2. Verificar Base de Conhecimento
print("\n2. DOCUMENTOS NA BASE DE CONHECIMENTO (user_0_cond_0):")
print("-" * 40)
try:
    results = index.query(
        vector=[0.1]*1536,
        top_k=100,
        namespace="user_0_cond_0",
        include_metadata=True
    )
    
    filenames = set()
    for match in results.matches:
        if 'filename' in match.metadata:
            filenames.add(match.metadata['filename'])
    
    for filename in sorted(filenames):
        print(f"  - {filename}")
    print(f"\nTotal de arquivos únicos: {len(filenames)}")
    
except Exception as e:
    print(f"  ERRO: {e}")

# 3. Verificar namespace do Galileu (Jony)
print("\n3. DOCUMENTOS DO GALILEU (user_2_cond_12):")
print("-" * 40)
try:
    results = index.query(
        vector=[0.1]*1536,
        top_k=100,
        namespace="user_2_cond_12",
        include_metadata=True
    )
    
    if results.matches:
        for match in results.matches:
            md = match.metadata
            print(f"  ID: {match.id}")
            print(f"  Filename: {md.get('filename', 'N/A')}")
            print(f"  Text preview: {md.get('text', '')[:100]}...")
            print()
    else:
        print("  NAMESPACE VAZIO!")
        
except Exception as e:
    print(f"  ERRO: {e}")

# 4. Verificar namespace do Santo Agostinho (Julia)
print("\n4. DOCUMENTOS DO SANTO AGOSTINHO (user_1_cond_9):")
print("-" * 40)
try:
    results = index.query(
        vector=[0.1]*1536,
        top_k=100,
        namespace="user_1_cond_9",
        include_metadata=True
    )
    
    if results.matches:
        for match in results.matches:
            md = match.metadata
            print(f"  ID: {match.id}")
            print(f"  Filename: {md.get('filename', 'N/A')}")
            print()
    else:
        print("  NAMESPACE NÃO EXISTE OU ESTÁ VAZIO!")
        
except Exception as e:
    print(f"  ERRO: {e}")

print("\n" + "=" * 60)
print("FIM DO DEBUG")
print("=" * 60)
