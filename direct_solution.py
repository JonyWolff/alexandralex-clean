# Solução direta que funciona com Pinecone
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

# Clientes
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("alexandralex")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_direct(question: str, namespace: str):
    """Busca direta no Pinecone e gera resposta com OpenAI"""
    
    # Criar embedding da pergunta
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=question
    )
    query_vec = response.data[0].embedding
    
    # Buscar no Pinecone
    results = index.query(
        vector=query_vec,
        top_k=5,
        namespace=namespace,
        include_metadata=True
    )
    
    # Montar contexto
    context_parts = []
    for match in results['matches']:
        if 'text' in match.get('metadata', {}):
            context_parts.append(match['metadata']['text'])
    
    context = "\n\n".join(context_parts)
    
    if not context:
        return "Não encontrei informações sobre isso nos documentos."
    
    # Gerar resposta com GPT
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é a Dra. Alexandra, especialista em direito condominial. Responda baseando-se APENAS no contexto fornecido."},
            {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {question}"}
        ],
        temperature=0.3
    )
    
    return completion.choices[0].message.content

# Testar
namespace = "user_1_cond_1"
question = "Qual o horário da piscina?"

print(f"Pergunta: {question}")
print(f"Namespace: {namespace}")
print("\nResposta:")
print(query_direct(question, namespace))
