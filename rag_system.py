# app/rag_system.py
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pinecone import Pinecone
from openai import OpenAI
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RAGSystem:
    """Sistema RAG usando Pinecone v6.0.0 e OpenAI (compatível com dois fluxos):
       1) Fluxo antigo: process_*_content(pdf/txt) com doc_id + metadata → gera embeddings e faz upsert direto
       2) Fluxo novo: process_*_content(pdf/txt) com title + category → usa upsert_texts
    """

    def __init__(self):
        # Configurar OpenAI
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Configurar Pinecone (nova API)
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # Conectar ao índice
        self.index_name = "alexandralex"
        self.index = self.pc.Index(self.index_name)

        # Modelo de embeddings (mantido como no original, para compatibilidade)
        self.embedding_model = "text-embedding-ada-002"

    @staticmethod
    def namespace_for(sindico_id: int, condo_id: int) -> str:
        """Gera namespace único por síndico/condomínio"""
        return f"user_{sindico_id}_cond_{condo_id}"

    # ---------------------- Helpers internos ----------------------

    def _create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Divide texto em chunks com overlap (modo original)"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start = end - overlap if end < len(text) else end

        return chunks

    def _create_chunks_tokenless(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Chunking simples por tamanho aproximado em caracteres (modo 'novo' do trecho que você enviou)."""
        if not text:
            return []
        words = text.split()
        chunks: List[str] = []
        current: List[str] = []

        for w in words:
            current.append(w)
            if len(" ".join(current)) > chunk_size:
                chunks.append(" ".join(current[:-1]))
                current = [w]

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings usando OpenAI"""
        print(f"DEBUG EMBED: Gerando embeddings para {len(texts)} textos")
        embeddings: List[List[float]] = []
        
        try:
            for i, text in enumerate(texts):
                print(f"DEBUG EMBED: Processando texto {i+1}/{len(texts)} com {len(text)} caracteres")
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                embeddings.append(response.data[0].embedding)
            print(f"DEBUG EMBED: {len(embeddings)} embeddings criados com sucesso")
        except Exception as e:
            print(f"ERRO EMBED: Falha ao criar embeddings: {str(e)}")
            raise
            
        return embeddings

    def upsert_texts(
        self,
        texts: List[str],
        namespace: str,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        base_doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Novo helper: recebe textos + metadados e realiza:
        - embeddings
        - upsert no Pinecone

        Se base_doc_id não for fornecido, ele é gerado automaticamente.
        """
        try:
            print(f"DEBUG UPSERT: Iniciando upsert de {len(texts)} textos no namespace {namespace}")
            
            if not texts:
                return {"success": False, "error": "Sem textos para indexar", "chunks_created": 0, "embeddings_created": 0}

            # Embeddings
            embeddings = self._embed(texts)

            # Preparar vetores
            if not base_doc_id:
                base_doc_id = f"doc_{int(datetime.now().timestamp())}"

            print(f"DEBUG UPSERT: Preparando {len(embeddings)} vetores com doc_id base: {base_doc_id}")

            vectors = []
            for i, (chunk, embedding) in enumerate(zip(texts, embeddings)):
                md = (metadata_list[i] if (metadata_list and i < len(metadata_list)) else {}) or {}
                vectors.append({
                    "id": f"{base_doc_id}_{i}",
                    "values": embedding,
                    "metadata": {
                        **md,
                        "text": chunk[:1000],        # limitar tamanho do metadata
                        "chunk_index": i,
                        "total_chunks": len(texts)
                    }
                })

            # Upsert
            print(f"DEBUG UPSERT: Enviando {len(vectors)} vetores para o Pinecone...")
            self.index.upsert(vectors=vectors, namespace=namespace)
            print(f"DEBUG UPSERT: Upsert concluído com sucesso!")

            return {
                "success": True,
                "chunks_created": len(texts),
                "embeddings_created": len(embeddings),
                "doc_id": base_doc_id
            }

        except Exception as e:
            logger.error(f"Erro no upsert_texts: {e}")
            print(f"ERRO UPSERT: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "chunks_created": 0,
                "embeddings_created": 0
            }

    # ---------------------- Processamento de conteúdos (modo original) ----------------------

    def process_pdf_content(
        self,
        pdf_content: bytes,
        sindico_id: int,
        condo_id: int,
        doc_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        # parâmetros do "modo novo" — opcionais
        title: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processa conteúdo PDF diretamente.
        Compatível com dois modos:
        - Modo original: usa doc_id + metadata → faz embeddings e upsert direto (como no seu código original).
        - Modo novo: usa title + category → cria metadata_list e utiliza upsert_texts.
        """
        try:
            print(f"DEBUG PDF: Iniciando processamento de PDF com {len(pdf_content)} bytes")
            print(f"DEBUG PDF: Sindico ID: {sindico_id}, Condo ID: {condo_id}")
            
            # Extrair texto do PDF
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            print(f"DEBUG PDF: PDF tem {len(pdf_reader.pages)} páginas")
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                print(f"DEBUG PDF: Página {page_num + 1} extraída com {len(page_text) if page_text else 0} caracteres")
                if page_text:
                    text += f"\n[Página {page_num + 1}]\n{page_text}"

            print(f"DEBUG PDF: Total de texto extraído: {len(text)} caracteres")
            print(f"DEBUG PDF: Primeiros 200 chars do texto: {text[:200] if text else 'VAZIO'}")

            if not text.strip():
                print("ERRO PDF: Nenhum texto foi extraído do PDF")
                return {"success": False, "error": "PDF vazio ou sem texto extraível", "chunks_created": 0, "embeddings_created": 0}

            namespace = self.namespace_for(sindico_id, condo_id)
            print(f"DEBUG PDF: Usando namespace: {namespace}")

            # --- MODO NOVO (title/category) ---
            # Se title/category forem fornecidos (ou doc_id/metadata ausentes), trilhar pelo helper upsert_texts
            if (title or category) and (metadata is None or doc_id is None):
                print(f"DEBUG PDF: Usando MODO NOVO com title={title}, category={category}")
                chunks = self._create_chunks_tokenless(text, chunk_size=1000)
                print(f"DEBUG PDF: Criados {len(chunks)} chunks")

                metadata_list: List[Dict[str, Any]] = []
                for i, _ in enumerate(chunks):
                    metadata_list.append({
                        "title": title or "Documento",
                        "category": category or "geral",
                        "sindico_id": sindico_id,
                        "condo_id": condo_id,
                        "type": "pdf"
                    })

                return self.upsert_texts(
                    texts=chunks,
                    namespace=namespace,
                    metadata_list=metadata_list
                )

            # --- MODO ORIGINAL (doc_id + metadata) ---
            print(f"DEBUG PDF: Usando MODO ORIGINAL com doc_id={doc_id}")
            # Criar chunks com overlap (como no arquivo original)
            chunks = self._create_chunks(text)
            print(f"DEBUG PDF: Criados {len(chunks)} chunks com overlap")

            # Embeddings
            embeddings = self._embed(chunks)

            # Preparar vetores
            if not doc_id:
                doc_id = f"doc_{int(datetime.now().timestamp())}"

            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vectors.append({
                    "id": f"{doc_id}_{i}",
                    "values": embedding,
                    "metadata": {
                        **(metadata or {}),
                        "text": chunk[:1000],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "sindico_id": str(sindico_id),
                        "condo_id": str(condo_id)
                    }
                })

            # Upsert
            print(f"DEBUG PDF: Fazendo upsert de {len(vectors)} vetores...")
            self.index.upsert(vectors=vectors, namespace=namespace)
            print(f"DEBUG PDF: Upsert concluído!")

            return {
                "success": True,
                "chunks_created": len(chunks),
                "embeddings_created": len(embeddings),
                "doc_id": doc_id
            }

        except Exception as e:
            logger.error(f"Erro em process_pdf_content: {e}")
            print(f"ERRO PDF: {str(e)}")
            import traceback
            print(f"ERRO PDF TRACEBACK: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "chunks_created": 0,
                "embeddings_created": 0
            }

    def process_txt_content(
        self,
        txt_content: str,
        sindico_id: int,
        condo_id: int,
        doc_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        # parâmetros do "modo novo" — opcionais
        title: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processa conteúdo TXT diretamente.
        Compatível com dois modos:
        - Modo original: usa doc_id + metadata → faz embeddings e upsert direto (como no seu código original).
        - Modo novo: usa title + category → cria metadata_list e utiliza upsert_texts.
        """
        try:
            print(f"DEBUG TXT: Iniciando processamento de TXT")
            print(f"DEBUG TXT: Sindico ID: {sindico_id}, Condo ID: {condo_id}")
            
            # Garantir str
            if isinstance(txt_content, bytes):
                txt_content = txt_content.decode("utf-8", errors="ignore")
                print(f"DEBUG TXT: Convertido de bytes para string")

            print(f"DEBUG TXT: Tamanho do texto: {len(txt_content)} caracteres")
            print(f"DEBUG TXT: Primeiros 200 chars: {txt_content[:200]}")

            if not txt_content.strip():
                print("ERRO TXT: Arquivo vazio")
                return {"success": False, "error": "Arquivo TXT vazio", "chunks_created": 0, "embeddings_created": 0}

            namespace = self.namespace_for(sindico_id, condo_id)
            print(f"DEBUG TXT: Usando namespace: {namespace}")

            # --- MODO NOVO (title/category) ---
            if (title or category) and (metadata is None or doc_id is None):
                print(f"DEBUG TXT: Usando MODO NOVO com title={title}, category={category}")
                chunks = self._create_chunks_tokenless(txt_content, chunk_size=1000)
                print(f"DEBUG TXT: Criados {len(chunks)} chunks")

                metadata_list: List[Dict[str, Any]] = []
                for i, _ in enumerate(chunks):
                    metadata_list.append({
                        "title": title or "Documento",
                        "category": category or "geral",
                        "sindico_id": sindico_id,
                        "condo_id": condo_id,
                        "type": "txt"
                    })

                return self.upsert_texts(
                    texts=chunks,
                    namespace=namespace,
                    metadata_list=metadata_list
                )

            # --- MODO ORIGINAL (doc_id + metadata) ---
            print(f"DEBUG TXT: Usando MODO ORIGINAL com doc_id={doc_id}")
            chunks = self._create_chunks(txt_content)
            print(f"DEBUG TXT: Criados {len(chunks)} chunks com overlap")

            embeddings = self._embed(chunks)

            if not doc_id:
                doc_id = f"doc_{int(datetime.now().timestamp())}"

            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vectors.append({
                    "id": f"{doc_id}_{i}",
                    "values": embedding,
                    "metadata": {
                        **(metadata or {}),
                        "text": chunk[:1000],
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "sindico_id": str(sindico_id),
                        "condo_id": str(condo_id)
                    }
                })

            print(f"DEBUG TXT: Fazendo upsert de {len(vectors)} vetores...")
            self.index.upsert(vectors=vectors, namespace=namespace)
            print(f"DEBUG TXT: Upsert concluído!")

            return {
                "success": True,
                "chunks_created": len(chunks),
                "embeddings_created": len(embeddings),
                "doc_id": doc_id
            }

        except Exception as e:
            logger.error(f"Erro em process_txt_content: {e}")
            print(f"ERRO TXT: {str(e)}")
            import traceback
            print(f"ERRO TXT TRACEBACK: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "chunks_created": 0,
                "embeddings_created": 0
            }

    # ---------------------- Busca ----------------------

    def query(
        self,
        query: str,
        sindico_id: int,
        condo_id: int,
        namespace: Optional[str] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """Busca semântica nos documentos"""
        try:
            if not namespace:
                namespace = self.namespace_for(sindico_id, condo_id)

            # ADICIONAR ESTES PRINTS DE DEBUG:
            print(f"DEBUG RAG: Query recebida = '{query}'")
            print(f"DEBUG RAG: Sindico ID = {sindico_id}, Condo ID = {condo_id}")
            print(f"DEBUG RAG: Namespace = {namespace}")

            # Criar embedding da query
            query_embedding = self._embed([query])[0]
            print("DEBUG RAG: Embedding criado com sucesso")

            # Buscar no Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                namespace=namespace,
                include_metadata=True
            )

            print(f"DEBUG RAG: Pinecone retornou {len(results.matches) if hasattr(results, 'matches') and results.matches is not None else 0} resultados")

            if not getattr(results, "matches", None):
                return {
                    "success": False,
                    "answer": "Não encontrei informações relevantes nos documentos.",
                    "sources": [],
                    "confidence": 0.0
                }

            relevant_chunks: List[str] = []
            sources = set()

            for match in results.matches:
                # limiar simples — mantenho o 0.7 do original
                if getattr(match, "score", 0) > 0.7:
                    md = getattr(match, "metadata", {}) or {}
                    relevant_chunks.append(md.get("text", ""))
                    sources.add(md.get("filename", md.get("title", "Documento")))

            if not relevant_chunks:
                return {
                    "success": False,
                    "answer": "Não encontrei informações suficientemente relevantes.",
                    "sources": [],
                    "confidence": 0.0
                }

            # Gerar resposta usando GPT
            context = "\n\n".join(relevant_chunks[:3])

            prompt = f"""Baseado nos seguintes trechos dos documentos do condomínio:

{context}

Pergunta: {query}

Responda de forma clara e direta, citando as informações dos documentos quando relevante."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado em gestão condominial. Responda baseado apenas nas informações fornecidas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            answer = response.choices[0].message.content

            return {
                "success": True,
                "answer": answer,
                "sources": list(sources),
                "confidence": results.matches[0].score if results.matches else 0.0,
                "chunks_used": len(relevant_chunks)
            }

        except Exception as e:
            logger.error(f"Erro na query: {e}")
            return {
                "success": False,
                "answer": f"Erro ao processar busca: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }


# ---------------------- Singleton ----------------------

_rag_instance: Optional[RAGSystem] = None

def get_or_create_rag() -> RAGSystem:
    """Retorna instância única do RAG System"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGSystem()
    return _rag_instance