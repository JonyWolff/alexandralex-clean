# app/rag_system.py
import os
import re
import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pinecone import Pinecone
from openai import OpenAI
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import traceback

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RAGSystem:
    """Sistema RAG otimizado para documentos jurídicos/condominiais
    Com busca dupla: condomínio específico + base de conhecimento geral
    """

    def __init__(self):
        # Configurar OpenAI
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Configurar Pinecone (nova API)
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Conectar ao índice
        self.index_name = "alexandralex"
        self.index = self.pc.Index(self.index_name)
        
        # Modelo de embeddings
        self.embedding_model = "text-embedding-ada-002"

    @staticmethod
    def namespace_for(sindico_id: int, condo_id: int) -> str:
        """Gera namespace único por síndico/condomínio"""
        return f"user_{sindico_id}_cond_{condo_id}"

    # ---------------------- Helpers internos OTIMIZADOS ----------------------

    def _create_chunks(self, text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
        """Divide texto em chunks com overlap OTIMIZADO
        MODIFICADO: chunk_size=800, overlap=200 (era 600/75)
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start = end - overlap if end < len(text) else end

        return chunks

    def _create_chunks_tokenless(self, text: str, chunk_size: int = 800) -> List[str]:
        """Chunking simples por tamanho OTIMIZADO
        MODIFICADO: chunk_size=800 (era 600)
        """
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

    # ---------------------- Chunking Estruturado ----------------------
    
    def _create_structured_chunks(self, text: str, doc_type: str = "general") -> List[Dict[str, Any]]:
        """
        Cria chunks preservando estrutura legal/documental
        OTIMIZADO: chunks de 800-1200 caracteres
        """
        chunks_with_metadata = []
        
        # Padrões para detectar estruturas legais
        article_pattern = r'(?=(?:Art(?:igo)?\.?\s*\d+|CAPÍTULO\s+[IVXLCDM]+|SEÇÃO\s+[IVXLCDM]+|§\s*\d+º?))'
        
        if doc_type in ["convenção", "regimento", "estatuto"]:
            sections = re.split(article_pattern, text)
            
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                    
                article_match = re.match(r'Art(?:igo)?\.?\s*(\d+)', section)
                article_num = article_match.group(1) if article_match else None
                
                # Se a seção for muito grande, dividir em chunks de 800 chars
                if len(section) > 1200:
                    # Usar chunking otimizado para seções grandes
                    sub_chunks = self._create_chunks(section, chunk_size=800, overlap=200)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunks_with_metadata.append({
                            'text': sub_chunk,
                            'metadata': {
                                'chunk_type': 'paragraph',
                                'article_num': str(article_num) if article_num else "0",
                                'paragraph_num': j,
                                'doc_type': doc_type
                            }
                        })
                else:
                    chunks_with_metadata.append({
                        'text': section.strip(),
                        'metadata': {
                            'chunk_type': 'article',
                            'article_num': str(article_num) if article_num else "0",
                            'doc_type': doc_type
                        }
                    })
        else:
            # Para documentos gerais - usar chunking otimizado
            simple_chunks = self._create_chunks(text, chunk_size=800, overlap=200)
            for i, chunk in enumerate(simple_chunks):
                chunks_with_metadata.append({
                    'text': chunk,
                    'metadata': {
                        'chunk_type': 'mixed',
                        'doc_type': doc_type
                    }
                })
        
        logger.info(f"Criados {len(chunks_with_metadata)} chunks estruturados")
        return chunks_with_metadata

    def _detect_document_type(self, text: str, filename: str) -> str:
        """Detecta o tipo de documento para otimizar chunking"""
        text_lower = text.lower()[:1000]
        filename_lower = filename.lower()
        
        if any(term in text_lower or term in filename_lower for term in ['convenção', 'convencao']):
            return 'convenção'
        elif any(term in text_lower or term in filename_lower for term in ['regimento']):
            return 'regimento'
        elif any(term in text_lower or term in filename_lower for term in ['ata']):
            return 'ata'
        elif any(term in text_lower or term in filename_lower for term in ['estatuto']):
            return 'estatuto'
        elif any(term in text_lower or term in filename_lower for term in ['piscina']):
            return 'regras_piscina'
        elif any(term in text_lower or term in filename_lower for term in ['churrasqueira']):
            return 'regras_churrasqueira'
        elif any(term in text_lower or term in filename_lower for term in ['bicicletário', 'bicicletario']):
            return 'regras_bicicletario'
        else:
            return 'general'

    # ---------------------- Classificação e Expansão de Query ----------------------
    
    def _classify_query(self, query: str) -> Dict[str, Any]:
        """Classifica o tipo de query para otimizar busca"""
        query_lower = query.lower()
        
        if re.search(r'\bart(?:igo)?\.?\s*\d+|\blei\b|\bdecreto\b|\bnorma\b', query_lower):
            return {'type': 'lookup', 'boost_exact_match': True}
        elif re.search(r'\bposso\b|\bpode\b|\bpermitido\b|\bproibido\b|\bdeve\b', query_lower):
            return {'type': 'interpretative', 'boost_semantic': True}
        elif re.search(r'\bcompar\b|\bdiferenç\b|\bmelhor\b|\bpior\b', query_lower):
            return {'type': 'comparative', 'need_multiple_docs': True}
        elif re.search(r'\bcomo\b|\bpasso\s*a\s*passo\b|\bprocedimento\b|\bprocesso\b', query_lower):
            return {'type': 'procedural', 'preserve_order': True}
        elif re.search(r'\b\d{4}\b|\bvigente\b|\batual\b|\bantigo\b|\brecente\b', query_lower):
            return {'type': 'temporal', 'use_date_filter': True}
        else:
            return {'type': 'general', 'balanced': True}

    def _expand_query(self, query: str, classification: Dict) -> List[str]:
        """Expande query em múltiplas variações para melhor recall"""
        queries = [query]  # Original sempre incluída
        
        # Adicionar variações baseadas no tipo
        if classification['type'] == 'lookup':
            if 'artigo' not in query.lower():
                queries.append(f"artigo sobre {query}")
            queries.append(f"regulamento {query}")
        elif classification['type'] == 'interpretative':
            queries.append(f"regras sobre {query}")
            queries.append(f"normas de {query}")
            queries.append(f"regulamento {query}")
        
        # Sinônimos condominiais expandidos
        synonyms = {
            'animal': ['pet', 'bicho', 'cachorro', 'gato', 'animais de estimação'],
            'barulho': ['ruído', 'som', 'silêncio', 'ruido', 'barulho excessivo'],
            'festa': ['evento', 'comemoração', 'reunião', 'festividade'],
            'multa': ['penalidade', 'sanção', 'punição', 'advertência'],
            'piscina': ['área de lazer', 'área da piscina', 'piscinas'],
            'churrasqueira': ['área de churrasqueira', 'churrasqueiras', 'churrasco'],
            'bicicletário': ['bicicletario', 'bicicleta', 'bicicletas']
        }
        
        for term, syns in synonyms.items():
            if term in query.lower():
                for syn in syns[:3]:  # Até 3 sinônimos
                    queries.append(query.lower().replace(term, syn))
        
        return list(dict.fromkeys(queries))[:5]  # Max 5 queries únicas

    # ---------------------- Embed com Debug ----------------------

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

    # ---------------------- upsert_texts ----------------------

    def upsert_texts(
        self,
        texts: List[str],
        namespace: str,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        base_doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Helper para fazer upsert de textos com deduplicação"""
        try:
            print(f"DEBUG UPSERT: Iniciando upsert de {len(texts)} textos no namespace {namespace}")
            
            if not texts:
                return {"success": False, "error": "Sem textos para indexar", "chunks_created": 0, "embeddings_created": 0}

            # Verificar duplicatas
            unique_texts = []
            unique_metadatas = []
            
            for i, text in enumerate(texts):
                content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
                
                filename = "doc"
                if metadata_list and i < len(metadata_list):
                    filename = metadata_list[i].get('filename', metadata_list[i].get('title', 'doc'))
                
                check_id = f"{filename}_{content_hash}_0"
                
                try:
                    existing = self.index.fetch(ids=[check_id], namespace=namespace)
                    if not existing.vectors:
                        unique_texts.append(text)
                        if metadata_list and i < len(metadata_list):
                            unique_metadatas.append(metadata_list[i])
                    else:
                        print(f"DEBUG: Documento {filename} chunk {i} já existe, pulando...")
                except Exception as e:
                    print(f"DEBUG: Erro ao verificar duplicata: {e}")
                    unique_texts.append(text)
                    if metadata_list and i < len(metadata_list):
                        unique_metadatas.append(metadata_list[i])
            
            if not unique_texts:
                print("DEBUG: Todos os documentos já existem no índice")
                return {"success": True, "message": "Documentos já existem", "chunks_created": 0, "embeddings_created": 0}
            
            print(f"DEBUG UPSERT: {len(unique_texts)} de {len(texts)} são novos")

            # Embeddings apenas dos textos únicos
            embeddings = self._embed(unique_texts)

            # Preparar vetores
            if not base_doc_id:
                first_hash = hashlib.md5(unique_texts[0].encode()).hexdigest()[:8]
                base_doc_id = f"doc_{first_hash}_{int(datetime.now().timestamp())}"

            print(f"DEBUG UPSERT: Preparando {len(embeddings)} vetores com doc_id base: {base_doc_id}")

            vectors = []
            for i, (chunk, embedding) in enumerate(zip(unique_texts, embeddings)):
                md = (unique_metadatas[i] if (unique_metadatas and i < len(unique_metadatas)) else {}) or {}
                
                content_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                filename = md.get('filename', md.get('title', 'doc'))
                vector_id = f"{filename}_{content_hash}_{i}"
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        **md,
                        "text": chunk[:1000],
                        "full_text": chunk,  # IMPORTANTE: texto completo
                        "chunk_index": i,
                        "total_chunks": len(unique_texts)
                    }
                })

            # Upsert
            print(f"DEBUG UPSERT: Enviando {len(vectors)} vetores para o Pinecone...")
            self.index.upsert(vectors=vectors, namespace=namespace)
            print(f"DEBUG UPSERT: Upsert concluído com sucesso!")

            return {
                "success": True,
                "chunks_created": len(unique_texts),
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

    # ---------------------- Processamento PDF MELHORADO ----------------------

    def process_pdf_content(
        self,
        pdf_content: bytes,
        sindico_id: int,
        condo_id: int,
        doc_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        title: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Processa PDF com chunking otimizado"""
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

            # Detectar tipo de documento
            filename = metadata.get('filename', title or 'documento.pdf') if metadata else (title or 'documento.pdf')
            doc_type = self._detect_document_type(text, filename)
            print(f"DEBUG PDF: Tipo de documento detectado: {doc_type}")

            # MODO NOVO (title/category)
            if (title or category) and (metadata is None or doc_id is None):
                print(f"DEBUG PDF: Usando MODO NOVO com title={title}, category={category}")
                
                # Usar chunking estruturado quando disponível
                if doc_type in ['convenção', 'regimento', 'estatuto', 'regras_piscina', 'regras_churrasqueira', 'regras_bicicletario']:
                    chunks_data = self._create_structured_chunks(text, doc_type)
                    chunks = [cd['text'] for cd in chunks_data]
                else:
                    chunks = self._create_chunks(text, chunk_size=800, overlap=200)  # OTIMIZADO
                
                print(f"DEBUG PDF: Criados {len(chunks)} chunks")

                metadata_list: List[Dict[str, Any]] = []
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        "title": title or "Documento",
                        "filename": title or "documento.pdf",
                        "category": category or "geral",
                        "sindico_id": sindico_id,
                        "condo_id": condo_id,
                        "type": "pdf",
                        "doc_type": doc_type
                    }
                    
                    # Adicionar metadata estruturada se disponível
                    if doc_type in ['convenção', 'regimento', 'estatuto'] and i < len(chunks_data):
                        chunk_metadata.update(chunks_data[i]['metadata'])
                    
                    metadata_list.append(chunk_metadata)

                return self.upsert_texts(
                    texts=chunks,
                    namespace=namespace,
                    metadata_list=metadata_list
                )

            # MODO ORIGINAL (doc_id + metadata)
            print(f"DEBUG PDF: Usando MODO ORIGINAL com doc_id={doc_id}")
            
            # Usar chunking estruturado quando apropriado
            if doc_type in ['convenção', 'regimento', 'estatuto', 'regras_piscina', 'regras_churrasqueira', 'regras_bicicletario']:
                chunks_data = self._create_structured_chunks(text, doc_type)
                chunks = [cd['text'] for cd in chunks_data]
                chunks_metadata = [cd['metadata'] for cd in chunks_data]
            else:
                chunks = self._create_chunks(text, chunk_size=800, overlap=200)  # OTIMIZADO
                chunks_metadata = [{'chunk_type': 'mixed', 'doc_type': doc_type}] * len(chunks)
            
            print(f"DEBUG PDF: Criados {len(chunks)} chunks")

            # Verificação de duplicatas
            unique_chunks = []
            unique_indices = []
            unique_chunk_metadata = []
            
            if not doc_id:
                doc_id = f"doc_{int(datetime.now().timestamp())}"
            
            for i, (chunk, chunk_meta) in enumerate(zip(chunks, chunks_metadata)):
                content_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                check_id = f"{doc_id}_{content_hash}_{i}"
                
                try:
                    existing = self.index.fetch(ids=[check_id], namespace=namespace)
                    if not existing.vectors:
                        unique_chunks.append(chunk)
                        unique_indices.append(i)
                        unique_chunk_metadata.append(chunk_meta)
                    else:
                        print(f"DEBUG: Chunk {i} do documento já existe")
                except:
                    unique_chunks.append(chunk)
                    unique_indices.append(i)
                    unique_chunk_metadata.append(chunk_meta)
            
            if not unique_chunks:
                return {"success": True, "message": "Documento já existe", "chunks_created": 0, "embeddings_created": 0}

            # Embeddings apenas dos chunks únicos
            embeddings = self._embed(unique_chunks)

            vectors = []
            for idx, (chunk, embedding, orig_idx, chunk_meta) in enumerate(zip(
                unique_chunks, embeddings, unique_indices, unique_chunk_metadata
            )):
                content_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                vector_id = f"{doc_id}_{content_hash}_{orig_idx}"
                
                vector_metadata = {
                    **(metadata or {}),
                    **chunk_meta,
                    "text": chunk[:1000],
                    "full_text": chunk,  # IMPORTANTE: texto completo
                    "chunk_index": orig_idx,
                    "total_chunks": len(chunks),
                    "sindico_id": str(sindico_id),
                    "condo_id": str(condo_id),
                    "doc_type": doc_type,
                    "indexed_at": datetime.now().isoformat()
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": vector_metadata
                })

            # Upsert
            print(f"DEBUG PDF: Fazendo upsert de {len(vectors)} vetores...")
            batch_size = 50
            for i in range(0, len(vectors), batch_size):
                self.index.upsert(vectors=vectors[i:i+batch_size], namespace=namespace)
            print(f"DEBUG PDF: Upsert concluído!")

            return {
                "success": True,
                "chunks_created": len(unique_chunks),
                "embeddings_created": len(embeddings),
                "doc_id": doc_id,
                "doc_type": doc_type
            }

        except Exception as e:
            logger.error(f"Erro em process_pdf_content: {e}")
            print(f"ERRO PDF: {str(e)}")
            print(f"ERRO PDF TRACEBACK: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "chunks_created": 0,
                "embeddings_created": 0
            }

    # ---------------------- Process TXT ----------------------

    def process_txt_content(
        self,
        txt_content: str,
        sindico_id: int,
        condo_id: int,
        doc_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        title: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Processa TXT com chunking otimizado"""
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

            # Detectar tipo de documento
            filename = metadata.get('filename', title or 'documento.txt') if metadata else (title or 'documento.txt')
            doc_type = self._detect_document_type(txt_content, filename)
            print(f"DEBUG TXT: Tipo de documento detectado: {doc_type}")

            # MODO NOVO (title/category)
            if (title or category) and (metadata is None or doc_id is None):
                print(f"DEBUG TXT: Usando MODO NOVO com title={title}, category={category}")
                
                if doc_type in ['convenção', 'regimento', 'estatuto', 'regras_piscina', 'regras_churrasqueira', 'regras_bicicletario']:
                    chunks_data = self._create_structured_chunks(txt_content, doc_type)
                    chunks = [cd['text'] for cd in chunks_data]
                else:
                    chunks = self._create_chunks(txt_content, chunk_size=800, overlap=200)  # OTIMIZADO
                
                print(f"DEBUG TXT: Criados {len(chunks)} chunks")

                metadata_list: List[Dict[str, Any]] = []
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        "title": title or "Documento",
                        "filename": title or "documento.txt",
                        "category": category or "geral",
                        "sindico_id": sindico_id,
                        "condo_id": condo_id,
                        "type": "txt",
                        "doc_type": doc_type
                    }
                    
                    if doc_type in ['convenção', 'regimento', 'estatuto'] and i < len(chunks_data):
                        chunk_metadata.update(chunks_data[i]['metadata'])
                    
                    metadata_list.append(chunk_metadata)

                return self.upsert_texts(
                    texts=chunks,
                    namespace=namespace,
                    metadata_list=metadata_list
                )

            # MODO ORIGINAL
            print(f"DEBUG TXT: Usando MODO ORIGINAL com doc_id={doc_id}")
            
            if doc_type in ['convenção', 'regimento', 'estatuto', 'regras_piscina', 'regras_churrasqueira', 'regras_bicicletario']:
                chunks_data = self._create_structured_chunks(txt_content, doc_type)
                chunks = [cd['text'] for cd in chunks_data]
                chunks_metadata = [cd['metadata'] for cd in chunks_data]
            else:
                chunks = self._create_chunks(txt_content, chunk_size=800, overlap=200)  # OTIMIZADO
                chunks_metadata = [{'chunk_type': 'mixed', 'doc_type': doc_type}] * len(chunks)
            
            print(f"DEBUG TXT: Criados {len(chunks)} chunks")

            # Verificação de duplicatas
            unique_chunks = []
            unique_indices = []
            unique_chunk_metadata = []
            
            if not doc_id:
                doc_id = f"doc_{int(datetime.now().timestamp())}"
            
            for i, (chunk, chunk_meta) in enumerate(zip(chunks, chunks_metadata)):
                content_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                check_id = f"{doc_id}_{content_hash}_{i}"
                
                try:
                    existing = self.index.fetch(ids=[check_id], namespace=namespace)
                    if not existing.vectors:
                        unique_chunks.append(chunk)
                        unique_indices.append(i)
                        unique_chunk_metadata.append(chunk_meta)
                    else:
                        print(f"DEBUG: Chunk {i} do documento já existe")
                except:
                    unique_chunks.append(chunk)
                    unique_indices.append(i)
                    unique_chunk_metadata.append(chunk_meta)
            
            if not unique_chunks:
                return {"success": True, "message": "Documento já existe", "chunks_created": 0, "embeddings_created": 0}

            embeddings = self._embed(unique_chunks)

            vectors = []
            for idx, (chunk, embedding, orig_idx, chunk_meta) in enumerate(zip(
                unique_chunks, embeddings, unique_indices, unique_chunk_metadata
            )):
                content_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
                vector_id = f"{doc_id}_{content_hash}_{orig_idx}"
                
                vector_metadata = {
                    **(metadata or {}),
                    **chunk_meta,
                    "text": chunk[:1000],
                    "full_text": chunk,  # IMPORTANTE: texto completo
                    "chunk_index": orig_idx,
                    "total_chunks": len(chunks),
                    "sindico_id": str(sindico_id),
                    "condo_id": str(condo_id),
                    "doc_type": doc_type,
                    "indexed_at": datetime.now().isoformat()
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": vector_metadata
                })

            print(f"DEBUG TXT: Fazendo upsert de {len(vectors)} vetores...")
            batch_size = 50
            for i in range(0, len(vectors), batch_size):
                self.index.upsert(vectors=vectors[i:i+batch_size], namespace=namespace)
            print(f"DEBUG TXT: Upsert concluído!")

            return {
                "success": True,
                "chunks_created": len(unique_chunks),
                "embeddings_created": len(embeddings),
                "doc_id": doc_id,
                "doc_type": doc_type
            }

        except Exception as e:
            logger.error(f"Erro em process_txt_content: {e}")
            print(f"ERRO TXT: {str(e)}")
            print(f"ERRO TXT TRACEBACK: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "chunks_created": 0,
                "embeddings_created": 0
            }

    # ---------------------- Query MELHORADA COM BUSCA DUPLA ----------------------

    def query(
        self,
        query: str,
        sindico_id: int,
        condo_id: int,
        namespace: Optional[str] = None,
        k: int = 12  # MODIFICADO: de 30 para 12 por namespace (total 24)
    ) -> Dict[str, Any]:
        """
        Busca semântica MELHORADA com busca dupla:
        1. Namespace do condomínio específico
        2. Namespace da Base de Conhecimento (user_0_cond_0)
        """
        try:
            # Namespace do condomínio
            namespace_condo = namespace or self.namespace_for(sindico_id, condo_id)
            # Namespace da Base de Conhecimento Geral
            namespace_kb = "user_0_cond_0"

            print(f"DEBUG RAG: Query recebida = '{query}'")
            print(f"DEBUG RAG: Sindico ID = {sindico_id}, Condo ID = {condo_id}")
            print(f"DEBUG RAG: Namespace Condomínio = {namespace_condo}")
            print(f"DEBUG RAG: Namespace Base Conhecimento = {namespace_kb}")

            # Classificar query
            classification = self._classify_query(query)
            print(f"DEBUG RAG: Query classificada como: {classification['type']}")

            # Expandir query
            expanded_queries = self._expand_query(query, classification)
            print(f"DEBUG RAG: Query expandida para {len(expanded_queries)} variações")

            # BUSCA DUPLA: Condomínio + Base de Conhecimento
            all_matches = []
            seen_ids = set()
            
            for q in expanded_queries:
                # Criar embedding
                query_embedding = self._embed([q])[0]
                
                # 1. Buscar no namespace do condomínio
                if namespace_condo != namespace_kb:  # Evitar busca dupla se for o mesmo
                    results_condo = self.index.query(
                        vector=query_embedding,
                        top_k=k,
                        namespace=namespace_condo,
                        include_metadata=True
                    )
                    
                    for match in results_condo.matches:
                        if match.id not in seen_ids:
                            seen_ids.add(match.id)
                            # Adicionar boost para docs do condomínio (prioridade)
                            match.score = match.score * 1.1  
                            all_matches.append(match)
                
                # 2. Buscar na Base de Conhecimento Geral
                results_kb = self.index.query(
                    vector=query_embedding,
                    top_k=k,
                    namespace=namespace_kb,
                    include_metadata=True
                )
                
                for match in results_kb.matches:
                    if match.id not in seen_ids:
                        seen_ids.add(match.id)
                        all_matches.append(match)

            print(f"DEBUG RAG: Total de {len(all_matches)} matches únicos encontrados")
            print(f"DEBUG RAG: {len([m for m in all_matches if 'user_0_cond_0' in str(m)])} da Base de Conhecimento")

            if not all_matches:
                return {
                    "success": False,
                    "answer": "Não encontrei informações relevantes nos documentos.",
                    "sources": [],
                    "confidence": 0.0
                }

            # Filtrar por score com threshold adaptativo
            threshold = 0.35  # Threshold baixo para capturar mais informação
            filtered_matches = [m for m in all_matches if getattr(m, "score", 0) > threshold]
            print(f"DEBUG RAG: {len(filtered_matches)} matches após filtro de score {threshold}")

            if not filtered_matches:
                return {
                    "success": False,
                    "answer": "Não encontrei informações suficientemente relevantes.",
                    "sources": [],
                    "confidence": 0.0
                }

            # Ordenar por score
            filtered_matches.sort(key=lambda x: x.score, reverse=True)

            # Selecionar top chunks - AUMENTADO para 12
            top_matches = filtered_matches[:12]

            # Preparar contexto
            context_chunks = []
            sources = set()
            sources_detail = []
            
            for match in top_matches:
                md = getattr(match, "metadata", {}) or {}
                # Usar full_text se disponível, senão usar text
                chunk_text = md.get('full_text', md.get('text', ''))
                if chunk_text:
                    context_chunks.append(chunk_text)
                    source_name = md.get("filename", md.get("title", "Documento"))
                    sources.add(source_name)
                    
                    # Identificar origem do documento
                    if md.get('sindico_id') == '0' and md.get('condo_id') == '0':
                        sources_detail.append(f"{source_name} (Base de Conhecimento)")
                    else:
                        sources_detail.append(f"{source_name} (Condomínio)")

            # Gerar resposta com GPT
            context = "\n\n---\n\n".join(context_chunks)
            
            # Prompt baseado na classificação
            if classification['type'] == 'lookup':
                system_prompt = """Você é a Dra. Alexandra, especialista em direito condominial.
                Cite EXATAMENTE os artigos e textos relevantes dos documentos.
                Seja precisa e objetiva."""
            elif classification['type'] == 'interpretative':
                system_prompt = """Você é a Dra. Alexandra, especialista em direito condominial.
                Interprete as regras de forma clara e prática.
                Explique o que é permitido ou proibido com base nos documentos."""
            else:
                system_prompt = """Você é a Dra. Alexandra, especialista em direito condominial.
                Responda de forma profissional baseando-se nos documentos fornecidos."""

            # Instrução de resposta completa
            system_prompt += """
            
            IMPORTANTE:
            - Forneça uma resposta COMPLETA e DETALHADA com no mínimo 2000 caracteres
            - Use TODAS as informações relevantes do contexto
            - Cite artigos e trechos específicos dos regulamentos
            - Se houver múltiplas regras, mencione TODAS
            - Estruture a resposta com:
              * Introdução clara
              * Lista numerada ou com bullets das principais regras
              * Citação dos artigos relevantes
              * Conclusão ou orientação final
            - Use formatação markdown para melhor legibilidade
            """

            prompt = f"""Baseado nos seguintes documentos do condomínio e base de conhecimento geral:

{context}

Pergunta: {query}

Responda de forma completa e detalhada, incluindo TODAS as informações relevantes dos documentos.
Se houver regras específicas sobre o tema, liste todas elas."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=3000  # AUMENTADO para respostas completas
            )

            answer = response.choices[0].message.content

            print(f"DEBUG RAG: Resposta gerada com {len(answer)} caracteres")
            print(f"DEBUG RAG: Usou {len(context_chunks)} chunks de contexto")

            return {
                "success": True,
                "answer": answer,
                "sources": list(sources)[:5],
                "sources_detail": sources_detail[:5],  # NOVO: detalhes das fontes
                "confidence": filtered_matches[0].score if filtered_matches else 0.0,
                "chunks_used": len(context_chunks),
                "query_type": classification['type']
            }

        except Exception as e:
            logger.error(f"Erro na query: {e}")
            print(f"ERRO QUERY: {str(e)}")
            print(f"ERRO QUERY TRACEBACK: {traceback.format_exc()}")
            return {
                "success": False,
                "answer": f"Erro ao processar busca: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }

    # ---------------------- Delete ----------------------

    def delete_document(self, doc_id: str, sindico_id: int, condo_id: int) -> Dict[str, Any]:
        """Remove um documento do índice"""
        try:
            namespace = self.namespace_for(sindico_id, condo_id)
            
            # Listar todos os IDs relacionados ao documento
            ids_to_delete = []
            for i in range(100):  # Assumindo máximo de 100 chunks
                ids_to_delete.append(f"{doc_id}_{i}")
            
            # Deletar do Pinecone
            self.index.delete(ids=ids_to_delete, namespace=namespace)
            
            return {"success": True, "message": "Documento removido com sucesso"}
        
        except Exception as e:
            logger.error(f"Erro ao deletar documento: {e}")
            return {"success": False, "error": str(e)}


# ---------------------- Singleton ----------------------

_rag_instance: Optional[RAGSystem] = None

def get_or_create_rag() -> RAGSystem:
    """Retorna instância única do RAG System"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGSystem()
    return _rag_instance