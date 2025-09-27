# app/rag_langchain.py
"""
Sistema RAG refatorado com LangChain
Isolamento TOTAL entre:
1. Cada condomínio (namespaces separados)
2. Base de conhecimento geral
Nunca mistura contextos!
"""

import os
import hashlib
import logging
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
from dotenv import load_dotenv
import re
from io import BytesIO

# PDF processing
import PyPDF2
import pdfplumber

# LangChain - imports atualizados para versões instaladas
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Pinecone
from pinecone import Pinecone

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IsolatedLangChainRAG:
    """
    Sistema RAG com isolamento total usando LangChain
    Cada condomínio é completamente isolado
    Base de conhecimento é acessada separadamente
    """
    
    def __init__(self):
        """Inicializa sistema com isolamento garantido"""
        
        # Configurar OpenAI Embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-ada-002"
        )
        
        # Configurar Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "alexandralex"
        self.index = self.pc.Index(self.index_name)
        
        # LLM para geração
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=3000
        )
        
        # Text splitter para documentos
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=[
                "\n## ",    # Headers nível 2
                "\n### ",   # Headers nível 3  
                "\nArt. ",  # Artigos
                "\nArtigo ",
                "\n§ ",     # Parágrafos legais
                "\n\n",     # Parágrafos duplos
                "\n",       # Quebras de linha
                ". ",       # Sentenças
                "; ",       # Ponto e vírgula
                ", ",       # Vírgulas
                " "         # Espaços
            ]
        )
        
        # Cache de vectorstores por namespace (evita recriar)
        self._vectorstore_cache = {}
        
        # Prompts específicos para cada tipo de busca
        self.prompts = self._create_prompts()
        
        logger.info("✅ Sistema LangChain RAG inicializado com isolamento total")
    
    @staticmethod
    def namespace_for(sindico_id: int, condo_id: int) -> str:
        """Gera namespace único por síndico/condomínio"""
        return f"user_{sindico_id}_cond_{condo_id}"
    
    def _create_prompts(self) -> Dict[str, PromptTemplate]:
        """Cria prompts específicos para cada modo de busca"""
        
        prompts = {}
        
        # Prompt para busca APENAS no condomínio
        prompts['condo_only'] = PromptTemplate(
            input_variables=["context", "question"],
            template="""Você é a Dra. Alexandra, especialista em direito condominial.

CONTEXTO DOS DOCUMENTOS DO CONDOMÍNIO:
{context}

REGRAS ABSOLUTAS:
1. Use APENAS as informações fornecidas acima
2. Estas são as regras VÁLIDAS e APLICÁVEIS deste condomínio específico
3. Cite artigos EXATAMENTE como aparecem nos documentos
4. NÃO invente informações ou valores
5. Se não encontrar algo específico, diga "não especificado no regulamento"

PERGUNTA: {question}

RESPOSTA (mínimo 2000 caracteres, bem estruturada com citações exatas):"""
        )
        
        # Prompt para busca APENAS na base de conhecimento
        prompts['kb_only'] = PromptTemplate(
            input_variables=["context", "question"],
            template="""Você é a Dra. Alexandra, especialista em direito condominial.

CONTEXTO DA BASE DE CONHECIMENTO GERAL (modelos e referências):
{context}

AVISOS IMPORTANTES:
1. Estas são orientações GERAIS e modelos de referência
2. NÃO são regras específicas de nenhum condomínio
3. O usuário deve verificar o regulamento específico do seu condomínio
4. Use termos como "geralmente", "é comum", "modelos sugerem"

PERGUNTA: {question}

RESPOSTA (deixe claro que são orientações gerais):"""
        )
        
        # Prompt para busca híbrida COM SEPARAÇÃO
        prompts['hybrid'] = PromptTemplate(
            input_variables=["context_condo", "context_kb", "question"],
            template="""Você é a Dra. Alexandra, especialista em direito condominial.

DOCUMENTOS DO CONDOMÍNIO (REGRAS VÁLIDAS):
{context_condo}

---

BASE DE CONHECIMENTO (APENAS REFERÊNCIA):
{context_kb}

HIERARQUIA OBRIGATÓRIA:
1. SEMPRE priorize os DOCUMENTOS DO CONDOMÍNIO
2. Use BASE DE CONHECIMENTO apenas para complementar
3. Se houver conflito, prevalecem SEMPRE as regras do CONDOMÍNIO
4. Identifique claramente a origem de cada informação

PERGUNTA: {question}

RESPOSTA (cite claramente a origem de cada informação):"""
        )
        
        return prompts
    
    def _get_or_create_vectorstore(self, namespace: str) -> PineconeVectorStore:
        """
        Obtém ou cria um vectorstore para o namespace
        Usa cache para evitar recriar
        """
        if namespace not in self._vectorstore_cache:
            self._vectorstore_cache[namespace] = PineconeVectorStore(
                index_name="alexandralex",
                embedding=self.embeddings,
                text_key="text",
                namespace=namespace
            )
            logger.info(f"📦 Vectorstore criado para namespace: {namespace}")
        
        return self._vectorstore_cache[namespace]
    
    def _detect_document_type(self, text: str, filename: str = "") -> str:
        """Detecta tipo de documento para chunking otimizado"""
        text_lower = text.lower()[:1000]
        filename_lower = filename.lower()
        
        patterns = {
            'convenção': ['convenção', 'convencao'],
            'regimento': ['regimento'],
            'ata': ['ata'],
            'estatuto': ['estatuto'],
            'regras_piscina': ['piscina'],
            'regras_churrasqueira': ['churrasqueira'],
            'regras_bicicletario': ['bicicletário', 'bicicletario']
        }
        
        for doc_type, terms in patterns.items():
            if any(term in text_lower or term in filename_lower for term in terms):
                return doc_type
        
        return 'general'
    
    def process_pdf(
        self,
        pdf_content: bytes,
        sindico_id: int,
        condo_id: int,
        title: str = "Documento",
        category: str = "geral"
    ) -> Dict[str, Any]:
        """
        Processa PDF e armazena no namespace correto
        ISOLAMENTO: Cada condomínio tem seu namespace
        """
        try:
            namespace = self.namespace_for(sindico_id, condo_id)
            logger.info(f"📄 Processando PDF para namespace: {namespace}")
            
            # Extrair texto com pdfplumber (melhor que PyPDF2)
            text = ""
            try:
                import pdfplumber
                with pdfplumber.open(BytesIO(pdf_content)) as pdf:
                    for i, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n[Página {i+1}]\n{page_text}"
            except:
                # Fallback para PyPDF2
                pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Página {i+1}]\n{page_text}"
            
            if not text.strip():
                return {"success": False, "error": "PDF vazio ou sem texto"}
            
            # Detectar tipo de documento
            doc_type = self._detect_document_type(text, title)
            logger.info(f"📋 Tipo de documento: {doc_type}")
            
            # Criar documentos com metadata completa
            documents = []
            chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "title": title,
                        "category": category,
                        "sindico_id": sindico_id,
                        "condo_id": condo_id,
                        "doc_type": doc_type,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "source": "pdf",
                        "indexed_at": datetime.now().isoformat()
                    }
                )
                documents.append(doc)
            
            # Adicionar ao vectorstore ISOLADO do condomínio
            vectorstore = self._get_or_create_vectorstore(namespace)
            vectorstore.add_documents(documents)
            
            logger.info(f"✅ {len(documents)} chunks adicionados ao namespace {namespace}")
            
            return {
                "success": True,
                "chunks_created": len(documents),
                "namespace": namespace,
                "doc_type": doc_type
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar PDF: {e}")
            return {"success": False, "error": str(e)}
    
    def process_txt(
        self,
        txt_content: str,
        sindico_id: int,
        condo_id: int,
        title: str = "Documento",
        category: str = "geral"
    ) -> Dict[str, Any]:
        """Processa TXT com isolamento por namespace"""
        try:
            namespace = self.namespace_for(sindico_id, condo_id)
            logger.info(f"📝 Processando TXT para namespace: {namespace}")
            
            if isinstance(txt_content, bytes):
                txt_content = txt_content.decode("utf-8", errors="ignore")
            
            if not txt_content.strip():
                return {"success": False, "error": "Arquivo vazio"}
            
            doc_type = self._detect_document_type(txt_content, title)
            
            # Criar documentos
            documents = []
            chunks = self.text_splitter.split_text(txt_content)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "title": title,
                        "category": category,
                        "sindico_id": sindico_id,
                        "condo_id": condo_id,
                        "doc_type": doc_type,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "source": "txt",
                        "indexed_at": datetime.now().isoformat()
                    }
                )
                documents.append(doc)
            
            # Adicionar ao vectorstore ISOLADO
            vectorstore = self._get_or_create_vectorstore(namespace)
            vectorstore.add_documents(documents)
            
            logger.info(f"✅ {len(documents)} chunks adicionados ao namespace {namespace}")
            
            return {
                "success": True,
                "chunks_created": len(documents),
                "namespace": namespace,
                "doc_type": doc_type
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar TXT: {e}")
            return {"success": False, "error": str(e)}
    
    def query(
        self,
        question: str,
        sindico_id: int,
        condo_id: int,
        search_mode: Literal["condo_only", "kb_only", "hybrid"] = "condo_only",
        k: int = 10
    ) -> Dict[str, Any]:
        """Query com busca direta no Pinecone (solução que funciona)"""
        try:
            namespace = self.namespace_for(sindico_id, condo_id)
            logger.info(f"🔍 Query em namespace: {namespace}")
            
            # Criar embedding da pergunta
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=question
            )
            query_vec = response.data[0].embedding
            
            # Buscar no Pinecone
            results = self.index.query(
                vector=query_vec,
                top_k=k,
                namespace=namespace,
                include_metadata=True
            )
            
            # Montar contexto
            context_parts = []
            sources = []
            for match in results['matches']:
                if 'text' in match.get('metadata', {}):
                    context_parts.append(match['metadata']['text'])
                    sources.append({
                        'title': match['metadata'].get('title', 'Documento'),
                        'type': match['metadata'].get('doc_type', 'general')
                    })
            
            context = "\n\n".join(context_parts)
            
            if not context:
                answer = "Não encontrei informações sobre isso nos documentos do condomínio."
            else:
                # Gerar resposta com GPT
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Você é a Dra. Alexandra, especialista em direito condominial. Responda baseando-se APENAS no contexto fornecido."},
                        {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {question}"}
                    ],
                    temperature=0.3
                )
                answer = completion.choices[0].message.content
            
            return {
                "success": True,
                "answer": f"✅ **Fonte: Documentos do Condomínio**\n\n{answer}",
                "sources": sources[:5],
                "search_mode": search_mode,
                "namespace_used": namespace
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na query: {e}")
            return {
                "success": False,
                "answer": f"Erro ao processar: {str(e)}",
                "sources": []
            }
    
    def add_to_knowledge_base(
        self,
        content: str,
        title: str,
        category: str = "referência"
    ) -> Dict[str, Any]:
        """
        Adiciona documento à Base de Conhecimento Geral
        SEMPRE usa namespace user_0_cond_0
        """
        try:
            namespace_kb = "user_0_cond_0"
            logger.info(f"📚 Adicionando à Base de Conhecimento: {title}")
            
            # Criar documentos
            documents = []
            chunks = self.text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "title": title,
                        "category": category,
                        "sindico_id": 0,
                        "condo_id": 0,
                        "doc_type": self._detect_document_type(content, title),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "source": "knowledge_base",
                        "indexed_at": datetime.now().isoformat()
                    }
                )
                documents.append(doc)
            
            # Adicionar ao namespace da base geral
            vectorstore = self._get_or_create_vectorstore(namespace_kb)
            vectorstore.add_documents(documents)
            
            logger.info(f"✅ {len(documents)} chunks adicionados à Base de Conhecimento")
            
            return {
                "success": True,
                "chunks_created": len(documents),
                "namespace": namespace_kb
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao adicionar à base: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_document(self, doc_id: str, sindico_id: int, condo_id: int) -> Dict[str, Any]:
        """Remove documento do namespace específico"""
        try:
            namespace = self.namespace_for(sindico_id, condo_id)
            
            # LangChain não tem delete direto, precisamos usar Pinecone diretamente
            ids_to_delete = [f"{doc_id}_{i}" for i in range(100)]
            self.index.delete(ids=ids_to_delete, namespace=namespace)
            
            logger.info(f"🗑️ Documento {doc_id} removido de {namespace}")
            
            return {"success": True, "message": "Documento removido"}
            
        except Exception as e:
            logger.error(f"❌ Erro ao deletar: {e}")
            return {"success": False, "error": str(e)}


# ---------------------- Singleton ----------------------

_langchain_instance: Optional[IsolatedLangChainRAG] = None

def get_or_create_langchain_rag() -> IsolatedLangChainRAG:
    """Retorna instância única do sistema LangChain RAG"""
    global _langchain_instance
    if _langchain_instance is None:
        _langchain_instance = IsolatedLangChainRAG()
    return _langchain_instance


# ---------------------- Funções de Migração ----------------------

def migrate_from_old_system():
    """
    Função auxiliar para migrar do sistema antigo
    Pode ser usada para testes ou migração gradual
    """
    from app.rag_system import get_or_create_rag as get_old_rag
    
    old_rag = get_old_rag()
    new_rag = get_or_create_langchain_rag()
    
    logger.info("🔄 Iniciando migração do sistema antigo...")
    
    # Aqui você pode adicionar lógica de migração se necessário
    # Por enquanto, ambos sistemas podem coexistir
    
    return new_rag


if __name__ == "__main__":
    # Teste básico
    rag = get_or_create_langchain_rag()
    
    # Teste de upload
    test_text = """
    Regras da Piscina do Condomínio Teste
    
    Art. 1 - Horário de funcionamento: 8h às 22h
    Art. 2 - Proibido uso de garrafas de vidro
    Art. 3 - Crianças devem estar acompanhadas
    """
    
    result = rag.process_txt(
        test_text,
        sindico_id=1,
        condo_id=1,
        title="Regras Piscina",
        category="regras"
    )
    print(f"Upload teste: {result}")
    
    # Teste de query isolada
    result = rag.query(
        "Qual o horário da piscina?",
        sindico_id=1,
        condo_id=1,
        search_mode="condo_only"
    )
    print(f"Query teste: {result['answer'][:200]}...")