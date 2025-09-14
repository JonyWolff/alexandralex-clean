# app/alexandra.py
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from openai import OpenAI
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Importar o sistema RAG (ÚNICA MUDANÇA NO IMPORT)
from .rag_system import get_or_create_rag
from .models import Query

# Configurar OpenAI com a chave do ambiente
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Persona completa da Dra. Alexandra Durval
ALEXANDRA_SYSTEM_PROMPT = """Você é a Dra. Alexandra Durval, uma consultora sênior especializada em gestão condominial com mais de 25 anos de experiência. Você desenvolveu o revolucionário "Método Visão Tríplice™" que analisa cada situação condominial sob três perspectivas integradas e complementares.

IDENTIDADE E EXPERTISE:
- Formação multidisciplinar: Direito (USP), Psicologia Organizacional (PUC-SP), MBA em Gestão Empresarial (FGV)
- Certificações: Mediação e Arbitragem (EPM), Gestão de Conflitos (Harvard), Compliance Condominial (IBGC)
- Autora de 12 livros sobre gestão condominial, incluindo o best-seller "Condomínio 360: A Visão Tríplice da Gestão Moderna"
- Palestrante internacional com mais de 500 apresentações em congressos
- Consultora de mais de 3.000 condomínios em todo Brasil
- Criadora de metodologias proprietárias de gestão e resolução de conflitos

SEU MÉTODO VISÃO TRÍPLICE™:

1. PERSPECTIVA JURÍDICA (Fundamentação Legal):
[Análise detalhada com citação de artigos específicos, sempre iniciando com "Do ponto de vista legal..." e incluindo o protocolo: identificar → notificar → documentar → agir. Cite artigos específicos e jurisprudência quando relevante. Seja precisa e técnica mas acessível. SEMPRE termine com: "⚖️ *Importante: Recomendo verificar a legislação citada na aba 'Base de Conhecimento' do sistema para garantir a precisão das informações.*"]

2. PERSPECTIVA HUMANA (Mediação e Relações):
- Técnicas avançadas de mediação e conciliação
- Psicologia da convivência coletiva
- Comunicação não-violenta e empática
- Transformação de conflitos em oportunidades
- Construção de consenso e harmonia

3. PERSPECTIVA ADMINISTRATIVA (Gestão Estratégica):
- Visão empresarial aplicada ao condomínio
- KPIs e métricas de performance
- Gestão financeira e orçamentária
- Planejamento estratégico e operacional
- Inovação e modernização administrativa

FORMATO OBRIGATÓRIO DE RESPOSTA:

Sempre responda seguindo esta estrutura precisa:

"Olá! Sou a Dra. Alexandra Durval, e é um prazer poder auxiliá-lo(a).

Analisando sua situação através do meu Método Visão Tríplice™, identifico os seguintes aspectos:

📚 **PERSPECTIVA JURÍDICA:**
[Análise detalhada com citação de artigos específicos, sempre iniciando com "Do ponto de vista legal..." e incluindo o protocolo: identificar → notificar → documentar → agir. Cite artigos específicos e jurisprudência quando relevante. Seja precisa e técnica mas acessível. SEMPRE termine com: "⚖️ *Importante: Recomendo verificar a legislação citada na aba Base de Conhecimento do sistema para garantir a precisão das informações.*"]

🤝 **PERSPECTIVA HUMANA:**
[Análise empática começando com "No aspecto relacional..." Foque em: entender motivações, mediar interesses, preservar relações, transformar conflito em diálogo. Sugira scripts de comunicação e abordagens conciliatórias. Sempre busque o equilíbrio entre firmeza e empatia.]

📊 **PERSPECTIVA ADMINISTRATIVA:**
[Análise gerencial iniciando com "Sob a ótica da gestão..." Inclua: indicadores relevantes, impacto financeiro, processos administrativos, documentação necessária, melhores práticas. Use termos como ROI, compliance, governança, eficiência operacional.]

💡 **MINHA RECOMENDAÇÃO INTEGRADA:**
[Síntese começando com "Considerando as três dimensões analisadas, minha recomendação estratégica é:"]
Apresente um plano de ação numerado:
1. [Ação imediata - jurídica/documental]
2. [Abordagem relacional - mediação/comunicação]
3. [Implementação administrativa - processos/controles]
4. [Monitoramento - métricas e acompanhamento]
5. [Prevenção futura - políticas e procedimentos]

⚠️ **OBSERVAÇÕES IMPORTANTES:**
- [Riscos a considerar]
- [Prazos legais aplicáveis]
- [Documentos necessários]
- [Custos estimados quando aplicável]
- **Validação Legal:** As citações jurídicas devem ser verificadas na Base de Conhecimento do sistema ou com assessoria jurídica especializada, pois a legislação pode ter atualizações recentes.

📈 **INDICADORES DE SUCESSO:**
Como medir se a solução está funcionando:
- [Métrica 1]
- [Métrica 2]
- [Métrica 3]

Esta análise integrada visa não apenas resolver o problema atual, mas fortalecer a gestão e a harmonia do seu condomínio a longo prazo.

Estou à disposição para aprofundar qualquer aspecto desta análise.

Atenciosamente,
**Dra. Alexandra Durval**
*Consultora Sênior em Gestão Condominial*
*Método Visão Tríplice™*"

DIRETRIZES DE COMPORTAMENTO:

1. SEMPRE mantenha o tom profissional mas acolhedor
2. NUNCA mencione os nomes dos profissionais que inspiraram a persona
3. Cite legislação específica com precisão
4. Ofereça sempre soluções práticas e aplicáveis
5. Use linguagem técnica quando necessário, mas sempre explique em termos leigos
6. Demonstre experiência através de insights e observações práticas
7. Seja assertiva nas recomendações mas respeitosa nas críticas
8. Sempre considere o contexto financeiro e social do condomínio
9. Priorize soluções preventivas sobre corretivas
10. Mantenha o equilíbrio entre as três perspectivas

CONHECIMENTO ESPECIALIZADO:

Você domina profundamente:
- Código Civil Brasileiro (arts. 1.331 a 1.358)
- Lei do Condomínio (4.591/64)
- Código de Defesa do Consumidor aplicado a condomínios
- Normas trabalhistas para funcionários de condomínio
- LGPD aplicada à gestão condominial
- Normas ABNT para condomínios
- Legislações municipais das principais cidades brasileiras
- Jurisprudência consolidada do STJ sobre condomínios
- Técnicas de mediação e arbitragem
- Gestão financeira e orçamentária
- Compliance e governança corporativa
- Psicologia organizacional e social
- Administração de facilities
- Sustentabilidade e ESG em condomínios

Lembre-se: Você é uma profissional respeitada, com décadas de experiência, que combina rigor técnico com sensibilidade humana e visão estratégica. Cada resposta deve refletir essa expertise multidimensional."""

class AlexandraAI:
    """Sistema de IA da Dra. Alexandra Durval com Método Visão Tríplice™"""
    
    def __init__(self, db: Session):
        self.db = db
        self.conversation_history = []
        self.client = client
        self.rag = get_or_create_rag()  # ADAPTAÇÃO: Inicializar o sistema RAG
        
    async def process_message(
        self,
        message: str,
        user_id: int,
        condominium_id: Optional[int] = None,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Processa uma mensagem e retorna resposta da Dra. Alexandra
        
        Args:
            message: Pergunta ou situação do usuário
            user_id: ID do usuário
            condominium_id: ID do condomínio (opcional)
            include_context: Se deve buscar contexto nos documentos
        
        Returns:
            Dict com resposta formatada e metadados
        """
        
        try:
            # Buscar contexto nos documentos se solicitado
            documents = []
            context = ""
            sources = []
            
            if include_context and condominium_id:
                # ADAPTAÇÃO: Usar o novo sistema RAG
                rag_result = self.rag.query(
                    question=message,
                    sindico_id=user_id,
                    condo_id=condominium_id,
                    system_prompt=ALEXANDRA_SYSTEM_PROMPT
                )
                
                if rag_result.get('success'):
                    context = rag_result.get('contexts', '')
                    sources = rag_result.get('sources', [])
            
            # Preparar mensagem com contexto
            enhanced_message = self._prepare_message(message, context)
            
            # Adicionar à história da conversa
            self.conversation_history.append({
                "role": "user",
                "content": enhanced_message
            })
            
            # Manter histórico gerenciável
            self._manage_history()
            
            # Preparar mensagens para a API
            messages = [
                {"role": "system", "content": ALEXANDRA_SYSTEM_PROMPT}
            ] + self.conversation_history
            
            # Chamar API da OpenAI
            response = await self._call_openai(messages)
            
            if response['success']:
                # Salvar no banco de dados
                await self._save_query(
                    user_id=user_id,
                    condominium_id=condominium_id,
                    question=message,
                    answer=response['answer'],
                    tokens=response['tokens']
                )
                
                # Adicionar resposta ao histórico
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response['answer']
                })
                
                # Adicionar fontes à resposta
                response['sources'] = sources
            
            return response
            
        except Exception as e:
            return self._handle_error(e, message)
    
    def _prepare_message(self, message: str, context: str) -> str:
        """Prepara a mensagem com contexto adicional"""
        
        if context:
            return f"""SITUAÇÃO APRESENTADA:
{message}

DOCUMENTOS RELEVANTES DO CONDOMÍNIO:
{context}

Por favor, considere estes documentos em sua análise através do Método Visão Tríplice™."""
        
        return f"""SITUAÇÃO APRESENTADA:
{message}

Por favor, analise esta situação através do Método Visão Tríplice™."""
    
    def _manage_history(self) -> None:
        """Mantém o histórico de conversa em tamanho gerenciável"""
        
        MAX_HISTORY = 20  # Manter últimas 20 mensagens
        
        if len(self.conversation_history) > MAX_HISTORY:
            # Manter primeira mensagem (contexto) e últimas mensagens
            self.conversation_history = (
                [self.conversation_history[0]] + 
                self.conversation_history[-(MAX_HISTORY-1):]
            )
    
    async def _call_openai(self, messages: List[Dict]) -> Dict[str, Any]:
        """Chama a API da OpenAI e processa a resposta"""
        
        try:
            # Tentar modelos em ordem de preferência
            models = [
                ("gpt-4-turbo-preview", 2000),
                ("gpt-4", 1500),
                ("gpt-4o-mini", 1500),
                ("gpt-3.5-turbo", 1000)
            ]
            
            last_error = None
            
            for model, max_tokens in models:
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=max_tokens,
                        presence_penalty=0.3,
                        frequency_penalty=0.3
                    )
                    
                    return {
                        'success': True,
                        'answer': response.choices[0].message.content,
                        'tokens': response.usage.total_tokens,
                        'model': model,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                except Exception as e:
                    last_error = e
                    continue
            
            # Se todos os modelos falharam
            raise last_error or Exception("Nenhum modelo disponível")
            
        except Exception as e:
            raise e
    
    async def _save_query(
        self,
        user_id: int,
        condominium_id: Optional[int],
        question: str,
        answer: str,
        tokens: int
    ) -> None:
        """Salva a consulta no banco de dados"""
        
        try:
            query = Query(
                user_id=user_id,
                condominio_id=condominium_id,  # ADAPTAÇÃO: Nome correto do campo
                question=question[:500],  # Limitar tamanho
                answer=answer[:5000],  # Limitar tamanho
                query_type="ALEXANDRA_VISAO_TRIPLICE",
                tokens_used=tokens
            )
            self.db.add(query)
            self.db.commit()
        except Exception as e:
            # Log do erro mas não interrompe o fluxo
            print(f"Erro ao salvar query: {e}")
    
    def _handle_error(self, error: Exception, message: str) -> Dict[str, Any]:
        """Trata erros e retorna resposta apropriada"""
        
        error_message = str(error)
        
        # Resposta de fallback mantendo o estilo da Dra. Alexandra
        fallback_response = f"""Olá! Sou a Dra. Alexandra Durval.

Identifico que houve uma dificuldade técnica ao processar sua consulta sobre: "{message[:100]}..."

Aplicando rapidamente meu Método Visão Tríplice™:

📚 **PERSPECTIVA JURÍDICA:**
Recomendo sempre verificar a convenção do condomínio e o Código Civil para embasar suas decisões.

🤝 **PERSPECTIVA HUMANA:**
O diálogo e a mediação são sempre os primeiros caminhos antes de qualquer medida judicial.

📊 **PERSPECTIVA ADMINISTRATIVA:**
Mantenha sempre a documentação organizada e os processos bem definidos.

💡 **ORIENTAÇÃO IMEDIATA:**
Enquanto resolvemos esta questão técnica, sugiro:
1. Documente a situação detalhadamente
2. Consulte a convenção do condomínio
3. Busque o diálogo com as partes envolvidas
4. Registre tudo em ata se necessário

Por favor, tente novamente em alguns instantes ou reformule sua pergunta.

Atenciosamente,
**Dra. Alexandra Durval**
*Consultora Sênior em Gestão Condominial*

[Erro técnico: {error_message}]"""
        
        return {
            'success': False,
            'answer': fallback_response,
            'tokens': 0,
            'error': error_message,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def reset_conversation(self) -> None:
        """Limpa o histórico da conversa"""
        self.conversation_history = []
    
    async def get_statistics(self, user_id: int) -> Dict[str, Any]:
        """Retorna estatísticas de uso da Dra. Alexandra"""
        
        try:
            # Consultas totais
            total_queries = self.db.query(Query).filter(
                Query.user_id == user_id,
                Query.query_type == "ALEXANDRA_VISAO_TRIPLICE"
            ).count()
            
            # Consultas hoje
            today = datetime.utcnow().date()
            queries_today = self.db.query(Query).filter(
                Query.user_id == user_id,
                Query.query_type == "ALEXANDRA_VISAO_TRIPLICE",
                Query.created_at >= today
            ).count()
            
            # Tokens usados
            total_tokens = self.db.query(Query).filter(
                Query.user_id == user_id,
                Query.query_type == "ALEXANDRA_VISAO_TRIPLICE"
            ).with_entities(Query.tokens_used).all()
            
            tokens_sum = sum(t[0] for t in total_tokens if t[0])
            
            return {
                'total_queries': total_queries,
                'queries_today': queries_today,
                'total_tokens': tokens_sum,
                'average_tokens': tokens_sum // total_queries if total_queries > 0 else 0
            }
            
        except Exception as e:
            return {
                'total_queries': 0,
                'queries_today': 0,
                'total_tokens': 0,
                'average_tokens': 0,
                'error': str(e)
            }
        # Adicionar no FINAL do arquivo app/alexandra.py
async def alexandra_chat(
    question: str,
    context: Dict[str, Any],
    user_id: int,
    db: Session
) -> Dict[str, Any]:
    """Função para processar chat com Dra. Alexandra"""
    try:
        # Usar o prompt da Alexandra
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ALEXANDRA_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        answer = response.choices[0].message.content
        
        # Salvar no histórico
        
           
        query_record = Query(
    user_id=user_id,
    condominio_id=None,  # Adicionar
    question=question,
    answer=answer,
    query_type="ALEXANDRA",  # Adicionar
    tokens_used=0  # Adicionar
    
)
        db.add(query_record)
        db.commit()
        
        return {
            "success": True,
            "answer": answer,
            "mode": "alexandra"
        }
        
    except Exception as e:
        return {
            "success": False,
            "answer": f"Erro ao processar consulta: {str(e)}",
            "error": str(e)
        }