# DOCUMENTAÇÃO COMPLETA DO SISTEMA ALEXANDRALEX
## VERSÃO: 2.0 - STAGING (24/09/2025)

### 1. ARQUITETURA DO SISTEMA
- app/main.py: FastAPI principal
- app/rag_system.py: Sistema RAG com Pinecone (OTIMIZADO)
- app/alexandra.py: Dra. Alexandra com 3 personas
- app/knowledge_base.py: Upload/busca Base de Conhecimento
- app/upload.py: Upload docs condomínio
- app/auth.py: Autenticação JWT
- app/database.py: Conexão PostgreSQL
- app/models.py: Modelos SQLAlchemy
- static/dashboard.html: Interface síndicos
- static/knowledge_upload.html: Interface admin upload

### 2. CREDENCIAIS
PostgreSQL: postgresql://postgres:oPqFtipzRvZhbMDXevttbKtotikNofVIH@postgres.railway.internal:5432/railway
Pinecone: pcsk_6VFPHB_D2TwrH3vHwD4WPYNibbGAM7n8SsMhVHU897gegwzP9WqTf3TBER8EY1csN6X7AV
Index: alexandralex

### 3. URLS DO SISTEMA
Produção: https://alexandralex-staging-production.up.railway.app
Upload Admin: https://alexandralex-staging-production.up.railway.app/knowledge-upload
Admin emails: juliawolff6@gmail.com, jony.wolff@gmail.com

### 4. FUNCIONALIDADES
✅ Login JWT
✅ Upload PDFs com OCR
✅ Chunking 800/200
✅ Busca dupla (condomínio + base)
✅ Base de Conhecimento
✅ Dra. Alexandra 3 personas
✅ Anti-alucinação GPT

### 5. MELHORIAS APLICADAS (24/09)
- chunk_size = 800 (era 600)
- overlap = 200 (era 75)
- top_k = 12 (era 5)
- max_tokens = 3000
- Regras anti-alucinação no prompt

### 6. STATUS
Branch: develop (staging)
Último commit: fix anti-alucinação
Pronto para testes antes do merge para main
