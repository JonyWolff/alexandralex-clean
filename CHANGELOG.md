# AlexandraLex - Changelog

## [2024-12-21] - Base de Conhecimento Implementada

### Adicionado
- Sistema completo de Base de Conhecimento Geral
- Upload de documentos legais (PDFs, TXT) para todos síndicos
- Interface de busca na Base de Conhecimento
- Threshold condicional (0.35 para Base, 0.7 para condomínios)

### Arquivos Modificados
- `app/knowledge_base.py` - Sistema de Base de Conhecimento
- `app/rag_system.py` - Ajuste de threshold condicional
- `app/main.py` - Integração do router de knowledge
- `static/dashboard.html` - Nova aba Base de Conhecimento

### Funcionalidades Testadas
- ✅ Upload de documentos da Base
- ✅ Busca retornando resultados corretos
- ✅ Listagem de documentos
- ✅ Integração com Pinecone namespace user_0_cond_0

### Pendente Verificação
- [ ] Upload de documentos dos condomínios ainda funciona
- [ ] Busca nos documentos dos condomínios mantém threshold 0.7
- [ ] Dra. Alexandra continua funcionando normalmente
