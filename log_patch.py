import fileinput
import sys

# Arquivos para adicionar logs
files = ['app/rag_langchain.py', 'app/rag_system.py']

# Adicionar imports no início
for filename in files:
    with open(filename, 'r') as f:
        content = f.read()
    
    if 'logger = logging.getLogger' not in content:
        # Adicionar após os imports
        content = content.replace('import os\n', 'import os\nimport logging\nlogging.basicConfig(level=logging.DEBUG)\nlogger = logging.getLogger(__name__)\n')
    
    # Adicionar logs nos métodos query
    content = content.replace(
        'def query(',
        'def query('
    )
    
    # Adicionar log no início do query
    content = content.replace(
        'namespace = self.namespace_for(sindico_id, condo_id)',
        'namespace = self.namespace_for(sindico_id, condo_id)\n        logger.info(f"RASTREAMENTO Query: namespace={namespace}, user={sindico_id}, condo={condo_id}")'
    )
    
    with open(filename, 'w') as f:
        f.write(content)

print("Logs adicionados!")
