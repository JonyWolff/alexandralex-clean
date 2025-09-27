import fileinput
import sys

# Procurar a linha que cria o PineconeVectorStore
for line in fileinput.input('app/rag_langchain.py', inplace=True):
    # Se encontrar a linha que cria o vectorstore
    if 'PineconeVectorStore(' in line and 'self._vectorstore_cache[namespace]' in line:
        # Substituir por versão que funciona
        print('            self._vectorstore_cache[namespace] = PineconeVectorStore(')
        print('                index_name="alexandralex",')
        print('                embedding=self.embeddings,')
        print('                namespace=namespace,')
        print('                text_key="text"')
        print('            )')
    elif 'index=self.index,' in line:
        # Pular esta linha (substituída acima)
        continue
    else:
        # Manter linha original
        sys.stdout.write(line)
