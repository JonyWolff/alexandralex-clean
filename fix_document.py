import sys

# Ler o arquivo
with open('app/main.py', 'r') as f:
    lines = f.readlines()

# Encontrar e substituir o bloco problemático
for i in range(len(lines)):
    if 'doc = Document(' in lines[i]:
        # Substituir as próximas linhas até o fechamento do parêntese
        j = i
        while j < len(lines) and ')' not in lines[j]:
            j += 1
        
        # Substituir pelo código correto
        new_block = '''            doc = Document(
                filename=file.filename,
                file_path=f"uploads/{file.filename}",
                condo_id=condominium_id,
                uploaded_by=current_user.id,
                file_size=len(content)
            )
'''
        lines[i:j+1] = [new_block]
        break

# Salvar
with open('app/main.py', 'w') as f:
    f.writelines(lines)

print("Correção aplicada!")
