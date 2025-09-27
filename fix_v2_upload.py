import re

with open('app/main.py', 'r') as f:
    content = f.read()

# Encontrar o bloco do Document no v2/upload
pattern = r'doc = Document\((.*?)\)'
match = re.search(pattern, content, re.DOTALL)

if match:
    # Substituir pelo código correto
    new_block = '''doc = Document(
                filename=file.filename,
                file_path=f"uploads/{file.filename}",
                condo_id=condominium_id,
                uploaded_by=current_user.id,
                file_size=len(content)
            )'''
    
    content = re.sub(pattern, new_block, content, count=1, flags=re.DOTALL)
    
    with open('app/main.py', 'w') as f:
        f.write(content)
    print("✅ Corrigido!")
