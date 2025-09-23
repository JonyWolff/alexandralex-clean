# Teste simples com TXT
with open("piscina.txt", "r") as arquivo:
    texto = arquivo.read()
    print("CONTEÚDO DO ARQUIVO:")
    print("-" * 50)
    print(texto)
    print("-" * 50)
    print(f"Total de caracteres: {len(texto)}")
