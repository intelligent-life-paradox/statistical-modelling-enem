import nbformat
import os
import sys
from openai import OpenAI


api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERRO: A variável GROQ_API_KEY não foi encontrada!")
    print("Verifique se o nome no YAML e nos Secrets do GitHub é o mesmo.")
    sys.exit(1) 

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

def extrair_conteudo_notebook(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    texto_resumo = []
    for cell in nb.cells[:35]: 
        if cell.cell_type == 'markdown':
            texto_resumo.append(f"[TEXTO]: {cell.source[:300]}")
        elif cell.cell_type == 'code':
            texto_resumo.append(f"[CÓDIGO]: {cell.source[:400]}")
            
    return "\n".join(texto_resumo)

def analisar_com_ia(nome_arquivo, conteudo):
    prompt = f"Analise o notebook '{nome_arquivo}' sobre o ENEM. Resuma objetivo, técnicas e insights.\nConteúdo:\n{conteudo}"
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Você é um especialista em Ciência de Dados."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro na análise: {e}"

notebooks = [
    'notebooks/analise_exploratoria.ipynb',
    'notebooks/clusters.ipynb',
    'notebooks/modelagem_stat.ipynb',
    'notebooks/causal_tree.ipynb'
]

relatorio_final = "#  Relatório de IA - ENEM\n\n"

for arquivo in notebooks:
    if os.path.exists(arquivo):
        print(f"Processando: {arquivo}")
        conteudo = extrair_conteudo_notebook(arquivo)
        analise = analisar_com_ia(arquivo, conteudo)
        relatorio_final += f"## {os.path.basename(arquivo)}\n\n{analise}\n\n---\n\n"

with open("relatorio_ia.md", "w", encoding="utf-8") as f:
    f.write(relatorio_final)