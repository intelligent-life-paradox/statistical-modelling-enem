import nbformat
import os
from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
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

def analisar_com_groq(nome_arquivo, conteudo):
    prompt = f"""
    Analise o Jupyter Notebook '{nome_arquivo}' sobre o ENEM.
    Resuma em português:
    1. Objetivo do modelo.
    2. Técnicas estatísticas/ML usadas.
    3. Insights principais.
    
    Conteúdo:
    {conteudo}
    """
    
    try:
        response = client.chat.completions.create(
            # tomando o llama pois ele é gratuito 
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Você é um Engenheiro de Dados especialista em ENEM."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro na análise: {e}"

#  notebooks
notebooks = [
    'notebooks/analise_exploratoria.ipynb',
    'notebooks/clusters.ipynb',
    'notebooks/modelagem_stat.ipynb',
    'notebooks/causal_tree.ipynb'
]

aviso = "> [!IMPORTANT]\n> Análise técnica gerada via **Groq (Llama 3.3)**. Baseada em amostra teste.\n\n"
relatorio_final = aviso + "#  Relatório de Inteligência Artificial - Modelagem ENEM\n\n"

for arquivo in notebooks:
    if os.path.exists(arquivo):
        print(f"Analisando com Groq: {arquivo}...")
        conteudo = extrair_conteudo_notebook(arquivo)
        analise = analisar_com_groq(arquivo, conteudo)
        relatorio_final += f"## Arquivo: {os.path.basename(arquivo)}\n\n{analise}\n\n---\n\n"

with open("relatorio_ia.md", "w", encoding="utf-8") as f:
    f.write(relatorio_final)