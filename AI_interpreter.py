import nbformat
import os
from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def extrair_conteudo_notebook(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    texto_resumo = []
    for cell in nb.cells[:40]: 
        if cell.cell_type == 'markdown':
            texto_resumo.append(f"[TEXTO]: {cell.source[:300]}")
        elif cell.cell_type == 'code':
            texto_resumo.append(f"[CÓDIGO]: {cell.source[:400]}")
            
    return "\n".join(texto_resumo)

def analisar_com_deepseek(nome_arquivo, conteudo):
    prompt = f"""
    Analise o Jupyter Notebook '{nome_arquivo}' sobre o ENEM.
    Resuma:
    1. Objetivo.
    2. Técnicas (statsmodels, econml, etc).
    3. Insights dos dados.
    
    Conteúdo:
    {conteudo}
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Você é um especialista em Ciência de Dados."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro na análise: {e}"

# notebooks
notebooks = [
    'notebooks/analise_exploratoria.ipynb',
    'notebooks/clusters.ipynb',
    'notebooks/modelagem_stat.ipynb',
    'notebooks/causal_tree.ipynb'
]

aviso = "> [!IMPORTANT]\n> Análise gerada por IA (DeepSeek) baseada em amostra reduzida.\n\n"
relatorio_final = aviso + "#  Relatório DeepSeek - Modelagem ENEM\n\n"

for arquivo in notebooks:
    if os.path.exists(arquivo):
        print(f"Analisando com DeepSeek: {arquivo}...")
        conteudo = extrair_conteudo_notebook(arquivo)
        analise = analisar_com_deepseek(arquivo, conteudo)
        relatorio_final += f"## Arquivo: {os.path.basename(arquivo)}\n\n{analise}\n\n---\n\n"

with open("relatorio_ia.md", "w", encoding="utf-8") as f:
    f.write(relatorio_final)