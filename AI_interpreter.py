import nbformat
import os
import google.generativeai as genai


API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def extrair_conteudo_notebook(caminho_arquivo):
    """Lê o notebook e resume o conteúdo para a IA não ficar confusa com metadados."""
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    texto_resumo = []
    # Pegamos as primeiras 30 células para evitar estourar o limite de processamento
    for cell in nb.cells[:30]: 
        if cell.cell_type == 'markdown':
            
            texto_resumo.append(f"[EXPLICAÇÃO]: {cell.source}")
        elif cell.cell_type == 'code':
            
            texto_resumo.append(f"[CÓDIGO]: {cell.source}")
            
    return "\n".join(texto_resumo)

def analisar_com_ia(nome_arquivo, conteudo_limpo):
    """Envia o conteúdo para o Gemini e solicita a análise técnica."""
    prompt = f"""
    Você é um Engenheiro de Machine Learning sênior especializado em dados do ENEM.
    Analise o conteúdo do notebook '{nome_arquivo}' abaixo e forneça:
    
    1. **Objetivo**: Qual o propósito principal desta análise?
    2. **Técnicas**: Quais bibliotecas (ex: statsmodels, econml, sklearn) e modelos foram usados? Foque em explicar a lógica dos modelos.
    3. **Insights**: Com base no código e comentários, o que este modelo revela sobre os microdados do ENEM?
    

    CONTEÚDO DO NOTEBOOK:
    {conteudo_limpo}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erro ao analisar este notebook: {e}"


notebooks_acesso = [
    'notebooks/analise_exploratoria.ipynb',
    'notebooks/clusters.ipynb',
    'notebooks/modelagem_stat.ipynb',
    'notebooks/causal_tree.ipynb'
]


relatorio_final = """
> [!IMPORTANT]
> **AVISO DE AMBIENTE DE TESTE**: As análises e estatísticas abaixo são geradas automaticamente por uma IA baseando-se em uma **amostra reduzida** (sample) de dados. 
> Os resultados, coeficientes e métricas reais do projeto completo podem divergir significativamente.

#  Relatório de Inteligência Artificial - Modelagem ENEM\n\n
"""

for arquivo in notebooks_acesso:
    if os.path.exists(arquivo):
        print(f"Analisando: {arquivo} :)")
        conteudo = extrair_conteudo_notebook(arquivo)
        analise = analisar_com_ia(arquivo, conteudo)
        
        relatorio_final += f"## Arquivo: {os.path.basename(arquivo)}\n\n"
        relatorio_final += analise + "\n\n---\n\n"
    else:
        print(f"Arquivo não encontrado: {arquivo}")

#  Salva o resultado em Markdown para o GitHub Actions ler
with open("relatorio_ia.md", "w", encoding="utf-8") as f:
    f.write(relatorio_final)

print("Relatório gerado com sucesso em relatorio_ia.md!")