# Modelagem Estatística - Impactos no Desempenho do ENEM

Este projeto tem a pretensão de investigar os fatores que influenciam as notas dos candidatos do ENEM utilizando abordagens estatísticas avançadas. O foco central é isolar o impacto de diversas variáveis no desempenho acadêmico, controlando por fatores determinantes como **renda** e nível socioeconômico.

## 🎯 Objetivos de Modelagem

A análise será conduzida através de três pilares metodológicos:

1.  **Modelos Estatísticos Clássicos:** Aplicação de regressões lineares múltiplas para identificar correlações e testar hipóteses sobre as variáveis que compõem o perfil do candidato.
2.  **Modelos Multiníveis (Hierárquicos):** Utilizados para lidar com a estrutura aninhada dos dados, permitindo separar a variação da nota atribuível ao indivíduo da variação atribuível ao contexto (escola, estado, capital cultural...).
3.  **Causal Trees (Inferência Causal):** Emprego de algoritmos de aprendizado de máquina para estimar efeitos de tratamento heterogêneos, buscando entender como diferentes perfis de renda reagem a variáveis de controle específicas e identificar relações de causalidade mais robustas.


## 📌 Resultados mais sólidos até agora (projeto em andamento)

Com base nas saídas já obtidas (estatística descritiva, OLS e inferência causal), estes são os achados mais robustos nesta fase:

- **Média de desempenho (MEDIA_CANDIDATO):** aproximadamente **529,17 pontos**, com **IC de 99% entre 525,40 e 532,95**.
- **Índice de consumo (SCORE_CONSUMO):** média de **15,21**, com **IC de 99% entre 14,77 e 15,65**.
- **Regressão OLS:** modelo globalmente significativo (**F = 339**, **p < 0,001**), com **R² = 0,337** (ajustado 0,336), indicando poder explicativo relevante para variáveis socioeconômicas e demográficas.
- **Efeito causal médio (ATE) do capital cultural:** cerca de **0,117 desvios-padrão** na nota por +1 desvio-padrão de capital cultural; com desvio-padrão da nota em torno de 80 pontos, isso equivale a aproximadamente **9 pontos** no ENEM.
- **Heterogeneidade por perfil socioeconômico:** os ganhos não são homogêneos; em subgrupos de maior renda, os efeitos podem ser maiores, chegando a magnitudes próximas de **16 pontos por desvio-padrão** de capital cultural em recortes específicos.

A leitura teórica desses resultados é inspirada nas teses de **Pierre Bourdieu** sobre capital cultural: recursos culturais acumulados pelas famílias tendem a se converter em melhor desempenho escolar.

> ⚠️ Este é um **projeto em andamento**. Os valores podem ser refinados nas próximas iterações de modelagem e validação causal.
