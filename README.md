# Modelagem Estatística — Renda como Força Causal no ENEM

> Estimativas causais do efeito da renda familiar e do capital cultural sobre o desempenho no ENEM · 2015–2019 · sample = 300.000 candidatos/ano para árvores causais | n = 2000/ano para modelo RLM

**[🔗 Ver resultados interativos](https://joaolucaanalysisproject.netlify.app/)**

---

## Sobre o projeto

Este projeto investiga os fatores que influenciam as notas dos candidatos do ENEM utilizando abordagens estatísticas avançadas. O foco central é **isolar o efeito causal da renda familiar** sobre o desempenho acadêmico, controlando por variáveis sociodemográficas e estruturais.


---

## 🎯 Objetivos de Modelagem

A análise é conduzida através de três pilares metodológicos:

1. **Modelos Estatísticos Clássicos** — Regressões lineares múltiplas (OLS e RLM com M-estimator de Huber) para identificar correlações e testar hipóteses sobre as variáveis que compõem o perfil do candidato.

2. **Double Machine Learning (DML)** — Estimação causal via EconML para isolar o efeito da renda controlando por confundidores observáveis, separando o que é correlação do que é efeito tratamento estimado.

3. **Causal Forest** — Algoritmo de aprendizado de máquina para estimar **efeitos de tratamento heterogêneos (CATEs)**, identificando quais perfis de candidatos respondem mais ou menos à renda como variável causal.

---

## 📊 Principais Resultados (série 2015–2019)

### Efeito causal da renda (DML)

| Ano  | ATE (σ) | ATE (pts ENEM) | IC 95%       |
|------|---------|----------------|--------------|
| 2015 | —       | ~19 pts        | —            |
| 2016 | —       | menor (recessão)| —           |
| 2017 | —       | retomada        | —            |
| 2018 | —       | crescente       | —            |
| 2019 | **0,289σ** | **≈ 23,3 pts** | ver app   |

- **Crescimento da série:** ATE aumentou ~**+21%** entre 2015 e 2019
- **Média histórica:** R$ 1.000 de renda familiar ≈ **+12 pontos no ENEM**

### Heterogeneidade por perfil (Causal Forest)

- A razão entre o CATE do grupo mais responsivo e o menos responsivo cresceu de **2,81×** (2015) para **3,86×** (2019)
- Candidatos sem internet, de escola pública não-federal e com baixo capital cultural concentram os maiores efeitos marginais da renda
- Para grupos com acesso já garantido a essas condições, o efeito marginal da renda estagna

### Regressão OLS (referência)

- Modelo globalmente significativo: **F = 339**, **p < 0,001**
- **R² = 0,337** (ajustado: 0,336) — poder explicativo relevante para variáveis socioeconômicas e demográficas

### Efeito do capital cultural

- ATE médio: **~0,117σ** na nota por +1σ de capital cultural ≈ **~9 pontos no ENEM**
- Em subgrupos de maior renda, o efeito pode chegar a **~16 pontos/σ** de capital cultural

---

## 🔬 Metodologia

### Dados

- **Fonte:** Microdados INEP — ENEM 2015 a 2019
- **Amostra:** ~300.000 candidatos por ano após filtragem
- **Tratamento de renda:** winsorização IQR para lidar com cauda longa e censura superior nos dados de renda declarada

### Variável de capital cultural (`score_cult_pais`)

Score composto e padronizado (μ=0, σ=1) construído a partir da escolaridade do pai e da mãe, operacionalizado segundo critérios da **ABEP**. Converte escolaridade em anos de formação equivalente e z-score na amostra de cada ano. Captura o *ambiente intelectual do domicílio*, não o poder de compra — permitindo separar os mecanismos causais da renda e do capital cultural no modelo DML.

### Stack técnico

| Componente          | Biblioteca                      |
|---------------------|---------------------------------|
| DML                 | `EconML`                        |
| Causal Forest       | `scikit-learn`                  |
| RLM Huber           | `statsmodels`                   |
| Pipeline de dados   | `pandas`, `numpy`               |
| App interativo      | Netlify (frontend estático)     |
| Versionamento       | `DVC` + Git                     |

---

## 📁 Estrutura do Repositório

```
statistical-modelling-enem/
│
├── apps/                        # Frontend do app de resultados (Netlify)
│
├── configs/                     # Configurações do projeto (parâmetros, paths)
│
├── dados/
│   └── microdados_enem_2019/    # Amostra de dados (tracked via DVC)
│
├── images/                      # Visualizações e gráficos exportados
│
├── legacy/                      # Versões anteriores e experimentos descartados
│
├── notebooks/                   # Análise exploratória e modelagem (Jupyter)
│
├── scripts/
│   └── enem_pipeline/           # Pipeline de ETL e estimação causal
│
├── .github/workflows/           # CI (GitHub Actions)
├── .dvcignore
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

---

## ⚠️ Limitações e Humildade Epistêmica

Os resultados são **condicionais aos dados observados** e pressupõem *unconfoundedness* — ausência de confundidores não-observados como motivação, saúde mental ou qualidade docente. Essa hipótese **não pode ser verificada** nos microdados do INEP.

Limitações técnicas específicas:

- **Winsorização IQR da renda** pode sub-estimar o efeito causal para candidatos de alta renda e introduz arbitrariedade no threshold escolhido. Os resultados são mais válidos para a faixa de renda baixa a média.
- **Correlação residual** entre renda e capital cultural pode inflar ambos os coeficientes mesmo após separação no DML.
- As hipóteses causais interpretativas são *plausíveis*, não conclusões estabelecidas — com dados observacionais, múltiplas explicações permanecem abertas.

> ⚠️ **Projeto em andamento.** Os valores podem ser refinados nas próximas iterações de modelagem e validação causal.

---

## Como executar

```bash
# Instalar dependências
pip install -r requirements.txt

# Ou com uv
uv sync

# Rodar pipeline
python scripts/enem_pipeline/run.py
```

Os notebooks em `notebooks/` documentam as etapas de análise exploratória, modelagem OLS, DML e Causal Forest com outputs intermediários.

Para saber mais sobre o Projeto, é essencial que você acesse o app acima!!!
---

*Microdados INEP · ENEM 2015–2019 · Double Machine Learning (EconML) · RLM statsmodels Huber M-estimator · Causal Forest (scikit-learn)*
