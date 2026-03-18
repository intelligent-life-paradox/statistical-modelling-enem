# Modelagem Estatística — Renda como Força Causal no ENEM

> Estimativas causais do efeito da renda familiar e do capital cultural sobre o desempenho no ENEM · 2015–2019 · n = 300.000 candidatos/ano | Para o modelo de regressão linear tomamos um sample de n = 2000 candidatos/ano

**[🔗 Ver resultados interativos](https://joaolucaanalysisproject.netlify.app/)**

---

## Sobre o projeto

Este projeto investiga os fatores que influenciam as notas dos candidatos do ENEM utilizando abordagens estatísticas avançadas. O foco central é **isolar o efeito causal da renda familiar e do capital cultural** sobre o desempenho acadêmico, controlando por variáveis sociodemográficas e estruturais.

---

## 🎯 Objetivos de Modelagem

A análise é conduzida através de três pilares metodológicos:

1. **RLM (Huber M-estimator)** — Regressão robusta para identificar correlações e testar hipóteses sobre as variáveis que compõem o perfil do candidato, reduzindo a influência desproporcional da cauda superior da renda declarada.

2. **Double Machine Learning (DML)** — Estimação causal via EconML para isolar o efeito da renda controlando por confundidores observáveis, separando o que é correlação do que é efeito tratamento estimado.

3. **Causal Forest** — Algoritmo de aprendizado de máquina para estimar **efeitos de tratamento heterogêneos (CATEs)**, identificando quais perfis de candidatos respondem mais ou menos à renda como variável causal.

---
## 📐 Como interpretar o ATE

O **ATE (Average Treatment Effect)** é o efeito médio de um aumento de 1 desvio-padrão na renda familiar sobre a nota do ENEM, estimado via Double Machine Learning — ou seja, *controlando* por raça, sexo, acesso à internet, tipo de escola e capital cultural dos pais.

Em linguagem direta: se dois candidatos têm o mesmo perfil socioeconômico em tudo, exceto que um tem renda 1σ maior que o outro (~R$1.700 a mais, dependendo do ano), o DML estima que o candidato de maior renda tende a tirar em média **+19 a +23 pontos a mais** no ENEM.

### O que é o σ de renda

A renda no ENEM é declarada em **faixas fixas** (ex: "de R$1.497 a R$2.245"), não como valor contínuo. O modelo usa o ponto médio de cada faixa e padroniza. Um desvio-padrão de renda (~R$1.700) corresponde aproximadamente a **subir 2–3 faixas de renda** na parte central da distribuição — por exemplo, de R$1.500 para R$3.500.

### O que é a métrica por R$1.000

A métrica **R$1.000 → X pts** é uma re-escala do ATE para uma unidade mais intuitiva. Como a renda é discreta, essa re-escala é uma **interpolação linear** — uma ordem de magnitude, não o efeito exato de uma variação contínua de R$1.000. Use-a para comparações relativas (entre anos, entre grupos), não como previsão pontual.

### O que o ATE *não* diz

- Não é o efeito de *dar* R$1.000 a um candidato — é uma associação causal estimada em dados observacionais
- Pressupõe que não há confundidores não-observados (motivação, qualidade docente, saúde mental)
- É um efeito **médio** — o CATE por subgrupo (folhas da árvore causal) é mais informativo para intervenções direcionadas
---
## 📊 Principais Resultados (série 2015–2019)

### Efeito causal da renda — DML + Causal Forest

| Ano  | ATE (σ)       | Pts ENEM | IC 95% (pts)  | +R$1.000 → pts | σ_ENEM |
|------|---------------|----------|---------------|---------------|--------|
| 2015 | 0,263 ± 0,047 | +19,6    | [12,8 ; 28,4] | +11,8         | 74,6   |
| 2016 | 0,230 ± 0,049 | +17,1    | [9,9 ; 24,3]  | +10,3         | 74,3   |
| 2017 | 0,263 ± 0,056 | +19,5    | [11,4 ; 27,7] | +11,7         | 74,3   |
| 2018 | 0,272 ± 0,054 | +22,6    | [13,9 ; 31,4] | +10,2         | 83,0   |
| 2019 | 0,289 ± —     | +23,3    | —             | +12,0         | —      |

Todos os ATEs significativos a p < 0,001. Distribuição dos efeitos individuais (P10/P50/P90):

| Ano  | P10   | P50   | P90   |
|------|-------|-------|-------|
| 2015 | 0,149 | 0,241 | 0,420 |
| 2016 | 0,138 | 0,224 | 0,328 |
| 2017 | 0,134 | 0,264 | 0,392 |
| 2018 | 0,128 | 0,269 | 0,413 |

- **Crescimento da série:** ATE aumentou ~**+21%** entre 2015 e 2019
- **Média histórica:** +R$ 1.000 de renda familiar ≈ **+12 pontos no ENEM** — pode não parecer muito, mas faz muita diferença em cursos mais concorridos, por isso cotas por renda fazem tanto sentido.

### Heterogeneidade por perfil (Causal Forest)

- A razão entre o CATE do grupo mais responsivo e o menos responsivo cresceu de **2,81×** (2015) para **3,86×** (2019)
- Candidatos sem internet, de escola pública não-federal e com baixo capital cultural concentram os maiores efeitos marginais da renda
- Para grupos com acesso já garantido a essas condições, o efeito marginal da renda estagna

### RLM — Coeficientes padronizados (2015, referência)

| Variável           | β (σ)  | Pts ENEM | Sig. | Observação               |
|--------------------|--------|----------|------|--------------------------|
| Renda              | +0,208 | +15,5    | ***  | +3,9 pts/R$1k            |
| Capital cultural   | +0,167 | +12,4    | ***  | +3,0 pts/unidade         |
| Internet (sim)     | +0,168 | +12,5    | ***  | vs. sem internet         |
| Escola federal     | +0,649 | +48,4    | ***  | vs. pública (T=3)        |
| Raça (ref. branca) | —      | —        | n.s. | n.s. na maioria dos anos |
| Idade (spline)     | —      | —        | —    | controlada               |

O modelo RLM se verificou com significância alta (p < 0,001) para renda, capital cultural, tipo de escola e raça (especificamente negros e pardos) ao longo dos anos.

### Screenshots do app

![Heterogeneidade do efeito causal — folhas da árvore CATE](images/app_enem_01.png)
![Grupos mais vs menos afetados — padrão transversal](images/app_enem_02.png)

---

## 🔬 Metodologia

### Dados

- **Fonte:** Microdados INEP — ENEM 2015 a 2019
- **Amostra:** ~300.000 candidatos/ano após filtragem (DML + Causal Forest) · 2.000/ano (RLM)
- **Escala de renda:** z-score clássico para estimação · fator R$1k calculado via std robusto (IQR / 1,3489), imune à censura superior declaratória em R$25.425
- **Mea Culpa 1:** Eu não tomei dados pós-2019 devido a LGPD. Com efeito, pode-se dizer que a LGPD anonimizou os dados pós-2019, como eu não queria ter essa complicação desnecessária de ligar o questionário ao estudante (revertendo as técnicas de anonimização, o que não garante certeza da chave-primária), eu acabei não usando esses dados. Contudo, não é difícil tomar esses dados numa pesquisa real... é burocrático.
- **Mea Culpa 2:** Quantificar o efeito ao se passar de uma classe de renda para outra seria o mais "correto" tendo em vista como os microdados do enem se apresentam ( para um indivíduo numa classe, a renda dele sempre assume o ponto médio das cotas daquela classe). Porém, há duas escolhas deliberadas aqui:
          1. acho a interpretação +1000 de renda -> X pontos no enem mais intuitiva. Eu poderia facilmente sacrificá-la por uma variável mais         robusta construída como uma combinação de fatores socioeconomômicos que explicassem mais variância —via PCA, por exemplo —, porém, no trade-off entre interpretabilidade e métricas mais fortes, eu preferi interpretabilidade;
          2. O algorítmo de Causal Forest é robusto quanto a esse problema ( afinal, é um algorítmo de árvores) e a heterogeneidade é mais importante que a escala absoluta do problema — ou melhor, ela seria mais "invariante" à escala.

### Variável de capital cultural (`score_cult_pais`)

Score composto e padronizado (μ=0, σ=1) construído a partir da escolaridade do pai e da mãe, operacionalizado segundo critérios da **ABEP**. Captura o ambiente intelectual do domicílio, não o poder de compra — permitindo separar os mecanismos causais da renda e do capital cultural no modelo DML.

### Stack técnico

| Componente          | Biblioteca                  |
|---------------------|-----------------------------|
| DML + Causal Forest | `EconML`                    |
| RLM Huber           | `statsmodels`               |
| Pipeline de dados   | `pandas`, `numpy`           |
| App interativo      | Netlify (frontend) |
| CI/CD               | GitHub Actions              |
| Versionamento       | Git                 |

---

## 📁 Estrutura do Repositório

```
statistical-modelling-enem/
│
├── docs/
│   ├── app.html                         # Frontend do app interativo (Netlify)
│   └── metricas-{year}-enem/
│       ├── metricas_estatisticas.json   # Outputs do RLM por ano
│       └── metricas_causais.json        # Outputs do DML + Causal Forest por ano
│
├── configs/
│   ├── causal_trees.yml                 # parâmetros do DML + Causal Forest
│   ├── configs.yml                      # configurações gerais do projeto
│   └── statistical_tests.yml           # parâmetros dos testes
│
├── dados/
│   └── microdados_enem_2019/            # Amostra de dados (tracked via DVC)
│
├── images/                              # Visualizações exportadas dos notebooks
│
├── legacy/                              # Versões anteriores e experimentos descartados
│
├── notebooks/
│   ├── analise_exploratoria.ipynb
│   ├── causal_tree.ipynb
│   ├── clusters.ipynb
│   ├── modelagem_stat.ipynb
│   └── tratamentos_de_dados.ipynb
│
├── scripts/
│   └── enem_pipeline/
        ├── generate_data_js.py          #alimenta o frontend
│       ├── gcs_utils.py                 # Autenticação e I/O com Google Cloud Storage
│       ├── ingest_raw_enem.py           # Ingestão dos microdados brutos do INEP
│       ├── process_enem.py              # Limpeza, feature engineering e padronização
│       ├── run_causal_trees.py          # DML + Causal Forest via EconML
│       └── run_statistical_tests.py     #roda os testes estatísticos

  # RLM Huber e testes estatísticos
│
├── .github/workflows/                   # CI — GitHub Actions
├── .dvcignore
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

---

## ⚙️ Pipeline de Execução (CI + Google Cloud)

O pipeline é orquestrado via **GitHub Actions** e integrado ao **Google Cloud Storage (GCS)**. O fluxo completo:

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Actions Runner                   │
│                                                             │
│  1. Checkout do repositório                                 │
│  2. Carrega secrets (GCP_PROJECT_ID, GCS_BUCKET,           │
│     credenciais de service account)                         │
│  3. Autentica no Google Cloud via OIDC                      │
│                          │                                  │
│            ┌─────────────▼──────────────┐                  │
│            │  ingest_raw_enem.py        │                  │
│            │  Lê microdados do bucket   │                  │
│            │  GCS → memória do runner   │                  │
│            └─────────────┬──────────────┘                  │
│                          │                                  │
│            ┌─────────────▼──────────────┐                  │
│            │  process_enem.py           │                  │
│            │  Limpeza + feature eng.    │                  │
│            │  + std robusto IQR         │                  │
│            └──────┬──────────┬──────────┘                  │
│                   │          │                              │
│     ┌─────────────▼──┐  ┌───▼─────────────────┐           │
│     │ run_statistical │  │ run_causal_trees.py  │           │
│     │ _tests.py       │  │ DML + Causal Forest  │           │
│     │ RLM Huber       │  │ CATEs por subgrupo   │           │
│     └─────────────┬──┘  └───┬─────────────────┘           │
│                   └────┬────┘                               │
│                        │                                    │
│         ┌──────────────▼──────────────────┐                │
│         │  metricas_estatisticas.json      │                │
│         │  metricas_causais.json           │                │
│         └──────┬─────────────────┬────────┘                │
│                │                 │                          │
│    ┌───────────▼────┐   ┌────────▼──────────┐             │
│    │  Upload p/ GCS  │   │ GitHub Artifacts  │             │
│    └────────────────┘   └───────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

---
## 🚀 Deploy Contínuo (CD) 
[![Netlify Status](https://api.netlify.com/api/v1/badges/b67b77e0-4fc5-4f97-87f5-8163e11bd7d2/deploy-status)](https://app.netlify.com/projects/joaolucaanalysisproject/deploys)

O app interativo é atualizado automaticamente a cada execução do pipeline, sem nenhuma configuração manual de deploy.

### Como funciona

Ao final de cada run do pipeline, o script `scripts/enem_pipeline/generate_data_js.py` lê os JSONs de resultado de `docs/enem-metrics-{year}/` e gera um único arquivo `docs/data.js` — que é a fonte de dados do frontend. O Netlify monitora o repositório via webhook e, ao detectar o push com o `data.js` atualizado, redeploya o site automaticamente em ~30 segundos.
```
GitHub Actions
  → processa dados e salva JSONs em docs/enem-metrics-{year}/
  → generate_data_js.py lê todos os anos e gera docs/data.js
  → git commit + push
    → Netlify detecta o push
      → site atualizado automaticamente
```

Não há tokens de deploy, CLI do Netlify ou steps adicionais — o CD é uma consequência direta do push ao repositório.

### Desativar a regeneração do data.js

Por padrão, o `data.js` é regenerado em **todo push**, mesmo quando os JSONs já existem no GCS e as análises são puladas. Para desativar temporariamente — por exemplo, para fazer um push de código sem atualizar o app — comente o step no workflow:
```yaml
# - name: Generate data.js
#   run: python scripts/enem_pipeline/generate_data_js.py
```

Para desativar permanentemente para um ano específico, use o input `year` do `workflow_dispatch` — o pipeline roda só para aquele ano e o `data.js` é regenerado apenas com os dados desse ano. Use `force=true` para reprocessar um ano já existente no GCS.


## ⚠️ Limitações e Humildade Epistêmica

Os resultados pressupõem *unconfoundedness* — ausência de confundidores não-observados como motivação, saúde mental ou qualidade docente. Essa hipótese não pode ser verificada nos microdados do INEP.

- O std robusto (IQR) para o fator R$1k pode subestimar o efeito para candidatos de alta renda — os resultados são mais válidos para a faixa baixa a média da distribuição
- Correlação residual entre renda e capital cultural pode inflar ambos os coeficientes mesmo após separação no DML
- As hipóteses causais interpretativas são plausíveis, não conclusões estabelecidas

> ⚠️ **Projeto em andamento.** Os valores podem ser refinados nas próximas iterações.

---

## Como executar

```bash
# Instalar dependências
pip install -r requirements.txt

# Ou com uv
uv sync
se tiver os dados localmente você deve mudar isso de no workflow| você também pode armazenar esses dados na cloud, setar as permissões IAM
necessária e guardar as variáveis nos secrets do github onde sua pipeline vai rodar tão boa quanto a minha*.

# Rodar pipeline
python scripts/enem_pipeline/run.py
```


Os notebooks em `notebooks/` documentam as etapas de análise exploratória, modelagem RLM, DML e Causal Forest com outputs intermediários.

---
* FEATURE FUTURA DO PROJETO:
    pretendo embarcar todo o sistema para que você possa rodar via Docker (projeto em andamento)
*Microdados INEP · ENEM 2015–2019 · Double Machine Learning (EconML) · RLM statsmodels Huber M-estimator · Causal Forest (EconML)*
