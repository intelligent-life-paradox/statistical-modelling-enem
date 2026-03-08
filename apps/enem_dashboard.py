from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="ENEM Analytics Dashboard", layout="wide")

PROCESSED_CANDIDATE_DIRS = [Path("data/processed"), Path("dados/microdados_enem_2019")]


def _processed_files() -> list[Path]:
    files: list[Path] = []
    for base in PROCESSED_CANDIDATE_DIRS:
        if base.exists():
            files.extend(base.glob("dados_enem_processados_*.parquet"))
    return files


def available_years() -> list[int]:
    years = set()
    for p in _processed_files():
        try:
            years.add(int(p.stem.split("_")[-1]))
        except ValueError:
            pass
    if not years:
        years.add(2019)
    return sorted(years)


def load_data(year: int) -> pd.DataFrame:
    for base in PROCESSED_CANDIDATE_DIRS:
        parquet_path = base / f"dados_enem_processados_{year}.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)

    # fallback para facilitar demo local sem pasta data/
    fallback = Path("dados/microdados_enem_2019/sample_dados_enem_processados.csv")
    if fallback.exists():
        return pd.read_csv(fallback)

    raise FileNotFoundError("Nenhum dataset processado encontrado (nem fallback disponível).")


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def generate_narrative(year: int, df: pd.DataFrame, stats: dict, causal: dict) -> str:
    media = float(df["MEDIA_CANDIDATO"].mean()) if "MEDIA_CANDIDATO" in df.columns else float("nan")
    std = float(df["MEDIA_CANDIDATO"].std()) if "MEDIA_CANDIDATO" in df.columns else float("nan")

    ols = stats.get("ols", {})
    r2 = ols.get("r2")
    f_pvalue = ols.get("f_pvalue")

    effects = {e.get("treatment"): e for e in causal.get("effects", [])}
    renda_ate = effects.get("RENDA", {}).get("ate")
    cult_ate = effects.get("SCORE_CULT_PAIS", {}).get("ate")

    txt = [
        f"### Descobertas automáticas para {year}",
        f"- A média da nota (`MEDIA_CANDIDATO`) foi **{media:.2f}** com desvio-padrão de **{std:.2f}**.",
    ]

    if r2 is not None:
        txt.append(f"- No modelo OLS, o poder explicativo foi **R²={r2:.3f}**.")
    if f_pvalue is not None:
        txt.append(f"- A significância global do OLS (p-valor do F-test) foi **{f_pvalue:.4g}**.")

    if renda_ate is not None:
        txt.append(f"- Na árvore/forest causal, o ATE estimado para **RENDA** foi **{renda_ate:.4f}**.")
    if cult_ate is not None:
        txt.append(f"- Na árvore/forest causal, o ATE estimado para **SCORE_CULT_PAIS** foi **{cult_ate:.4f}**.")

    txt.append("- **Leitura recomendada:** compare os efeitos causais entre anos para observar estabilidade dos sinais e magnitude.")
    return "\n".join(txt)


st.title("📊 ENEM - Painel de Análises por Ano")
st.caption("Dashboard em Streamlit com gráficos em Plotly e narrativa automática por ano.")

years = available_years()
year = st.selectbox("Escolha o ano", years, index=len(years) - 1)

df = load_data(year)

stats = load_json(Path(f"results/statistical/statistical_tests_{year}.json"))
causal = load_json(Path(f"results/causal/causal_effects_{year}.json"))

col1, col2, col3 = st.columns(3)
col1.metric("N observações", f"{len(df):,}")
if "MEDIA_CANDIDATO" in df.columns:
    col2.metric("Média da nota", f"{df['MEDIA_CANDIDATO'].mean():.2f}")
if "SCORE_CULT_PAIS" in df.columns:
    col3.metric("Média capital cultural", f"{df['SCORE_CULT_PAIS'].mean():.2f}")

st.markdown("## Visualizações")

if "MEDIA_CANDIDATO" in df.columns:
    fig_hist = px.histogram(df, x="MEDIA_CANDIDATO", nbins=40, title="Distribuição da Média do Candidato")
    st.plotly_chart(fig_hist, use_container_width=True)

if {"SCORE_CULT_PAIS", "MEDIA_CANDIDATO"}.issubset(df.columns):
    color_col = "RENDA" if "RENDA" in df.columns else None
    fig_scatter = px.scatter(
        df,
        x="SCORE_CULT_PAIS",
        y="MEDIA_CANDIDATO",
        color=color_col,
        title="Capital Cultural vs Nota",
        opacity=0.5,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

if {"TP_SEXO", "MEDIA_CANDIDATO"}.issubset(df.columns):
    fig_box = px.box(df, x="TP_SEXO", y="MEDIA_CANDIDATO", title="Distribuição da nota por sexo")
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown(generate_narrative(year, df, stats, causal))

with st.expander("Ver dados amostrados"):
    st.dataframe(df.head(50), use_container_width=True)