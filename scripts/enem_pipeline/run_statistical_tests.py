from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import yaml

from scripts.enem_pipeline.gcs_utils import download_file


def _has_min_levels(df: pd.DataFrame, col: str, min_levels: int = 2) -> bool:
    """Garante >= 2 níveis para evitar 'negative dimensions' no patsy."""
    n = df[col].dropna().nunique()
    if n < min_levels:
        print(f"[WARN] '{col}' tem {n} nível(is) único(s) — excluída da fórmula.")
    return n >= min_levels


def _build_ols_formula(df: pd.DataFrame) -> str | None:
    """
    Espelha o notebook (cell 13):
    MEDIA_CANDIDATO ~ RENDA + SCORE_CULT_PAIS
                    + bs(TP_FAIXA_ETARIA, df=5)
                    + C(TP_COR_RACA) + C(INTERNET) + C(TP_ESCOLA)
    """
    outcome = "MEDIA_CANDIDATO"
    if outcome not in df.columns:
        return None

    terms = []

    # Contínuas
    for c in ["RENDA", "SCORE_CULT_PAIS"]:
        if c in df.columns and df[c].notna().any():
            terms.append(c)

    # Spline em TP_FAIXA_ETARIA (se disponível)
    if "TP_FAIXA_ETARIA" in df.columns and df["TP_FAIXA_ETARIA"].notna().any():
        terms.append("bs(TP_FAIXA_ETARIA, df=5)")

    # Categóricas
    for c in ["TP_COR_RACA", "INTERNET", "TP_ESCOLA"]:
        if c in df.columns and _has_min_levels(df, c):
            terms.append(f"C({c})")

    if not terms:
        return None
    return f"{outcome} ~ {' + '.join(terms)}"


def _build_multilevel_formula(df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Espelha o notebook (cells 26-29):
    - modelo_2: grupos = SCORE_CULT_PAIS
    - modelo_3: grupos = RENDA
    """
    outcome = "MEDIA_CANDIDATO"
    if outcome not in df.columns:
        return []

    predictors  = ["RENDA", "SCORE_CULT_PAIS"]
    categorical = ["TP_COR_RACA", "INTERNET", "TP_ESCOLA"]

    cont_terms = [c for c in predictors if c in df.columns and df[c].notna().any()]
    cat_terms  = [f"C({c})" for c in categorical if c in df.columns and _has_min_levels(df, c)]
    all_terms  = cont_terms + cat_terms

    if not all_terms:
        return []

    formula = f"{outcome} ~ {' + '.join(all_terms)}"
    models  = []

    if "SCORE_CULT_PAIS" in df.columns and df["SCORE_CULT_PAIS"].notna().any():
        models.append((formula, "SCORE_CULT_PAIS"))

    if "RENDA" in df.columns and df["RENDA"].notna().any():
        models.append((formula, "RENDA"))

    return models


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    year                 = cfg["year"]
    sample_size_ols      = cfg.get("sample_size_ols", 2000)
    sample_size_multilevel = cfg.get("sample_size_multilevel", cfg.get("sample_size", 50000))
    seed                 = cfg.get("random_seed", 69)
    bucket               = cfg["gcs"]["bucket"]

    source_blob = f"processed/enem_{year}/dados_enem_processados_{year}.parquet"
    local_file  = Path("tmp") / f"dados_enem_processados_{year}.parquet"
    local_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Baixando gs://{bucket}/{source_blob}...")
    download_file(bucket, source_blob, local_file)

    df = pd.read_parquet(local_file)
    print(f"[INFO] Carregado: {len(df)} linhas | colunas: {list(df.columns)}")

    if len(df) == 0:
        raise RuntimeError(f"Parquet processed/enem_{year} está vazio.")

    # Amostragem estratificada por TP_COR_RACA (mantém proporções por raça)
    strat = "TP_COR_RACA" if "TP_COR_RACA" in df.columns else None

    def _sample(df: pd.DataFrame, target_size: int) -> pd.DataFrame:
        if len(df) <= target_size:
            return df
        if strat:
            return (
                df.groupby(strat, group_keys=False)
                .apply(lambda g: g.sample(
                    min(len(g), max(1, int(target_size * len(g) / len(df)))),
                    random_state=seed,
                ))
            )
        return df.sample(target_size, random_state=seed)

    df_ols = _sample(df, sample_size_ols)
    df_ml  = _sample(df, sample_size_multilevel)

    print(f"[INFO] Amostra para OLS: {len(df_ols)} linhas (target={sample_size_ols})")
    print(f"[INFO] Amostra para Multinível: {len(df_ml)} linhas (target={sample_size_multilevel})")

    # Normalização
    df_model_ols = df_ols.copy()
    df_model_ml  = df_ml.copy()
    scale_cols_ols = [c for c in ["RENDA", "SCORE_CULT_PAIS", "MEDIA_CANDIDATO"]
                       if c in df_model_ols.columns]
    scale_cols_ml  = [c for c in ["RENDA", "SCORE_CULT_PAIS", "MEDIA_CANDIDATO"]
                       if c in df_model_ml.columns]

    scaler = StandardScaler()
    if scale_cols_ols:
        df_model_ols[scale_cols_ols] = scaler.fit_transform(df_model_ols[scale_cols_ols])
    if scale_cols_ml:
        df_model_ml[scale_cols_ml] = scaler.fit_transform(df_model_ml[scale_cols_ml])

    results: dict = {
        "year": year,
        "sample_size_used": {
            "ols": int(len(df_model_ols)),
            "multilevel": int(len(df_model_ml)),
        },
    }

    # OLS 
    ols_formula = _build_ols_formula(df_model_ols)
    if ols_formula:
        print(f"[INFO] OLS: {ols_formula}")
        ols_model = smf.ols(ols_formula, data=df_model_ols).fit()
        results["ols"] = {
            "formula":      ols_formula,
            "r2":           float(ols_model.rsquared),
            "adj_r2":       float(ols_model.rsquared_adj),
            "f_stat":       float(ols_model.fvalue),
            "f_pvalue":     float(ols_model.f_pvalue),
            "coefficients": {k: float(v) for k, v in ols_model.params.to_dict().items()},
            "pvalues":      {k: float(v) for k, v in ols_model.pvalues.to_dict().items()},
        }
    else:
        print("[WARN] Colunas insuficientes para OLS.")
        results["ols"] = None

    #  Modelos Multiníveis 
    multilevel_results = []
    for formula, group_col in _build_multilevel_formula(df_model_ml):
        print(f"[INFO] Multinível: {formula} | grupos: {group_col}")
        try:
            ml = smf.mixedlm(formula, data=df_model_ml, groups=df_model_ml[group_col]).fit(reml=False)
            multilevel_results.append({
                "formula":        formula,
                "group_by":       group_col,
                "aic":            float(ml.aic),
                "bic":            float(ml.bic),
                "log_likelihood": float(ml.llf),
                "coefficients":   {k: float(v) for k, v in ml.params.to_dict().items()},
                "pvalues":        {k: float(v) for k, v in ml.pvalues.to_dict().items()},
            })
        except Exception as e:
            print(f"[WARN] Multinível com grupos={group_col} falhou: {e}")

    results["multilevel"] = multilevel_results if multilevel_results else None

    out_dir  = Path(cfg.get("output_dir", "results/statistical"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"statistical_tests_{year}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Resultados salvos em {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/statistical_tests.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()