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
    MEDIA_CANDIDATO ~ SCORE_CONSUMO + RENDA + SCORE_CULT_PAIS
                    + bs(TP_FAIXA_ETARIA, df=5)
                    + C(TP_COR_RACA) + C(INTERNET) + C(TP_ESCOLA)
    """
    outcome = "MEDIA_CANDIDATO"
    if outcome not in df.columns:
        return None

    terms = []

    # Contínuas
    for c in ["SCORE_CONSUMO", "RENDA", "SCORE_CULT_PAIS"]:
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
    - modelo_2: grupos = SG_UF_ESC
    - modelo_3: grupos = SCORE_CONSUMO
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

    if "SG_UF_ESC" in df.columns and df["SG_UF_ESC"].notna().any():
        models.append((formula, "SG_UF_ESC"))

    if "SCORE_CONSUMO" in df.columns and df["SCORE_CONSUMO"].notna().any():
        models.append((formula, "SCORE_CONSUMO"))

    return models


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    year        = cfg["year"]
    sample_size = cfg.get("sample_size", 10_000)
    seed        = cfg.get("random_seed", 69)
    bucket      = cfg["gcs"]["bucket"]

    source_blob = f"processed/enem_{year}/dados_enem_processados_{year}.parquet"
    local_file  = Path("tmp") / f"dados_enem_processados_{year}.parquet"
    local_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Baixando gs://{bucket}/{source_blob}...")
    download_file(bucket, source_blob, local_file)

    df = pd.read_parquet(local_file)
    print(f"[INFO] Carregado: {len(df)} linhas | colunas: {list(df.columns)}")

    if len(df) == 0:
        raise RuntimeError(f"Parquet processed/enem_{year} está vazio.")

    #  Amostragem estratificada por TP_COR_RACA 
    if len(df) > sample_size:
        strat = "TP_COR_RACA" if "TP_COR_RACA" in df.columns else None
        if strat:
            df = (
                df.groupby(strat, group_keys=False)
                .apply(lambda g: g.sample(
                    min(len(g), max(1, int(sample_size * len(g) / len(df)))),
                    random_state=seed,
                ))
            )
        else:
            df = df.sample(sample_size, random_state=seed)

    print(f"[INFO] Amostra final: {len(df)} linhas")

    # Normalização
    df_model = df.copy()
    scale_cols = [c for c in ["RENDA", "SCORE_CULT_PAIS", "SCORE_CONSUMO", "MEDIA_CANDIDATO"]
                  if c in df_model.columns]
    scaler = StandardScaler()
    df_model[scale_cols] = scaler.fit_transform(df_model[scale_cols])

    results: dict = {"year": year, "sample_size_used": int(len(df_model))}

    # OLS 
    ols_formula = _build_ols_formula(df_model)
    if ols_formula:
        print(f"[INFO] OLS: {ols_formula}")
        ols_model = smf.ols(ols_formula, data=df_model).fit()
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
    for formula, group_col in _build_multilevel_formula(df_model):
        print(f"[INFO] Multinível: {formula} | grupos: {group_col}")
        try:
            ml = smf.mixedlm(formula, data=df_model, groups=df_model[group_col]).fit(reml=False)
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