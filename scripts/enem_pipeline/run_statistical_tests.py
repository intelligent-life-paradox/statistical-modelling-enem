from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml

from scripts.enem_pipeline.gcs_utils import download_file


def _has_min_levels(df: pd.DataFrame, col: str, min_levels: int = 2) -> bool:
    """Verifica se a coluna tem pelo menos min_levels valores únicos não-nulos.
    C(col) no patsy requer >= 2 níveis, senão explode com 'negative dimensions'.
    """
    n = df[col].dropna().nunique()
    if n < min_levels:
        print(f"[WARN] Coluna '{col}' tem apenas {n} nível(is) único(s) — excluída da fórmula.")
    return n >= min_levels


def _build_ols_formula(df: pd.DataFrame) -> str | None:
    """Monta a fórmula OLS com apenas as colunas disponíveis e com níveis suficientes."""
    outcome = "MEDIA_CANDIDATO"
    if outcome not in df.columns:
        return None

    continuous  = ["SCORE_CULT_PAIS", "RENDA", "SCORE_CONSUMO"]
    categorical = ["INTERNET", "TP_SEXO", "TP_COR_RACA", "TP_ESCOLA"]

    terms = (
        [c for c in continuous if c in df.columns and df[c].notna().any()]
        + [f"C({c})" for c in categorical if c in df.columns and _has_min_levels(df, c)]
    )
    if not terms:
        return None
    return f"{outcome} ~ {' + '.join(terms)}"


def _build_multilevel_formula(df: pd.DataFrame) -> tuple[str, str] | tuple[None, None]:
    """Retorna (formula, group_col) ou (None, None) se colunas insuficientes."""
    outcome = "MEDIA_CANDIDATO"
    if outcome not in df.columns:
        return None, None

    predictors = ["SCORE_CULT_PAIS", "SG_UF_ESC", "INTERNET"]
    available  = [c for c in predictors if c in df.columns]
    if not available:
        return None, None

    group_col = "RENDA" if "RENDA" in df.columns else None
    if group_col is None:
        return None, None

    return f"{outcome} ~ {' + '.join(available)}", group_col


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    year        = cfg["year"]
    sample_size = cfg.get("sample_size", 5000)
    seed        = cfg.get("random_seed", 69)
    bucket      = cfg["gcs"]["bucket"] 

    source_blob = f"processed/{year}/dados_enem_processados_{year}.parquet"
    local_file  = Path("tmp") / f"dados_enem_processados_{year}.parquet"
    local_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Baixando gs://{bucket}/{source_blob}...")
    download_file(bucket, source_blob, local_file)

    df = pd.read_parquet(local_file)
    print(f"[INFO] Parquet carregado: {len(df)} linhas | colunas: {list(df.columns)}")

    if len(df) == 0:
        raise RuntimeError(
            f"O parquet processed/{year} está vazio. "
            "Re-execute o process_enem.py para regenerar o arquivo."
        )

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=seed)

    results: dict = {"year": year, "sample_size_used": int(len(df))}

   
    ols_formula = _build_ols_formula(df)
    if ols_formula:
        print(f"[INFO] OLS: {ols_formula}")
        ols_model = smf.ols(ols_formula, data=df).fit()
        results["ols"] = {
            "formula": ols_formula,
            "r2":      float(ols_model.rsquared),
            "adj_r2":  float(ols_model.rsquared_adj),
            "f_stat":  float(ols_model.fvalue),
            "f_pvalue": float(ols_model.f_pvalue),
            "coefficients": {k: float(v) for k, v in ols_model.params.to_dict().items()},
        }
    else:
        print("[WARN] Colunas insuficientes para OLS. Pulando.")
        results["ols"] = None

    
    ml_formula, group_col = _build_multilevel_formula(df)
    if ml_formula:
        print(f"[INFO] Multinível: {ml_formula} | grupos: {group_col}")
        multilevel = smf.mixedlm(ml_formula, data=df, groups=df[group_col]).fit(reml=False)
        results["multilevel"] = {
            "formula":       ml_formula,
            "group_by":      group_col,
            "aic":           float(multilevel.aic),
            "bic":           float(multilevel.bic),
            "log_likelihood": float(multilevel.llf),
            "coefficients":  {k: float(v) for k, v in multilevel.params.to_dict().items()},
        }
    else:
        print("[WARN] Colunas insuficientes para modelo multinível. Pulando.")
        results["multilevel"] = None

    out_dir  = Path(cfg.get("output_dir", "results/statistical"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"statistical_tests_{year}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Resultados salvos em {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Roda OLS e modelo multinível por ano")
    parser.add_argument("--config", default="configs/statistical_tests.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()