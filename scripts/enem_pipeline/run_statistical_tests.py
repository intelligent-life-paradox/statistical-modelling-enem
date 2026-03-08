from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml

from scripts.enem_pipeline.gcs_utils import  download_file


FORMULA_OLS = (
    "MEDIA_CANDIDATO ~ SCORE_CULT_PAIS + RENDA + SCORE_CONSUMO + C(INTERNET) + "
    "C(TP_SEXO) + C(TP_COR_RACA) + C(TP_ESCOLA) "
)


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    year = cfg["year"]
    sample_size = cfg.get("sample_size", 1500)
    seed = cfg.get("random_seed", 42)
    bucket = cfg["gcs"]["bucket"]
    source_blob = f"processed/{year}/dados_enem_processados_{year}.parquet"

    local_file = Path("tmp") / f"dados_enem_processados_{year}.parquet"
    download_file(bucket, source_blob, local_file)

    df = pd.read_parquet(local_file)
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=seed)

    ols_model = smf.ols(FORMULA_OLS, data=df).fit()

    # Multinível com intercepto aleatório por renda (proxy de classe social)
    multilevel = smf.mixedlm(
        "MEDIA_CANDIDATO ~ SCORE_CULT_PAIS + SG_UF_ESC  + INTERNET",
        data=df,
        groups=df["RENDA"],
    ).fit(reml=False)

    results = {
        "year": year,
        "sample_size_used": int(len(df)),
        "ols": {
            "r2": float(ols_model.rsquared),
            "adj_r2": float(ols_model.rsquared_adj),
            "f_stat": float(ols_model.fvalue),
            "f_pvalue": float(ols_model.f_pvalue),
            "coefficients": {k: float(v) for k, v in ols_model.params.to_dict().items()},
        },
        "multilevel": {
            "aic": float(multilevel.aic),
            "bic": float(multilevel.bic),
            "log_likelihood": float(multilevel.llf),
            "coefficients": {k: float(v) for k, v in multilevel.params.to_dict().items()},
        },
    }

    out_dir = Path(cfg.get("output_dir", "results/statistical"))
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