from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from gcs_utils import download_file, upload_file


RENDA_MAP = {
    "A": 0,
    "B": 998,
    "C": 1497,
    "D": 1996,
    "E": 2495,
    "F": 2994,
    "G": 3992,
    "H": 4990,
    "I": 5988,
    "J": 6986,
    "K": 7984,
    "L": 8982,
    "M": 9980,
    "N": 11976,
    "O": 14970,
    "P": 19960,
    "Q": 25000,
}

EDUC_MAP = {"A": 0, "B": 2, "C": 4, "D": 6, "E": 9, "F": 12, "G": 16, "H": 18}
PESSOAS_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10}
INTERNET_MAP = {"A": 0, "B": 1}


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    notas = [c for c in ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"] if c in df.columns]
    df = df.copy()

    df["MEDIA_CANDIDATO"] = df[notas].mean(axis=1)
    df["RENDA"] = df["Q006"].map(RENDA_MAP)
    df["SCORE_CULT_PAIS"] = df["Q001"].map(EDUC_MAP).fillna(0) + df["Q002"].map(EDUC_MAP).fillna(0)
    df["N_PESSOAS_MESMA_RED"] = df["Q005"].map(PESSOAS_MAP)
    df["INTERNET"] = df["Q024"].map(INTERNET_MAP)

    if "SG_UF_ESC" in df.columns:
        df["SG_UF_ESC"] = df["SG_UF_ESC"].astype("category").cat.codes

    keep = [
        "NU_INSCRICAO",
        "TP_FAIXA_ETARIA",
        "TP_SEXO",
        "TP_ESTADO_CIVIL",
        "TP_COR_RACA",
        "TP_NACIONALIDADE",
        "TP_ST_CONCLUSAO",
        "TP_ANO_CONCLUIU",
        "TP_ESCOLA",
        "TP_ENSINO",
        "SG_UF_ESC",
        "TP_DEPENDENCIA_ADM_ESC",
        "TP_LOCALIZACAO_ESC",
        "TP_SIT_FUNC_ESC",
        "N_PESSOAS_MESMA_RED",
        "INTERNET",
        "MEDIA_CANDIDATO",
        "SCORE_CULT_PAIS",
        "RENDA",
    ]

    out = df[[c for c in keep if c in df.columns]].dropna()

    # score de consumo proxy pela renda per capita domiciliar
    out["SCORE_CONSUMO"] = (out["RENDA"] / out["N_PESSOAS_MESMA_RED"]).clip(lower=0)

    return out


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    years = cfg["years"]
    bucket = cfg["gcs"]["bucket"]
    local_raw = Path(cfg["local"]["raw_dir"])
    local_processed = Path(cfg["local"]["processed_dir"])
    local_processed.mkdir(parents=True, exist_ok=True)

    for year in years:
        local_input = local_raw / f"enem_raw_{year}.parquet"
        if not local_input.exists():
            source_blob = f"raw/{year}/enem_raw_{year}.parquet"
            print(f"[INFO] Baixando {source_blob}...")
            download_file(bucket, source_blob, local_input)

        df = pd.read_parquet(local_input)
        processed = preprocess(df)

        output = local_processed / f"dados_enem_processados_{year}.parquet"
        processed.to_parquet(output, index=False)

        destination = f"processed/{year}/dados_enem_processados_{year}.parquet"
        uri = upload_file(bucket, output, destination)
        print(f"[OK] Processado {year}: {len(processed)} linhas -> {uri}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Processa dados brutos ENEM e salva no GCS")
    parser.add_argument("--config", default="config/pipeline.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()