from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scripts.enem_pipeline.gcs_utils import upload_file, download_file

RENDA_MAP = {
    "A": 0, "B": 998, "C": 1497, "D": 1996, "E": 2495, "F": 2994, "G": 3992,
    "H": 4990, "I": 5988, "J": 6986, "K": 7984, "L": 8982, "M": 9980,
    "N": 11976, "O": 14970, "P": 19960, "Q": 25000,
}

EDUC_MAP    = {"A": 0, "B": 2, "C": 4, "D": 6, "E": 9, "F": 12, "G": 16, "H": 18}
PESSOAS_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10}
INTERNET_MAP = {"A": 0, "B": 1}

# Alinhado com REQUIRED_CANONICAL_COLUMNS do ingest_raw_enem.py
COLS_NECESSARIAS = [
    "NU_INSCRICAO", "TP_SEXO", "TP_COR_RACA", "TP_ESCOLA",
    "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC", "TP_SIT_FUNC_ESC", "SG_UF_ESC",
    "Q001", "Q002", "Q005", "Q006", "Q024",
    "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO",
]


def _safe_read_parquet(path: Path, desired_cols: list[str]) -> pd.DataFrame:
    """Lê apenas as colunas que realmente existem no arquivo."""
    actual_cols = pq.read_schema(path).names
    cols_to_read = [c for c in desired_cols if c in actual_cols]
    missing = set(desired_cols) - set(cols_to_read)
    if missing:
        print(f"[WARN] Colunas ausentes no parquet (serão ignoradas): {sorted(missing)}")
    return pd.read_parquet(path, columns=cols_to_read)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    notas = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
    notas_presentes = [c for c in notas if c in df.columns]
    df["MEDIA_CANDIDATO"] = df[notas_presentes].mean(axis=1) if notas_presentes else pd.NA

    # Campos derivados — só calculados se a coluna-fonte existir
    df["RENDA"] = df["Q006"].map(RENDA_MAP) if "Q006" in df.columns else pd.NA

    if "Q001" in df.columns and "Q002" in df.columns:
        df["SCORE_CULT_PAIS"] = df["Q001"].map(EDUC_MAP).fillna(0) + df["Q002"].map(EDUC_MAP).fillna(0)
    elif "Q001" in df.columns:
        df["SCORE_CULT_PAIS"] = df["Q001"].map(EDUC_MAP).fillna(0)
    elif "Q002" in df.columns:
        df["SCORE_CULT_PAIS"] = df["Q002"].map(EDUC_MAP).fillna(0)
    else:
        df["SCORE_CULT_PAIS"] = pd.NA

    df["N_PESSOAS_MESMA_RED"] = df["Q005"].map(PESSOAS_MAP) if "Q005" in df.columns else pd.NA
    df["INTERNET"]            = df["Q024"].map(INTERNET_MAP) if "Q024" in df.columns else pd.NA

    if "SG_UF_ESC" in df.columns:
        df["SG_UF_ESC"] = df["SG_UF_ESC"].astype("category").cat.codes

    keep = [
        "NU_INSCRICAO", "TP_SEXO", "TP_COR_RACA", "TP_ESCOLA",
        "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC", "TP_SIT_FUNC_ESC", "SG_UF_ESC",
        "N_PESSOAS_MESMA_RED", "INTERNET", "MEDIA_CANDIDATO", "SCORE_CULT_PAIS", "RENDA",
    ]

    out = df[[c for c in keep if c in df.columns]].dropna()

    if "RENDA" in out.columns and "N_PESSOAS_MESMA_RED" in out.columns:
        out["SCORE_CONSUMO"] = (out["RENDA"] / out["N_PESSOAS_MESMA_RED"]).clip(lower=0)

    return out


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    years           = cfg["years"]
    bucket          = cfg["gcs"]["bucket"]
    local_raw       = Path(cfg["local"]["raw_dir"])
    local_processed = Path(cfg["local"]["processed_dir"])
    local_processed.mkdir(parents=True, exist_ok=True)

    for year in years:
        local_input = local_raw / f"enem_raw_{year}.parquet"

        if not local_input.exists():
            source_blob = f"raw/{year}/enem_raw_{year}.parquet"
            print(f"[INFO] Baixando {source_blob} do GCS...")
            download_file(bucket, source_blob, local_input)

        print(f"[INFO] Lendo dados de {year}...")
        df = _safe_read_parquet(local_input, COLS_NECESSARIAS)

        processed = preprocess(df)
        del df

        output = local_processed / f"dados_enem_processados_{year}.parquet"
        processed.to_parquet(output, index=False)

        destination = f"processed/{year}/dados_enem_processados_{year}.parquet"
        uri = upload_file(bucket, output, destination)
        print(f"[OK] Processado {year}: {len(processed)} linhas -> {uri}")


def main() -> None:
    
    parser = argparse.ArgumentParser(description="Processa dados brutos ENEM e salva no GCS")
    
    parser.add_argument("--config", default="configs/pipeline.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
