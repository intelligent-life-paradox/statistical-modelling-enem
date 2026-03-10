from __future__ import annotations

import argparse
import gc
import google.auth
import pandas as pd
import yaml
from google.cloud import storage
from pathlib import Path
from scripts.enem_pipeline.gcs_utils import upload_file, download_file

REQUIRED_CANONICAL_COLUMNS = [
    "NU_INSCRICAO", "TP_FAIXA_ETARIA", "TP_SEXO", "TP_COR_RACA", "TP_ESCOLA",
    "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC", "TP_SIT_FUNC_ESC", "SG_UF_ESC",
    "Q001", "Q002", "Q005", "Q006", "Q025",
    "Q007", "Q008", "Q010", "Q011", "Q012", "Q013", "Q014", "Q015", "Q016", "Q017", "Q020", "Q024",
    "TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT",
    "TP_STATUS_REDACAO", "IN_TREINEIRO",
    "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO",
]

CHUNK_SIZE = 200_000  # linhas por chunk 


def _blob_exists(bucket_name: str, blob_path: str, credentials) -> bool:
    client = storage.Client(credentials=credentials)
    return client.bucket(bucket_name).blob(blob_path).exists()


def _list_blobs_with_prefix(bucket_name: str, prefix: str, credentials) -> list[str]:
    client = storage.Client(credentials=credentials)
    return [b.name for b in client.bucket(bucket_name).list_blobs(prefix=prefix)]


def _find_csv_blob(bucket_name: str, year: int, credentials) -> str | None:
    prefix = f"raw/enem_{year}/"
    blobs = _list_blobs_with_prefix(bucket_name, prefix, credentials)
    csv_blobs = [b for b in blobs if b.lower().endswith(".csv")]
    if not csv_blobs:
        print(f"[WARN] Nenhum CSV em gs://{bucket_name}/{prefix}")
        return None
    return csv_blobs[0]


def _convert_csv_to_parquet(
    bucket_name: str,
    csv_blob: str,
    credentials,
    local_dir: Path,
    year: int,
) -> Path:
    """
    Baixa o CSV do GCS e converte para parquet lendo em chunks
    para evitar OOM no runner (7GB RAM).
    """
    local_csv = local_dir / f"enem_raw_{year}.csv"
    local_parquet = local_dir / f"enem_raw_{year}.parquet"

    
    if not local_csv.exists():
        print(f"[INFO] Baixando CSV de {year} do GCS (~1-2GB, aguarde)...")
        download_file(bucket_name, csv_blob, local_csv)
        print(f"[INFO] Download concluído: {local_csv.stat().st_size / 1e6:.0f} MB")

    
    available_cols = pd.read_csv(
        local_csv, encoding="latin-1", sep=";", nrows=0
    ).columns.tolist()
    cols_to_read = [c for c in REQUIRED_CANONICAL_COLUMNS if c in available_cols]
    missing = set(REQUIRED_CANONICAL_COLUMNS) - set(cols_to_read)
    if missing:
        print(f"[WARN] Colunas ausentes no CSV de {year}: {sorted(missing)}")

    # Lê em chunks e concatena
    print(f"[INFO] Convertendo CSV -> parquet em chunks de {CHUNK_SIZE:,} linhas...")
    chunks = []
    total = 0
    for i, chunk in enumerate(pd.read_csv(
        local_csv,
        encoding="latin-1",
        sep=";",
        usecols=cols_to_read,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )):
        chunks.append(chunk)
        total += len(chunk)
        print(f"  chunk {i+1}: {total:,} linhas acumuladas")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    df.to_parquet(local_parquet, index=False)
    print(f"[INFO] Parquet salvo: {local_parquet.stat().st_size / 1e6:.0f} MB | {len(df):,} linhas")

    # Remove CSV local para liberar espaço no runner
    local_csv.unlink()
    del df
    gc.collect()

    return local_parquet


def run(config_path: Path) -> None:
    credentials, _ = google.auth.default()

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    years     = cfg["years"]
    bucket    = cfg["gcs"]["bucket"]
    local_dir = Path(cfg["local"]["raw_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        parquet_blob = f"raw/enem_{year}/enem_raw_{year}.parquet"

        if _blob_exists(bucket, parquet_blob, credentials):
            print(f"[SKIP] Parquet de {year} já existe em gs://{bucket}/{parquet_blob}.")
            continue

        csv_blob = _find_csv_blob(bucket, year, credentials)
        if csv_blob is None:
            print(f"[ERROR] Ano {year}: CSV não encontrado. Pulando.")
            continue

        local_parquet = _convert_csv_to_parquet(
            bucket, csv_blob, credentials, local_dir, year
        )

        uri = upload_file(bucket, local_parquet, parquet_blob)
        print(f"[OK] Ano {year} -> {uri}")

        # Remove parquet local após upload para liberar espaço
        local_parquet.unlink()
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()