from __future__ import annotations

import argparse
import google.auth
import pandas as pd
import yaml
from google.cloud import storage
from pathlib import Path
from scripts.enem_pipeline.gcs_utils import upload_file, download_file



REQUIRED_CANONICAL_COLUMNS = [
    # Identificação e perfil
    "NU_INSCRICAO", "TP_FAIXA_ETARIA", "TP_SEXO", "TP_COR_RACA", "TP_ESCOLA",
    "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC", "TP_SIT_FUNC_ESC", "SG_UF_ESC",
    # Questionário socioeconômico base
    "Q001", "Q002", "Q005", "Q006", "Q025",
    # Questionário ABEP (score de consumo)
    "Q007", "Q008", "Q010", "Q011", "Q012", "Q013", "Q014", "Q015", "Q016", "Q017", "Q020", "Q024",
    # Presença e status (para filtrar apenas quem fez todas as provas)
    "TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT",
    "TP_STATUS_REDACAO", "IN_TREINEIRO",
    # Notas
    "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO",
]


def _blob_exists(bucket_name: str, blob_path: str, credentials) -> bool:
    client = storage.Client(credentials=credentials)
    return client.bucket(bucket_name).blob(blob_path).exists()


def _list_blobs_with_prefix(bucket_name: str, prefix: str, credentials) -> list[str]:
    """Lista todos os blobs com determinado prefixo."""
    client = storage.Client(credentials=credentials)
    return [b.name for b in client.bucket(bucket_name).list_blobs(prefix=prefix)]


def _find_csv_blob(bucket_name: str, year: int, credentials) -> str | None:
    """Encontra o CSV do ano dentro de raw/enem_{year}/."""
    prefix = f"raw/enem_{year}/"
    blobs = _list_blobs_with_prefix(bucket_name, prefix, credentials)
    csv_blobs = [b for b in blobs if b.lower().endswith(".csv")]
    if not csv_blobs:
        print(f"[WARN] Nenhum CSV encontrado em gs://{bucket_name}/{prefix}")
        return None
    return csv_blobs[0]  


def _read_csv_from_gcs(bucket_name: str, blob_path: str, credentials, local_dir: Path) -> pd.DataFrame:
    """Baixa o CSV do GCS e lê apenas as colunas necessárias."""
    local_csv = local_dir / Path(blob_path).name
    if not local_csv.exists():
        print(f"[INFO] Baixando {blob_path}...")
        download_file(bucket_name, blob_path, local_csv)

    # Lê o CSV — encoding latin-1 padrão dos microdados 
    available_cols = pd.read_csv(local_csv, encoding="latin-1", sep=";", nrows=0).columns.tolist()
    cols_to_read = [c for c in REQUIRED_CANONICAL_COLUMNS if c in available_cols]
    missing = set(REQUIRED_CANONICAL_COLUMNS) - set(cols_to_read)
    if missing:
        print(f"[WARN] Colunas ausentes no CSV de {Path(blob_path).name}: {sorted(missing)}")

    print(f"[INFO] Lendo CSV ({len(cols_to_read)} colunas)...")
    df = pd.read_csv(
        local_csv,
        encoding="latin-1",
        sep=";",
        usecols=cols_to_read,
        low_memory=False,
    )
    return df


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

        # SKIP se parquet já existe 
        if _blob_exists(bucket, parquet_blob, credentials):
            print(f"[SKIP] Parquet de {year} já existe em gs://{bucket}/{parquet_blob}.")
            continue
        

        csv_blob = _find_csv_blob(bucket, year, credentials)
        if csv_blob is None:
            print(f"[ERROR] Ano {year}: CSV não encontrado no bucket. Pulando.")
            continue

        df = _read_csv_from_gcs(bucket, csv_blob, credentials, local_dir)
        print(f"[INFO] Ano {year}: {len(df)} linhas lidas do CSV.")

        output = local_dir / f"enem_raw_{year}.parquet"
        df.to_parquet(output, index=False)
        del df

        uri = upload_file(bucket, output, parquet_blob)
        print(f"[OK] Ano {year} -> {uri}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()