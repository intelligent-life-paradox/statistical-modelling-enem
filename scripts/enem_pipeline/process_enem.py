from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scripts.enem_pipeline.gcs_utils import upload_file, download_file

# Pesos ABEP 
PESOS_ABEP = {
    "Q008": {"A": 0, "B": 3, "C": 7,  "D": 10, "E": 14},  # Banheiros
    "Q007": {"A": 0, "B": 3, "C": 7,  "D": 10, "E": 13},  # Empregados domésticos
    "Q010": {"A": 0, "B": 3, "C": 5,  "D": 8,  "E": 11},  # Automóveis
    "Q024": {"A": 0, "B": 3, "C": 6,  "D": 8,  "E": 11},  # Microcomputador
    "Q017": {"A": 0, "B": 3, "C": 6,  "D": 6,  "E": 6 },  # Lava louça
    "Q012": {"A": 0, "B": 2, "C": 3,  "D": 5,  "E": 5 },  # Geladeira
    "Q013": {"A": 0, "B": 2, "C": 4,  "D": 6,  "E": 6 },  # Freezer
    "Q014": {"A": 0, "B": 2, "C": 4,  "D": 6,  "E": 6 },  # Lava roupa
    "Q020": {"A": 0, "B": 1, "C": 3,  "D": 4,  "E": 6 },  # DVD
    "Q016": {"A": 0, "B": 2, "C": 4,  "D": 4,  "E": 4 },  # Micro-ondas
    "Q011": {"A": 0, "B": 1, "C": 3,  "D": 3,  "E": 3 },  # Motocicleta
    "Q015": {"A": 0, "B": 2, "C": 2,  "D": 2,  "E": 2 },  # Secadora roupa
}

# Pontos médios das faixas de renda (notebook cell 37)
MAPA_RENDA = {
    "A": 0,     "B": 499,   "C": 1247,  "D": 1746,  "E": 2245,
    "F": 2744,  "G": 3493,  "H": 4491,  "I": 5489,  "J": 6487,
    "K": 7485,  "L": 8483,  "M": 9481,  "N": 10978, "O": 13473,
    "P": 17465, "Q": 25000,
}

MAPA_INSTRUCAO = {
    "A": 0, "B": 0, "C": 1, "D": 2, "E": 4, "F": 7, "G": 7, "H": float("nan"),
}

MAPA_SEXO    = {"M": 0, "F": 1}
MAPA_PESSOAS = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5,
                 "F": 6, "G": 7, "H": 8, "I": 9, "J": 10}

COLS_NECESSARIAS = [
    "NU_INSCRICAO", "TP_FAIXA_ETARIA", "TP_SEXO", "TP_COR_RACA", "TP_ESCOLA",
    "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC", "TP_SIT_FUNC_ESC", "SG_UF_ESC",
    "Q001", "Q002", "Q005", "Q006", "Q025",
    "Q007", "Q008", "Q010", "Q011", "Q012", "Q013", "Q014", "Q015", "Q016", "Q017", "Q020", "Q024",
    "TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT",
    "TP_STATUS_REDACAO", "IN_TREINEIRO",
    "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO",
]


def _safe_read_parquet(path: Path, desired_cols: list[str]) -> pd.DataFrame:
    actual_cols = pq.read_schema(path).names
    cols_to_read = [c for c in desired_cols if c in actual_cols]
    missing = set(desired_cols) - set(cols_to_read)
    if missing:
        print(f"[WARN] Colunas ausentes no parquet: {sorted(missing)}")
    return pd.read_parquet(path, columns=cols_to_read)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # filtros a seguir 
    # Remove treineiros
    if "IN_TREINEIRO" in df.columns:
        df = df[df["IN_TREINEIRO"] == 0].drop(columns=["IN_TREINEIRO"])

    
    presenca_cols = ["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT"]
    presenca_presentes = [c for c in presenca_cols if c in df.columns]
    for col in presenca_presentes:
        df = df[df[col] == 1]

    if "TP_STATUS_REDACAO" in df.columns:
        df = df[df["TP_STATUS_REDACAO"] == 1]

    df = df.drop(columns=[c for c in presenca_cols + ["TP_STATUS_REDACAO"] if c in df.columns])

    
    drop_na_cols = [c for c in ["SG_UF_ESC"] if c in df.columns]
    if drop_na_cols:
        df = df.dropna(subset=drop_na_cols)

    notas = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
    notas_presentes = [c for c in notas if c in df.columns]
    if notas_presentes:
        df["MEDIA_CANDIDATO"] = df[notas_presentes].sum(axis=1) / 5
    df = df.drop(columns=[c for c in notas if c in df.columns])

    
    df["SCORE_CONSUMO"] = 0
    for col, mapping in PESOS_ABEP.items():
        if col in df.columns:
            df["SCORE_CONSUMO"] += df[col].map(mapping).fillna(0)

    
    pai = df["Q001"].map(MAPA_INSTRUCAO).fillna(0) if "Q001" in df.columns else 0
    mae = df["Q002"].map(MAPA_INSTRUCAO).fillna(0) if "Q002" in df.columns else 0
    df["SCORE_CULT_PAIS"] = pai + mae

   
    df["RENDA"] = df["Q006"].map(MAPA_RENDA).fillna(0) if "Q006" in df.columns else 0

    
    df["INTERNET"] = df["Q025"].map({"A": 0, "B": 1}).fillna(0) if "Q025" in df.columns else 0

    
    if "Q005" in df.columns:
        df = df.rename(columns={"Q005": "N_PESSOAS_MESMA_RED"})
        df["N_PESSOAS_MESMA_RED"] = df["N_PESSOAS_MESMA_RED"].map(MAPA_PESSOAS)

    
    if "TP_SEXO" in df.columns:
        df["TP_SEXO"] = df["TP_SEXO"].map(MAPA_SEXO)

    #  SG_UF_ESC numérico (notebook cells 50-52)
    if "SG_UF_ESC" in df.columns:
        df["SG_UF_ESC"] = df["SG_UF_ESC"].astype("category").cat.codes

    
    cols_to_drop = (
        list(PESOS_ABEP.keys())
        + ["Q001", "Q002", "Q006", "Q025"]
        + ["Q003", "Q004", "Q009", "Q018", "Q019", "Q021", "Q022", "Q023"]
    )
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    
    keep = [
        "NU_INSCRICAO", "TP_FAIXA_ETARIA", "TP_SEXO", "TP_COR_RACA", "TP_ESCOLA",
        "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC", "TP_SIT_FUNC_ESC", "SG_UF_ESC",
        "N_PESSOAS_MESMA_RED", "INTERNET", "MEDIA_CANDIDATO",
        "SCORE_CULT_PAIS", "RENDA", "SCORE_CONSUMO",
    ]
    out = df[[c for c in keep if c in df.columns]]
    out = out.dropna(subset=["MEDIA_CANDIDATO"])

    print(f"[INFO] Linhas após preprocess: {len(out)} | colunas: {list(out.columns)}")
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
        #SKIP se processed já existe no GCS
        processed_blob = f"processed/enem_{year}/dados_enem_processados_{year}.parquet"

        local_input = local_raw / f"enem_raw_{year}.parquet"
        if not local_input.exists():
            source_blob = f"raw/enem_{year}/enem_raw_{year}.parquet"
            print(f"[INFO] Baixando parquet raw de {year}...")
            download_file(bucket, source_blob, local_input)

        print(f"[INFO] Processando {year}...")
        df = _safe_read_parquet(local_input, COLS_NECESSARIAS)

        processed = preprocess(df)
        del df

        if len(processed) == 0:
            print(f"[WARN] Ano {year}: 0 linhas após processamento.")
            continue

        output = local_processed / f"dados_enem_processados_{year}.parquet"
        processed.to_parquet(output, index=False)

        uri = upload_file(bucket, output, processed_blob)
        print(f"[OK] Processado {year}: {len(processed)} linhas -> {uri}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()