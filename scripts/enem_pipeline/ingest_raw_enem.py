from __future__ import annotations

import argparse
from pathlib import Path


import pandas as pd
import google.auth
import yaml

from scripts.enem_pipeline.gcs_utils import upload_file

TABLE_FQN = "`basedosdados.br_inep_enem.microdados`"

# Mapeamento exaustivo para o padrão da BasedosDados
ALIASES = {
    "NU_INSCRICAO": ["id_inscricao"],
    "SG_UF_ESC": ["sigla_uf_escola"],
    "NU_NOTA_CN": ["nota_cn"],
    "NU_NOTA_CH": ["nota_ch"],
    "NU_NOTA_LC": ["nota_lc"],
    "NU_NOTA_MT": ["nota_mt"],
    "NU_NOTA_REDACAO": ["nota_redacao"],
}

REQUIRED_CANONICAL_COLUMNS = [
    "NU_INSCRICAO", "TP_FAIXA_ETARIA", "TP_SEXO", "TP_ESTADO_CIVIL", "TP_COR_RACA",
    "TP_NACIONALIDADE", "TP_ST_CONCLUSAO", "TP_ANO_CONCLUIU", "TP_ESCOLA", "TP_ENSINO",
    "SG_UF_ESC", "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC", "TP_SIT_FUNC_ESC",
    "Q001", "Q002", "Q005", "Q006", "Q024",
    "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO",
]

def _resolve_config_path(config_path: Path) -> Path:
    if config_path.exists(): return config_path
    alt = Path(str(config_path).replace("configs/", "config/", 1))
    if alt.exists(): return alt
    raise FileNotFoundError(f"Config não encontrado: {config_path}")

def _fetch_available_columns(billing_project_id: str, credentials) -> set[str]:
    schema_query = "SELECT column_name FROM `basedosdados.br_inep_enem.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = 'microdados'"
    cols = pd.read_gbq(schema_query, project_id=billing_project_id, credentials=credentials, dialect="standard")
    return set(cols["column_name"].astype(str).tolist())

def _build_select_list(available_cols: set[str]) -> str:
    select_exprs = []
    for canonical in REQUIRED_CANONICAL_COLUMNS:
        #Tenta o nome original 
        if canonical in available_cols:
            select_exprs.append(canonical)
        #Tenta a versão minúscula (padrão BasedosDados para TP_..., Q...)
        elif canonical.lower() in available_cols:
            select_exprs.append(f"{canonical.lower()} AS {canonical}")
        #Tenta os aliases manuais (ex: id_inscricao)
        else:
            matched = next((a for a in ALIASES.get(canonical, []) if a in available_cols), None)
            if matched:
                select_exprs.append(f"{matched} AS {canonical}")
            else:
                print(f"[WARN] Coluna não encontrada: {canonical}")
    return ",\n  ".join(select_exprs)

def run(config_path: Path) -> None:
   
    credentials, _ = google.auth.default()
    
    cfg_path = _resolve_config_path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    years = cfg["years"]
    bucket = cfg["gcs"]["bucket"]
    billing_project_id = cfg["billing_project_id"]
    local_dir = Path(cfg["local"]["raw_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    available_cols = _fetch_available_columns(billing_project_id, credentials)
    select_list = _build_select_list(available_cols)

    for year in years:
        print(f"[INFO] Baixando ENEM {year}...")
        query = f"SELECT {select_list} FROM {TABLE_FQN} WHERE ano = {year}"
        
        #Uso do pandas_gbq direto para evitar tentativa de abrir navegador
        df = pd.read_gbq(query, project_id=billing_project_id, credentials=credentials, dialect="standard")

        output = local_dir / f"enem_raw_{year}.parquet"
        df.to_parquet(output, index=False)
        uri = upload_file(bucket, output, f"raw/{year}/enem_raw_{year}.parquet")
        print(f"[OK] Ano {year} salvo em {uri} | Linhas: {len(df)}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()