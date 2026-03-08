from __future__ import annotations

import argparse

from pathlib import Path


import pandas as pd
import google.auth
import yaml

from scripts.enem_pipeline.gcs_utils import upload_file

TABLE_FQN = "`basedosdados.br_inep_enem.microdados`"


ALIASES = {
    "NU_INSCRICAO": ["id_inscricao"],
    "SG_UF_ESC": ["sigla_uf_escola"],
    "NU_NOTA_CN": ["nota_cn"],
    "NU_NOTA_CH": ["nota_ch"],
    "NU_NOTA_LC": ["nota_lc"],
    "NU_NOTA_MT": ["nota_mt"],
    "NU_NOTA_REDACAO": ["nota_redacao"],
    # Mapeamentos para colunas que perderam o prefixo 'tp_'
    "TP_FAIXA_ETARIA": ["faixa_etaria"],
    "TP_SEXO": ["sexo"],
    "TP_ESTADO_CIVIL": ["estado_civil"],
    "TP_COR_RACA": ["cor_raca"],
    "TP_NACIONALIDADE": ["nacionalidade"],
    "TP_ST_CONCLUSAO": ["st_conclusao"],
    "TP_ANO_CONCLUIU": ["ano_concluiu"],
    "TP_ESCOLA": ["escola"],
    "TP_ENSINO": ["ensino"],
    "TP_DEPENDENCIA_ADM_ESC": ["dependencia_adm_esc"],
    "TP_LOCALIZACAO_ESC": ["localizacao_esc"],
    "TP_SIT_FUNC_ESC": ["sit_func_esc"],
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
    return alt if alt.exists() else config_path

def _fetch_available_columns(billing_project_id: str, credentials) -> set[str]:
    
    schema_query = "SELECT column_name FROM `basedosdados.br_inep_enem.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = 'microdados'"
    try:
        cols = pd.read_gbq(schema_query, project_id=billing_project_id, credentials=credentials, dialect="standard")
        return set(cols["column_name"].astype(str).tolist())
    except Exception as e:
        print(f"[ERROR] Falha ao ler esquema: {e}. Tentando nomes padrão...")
        return set()

def _build_select_list(available_cols: set[str]) -> str:
    select_exprs = []
    # Se não conseguirmos ler o esquema, assumimos que as colunas existem em minúsculo (padrão BasedosDados)
    use_fallback = len(available_cols) == 0

    for canonical in REQUIRED_CANONICAL_COLUMNS:
        if not use_fallback and canonical in available_cols:
            select_exprs.append(canonical)
        elif not use_fallback and canonical.lower() in available_cols:
            
            select_exprs.append(f"{canonical.lower()} AS {canonical}")
        
        else:
            # Tenta aliases manuais ou assume minúsculo como fallback
            matched = next((a for a in ALIASES.get(canonical, []) if not use_fallback and a in available_cols), None)
            if matched:
                select_exprs.append(f"{matched} AS {canonical}")
            else:
                #=
                select_exprs.append(f"{ALIASES.get(canonical, [canonical.lower()])[0]} AS {canonical}")
    
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
        print(f" Baixando ENEM {year} do BigQuery...")
        query = f"SELECT {select_list} FROM {TABLE_FQN} WHERE ano = {year}"
        
        
        df = pd.read_gbq(query, project_id=billing_project_id, credentials=credentials, dialect="standard")

        output = local_dir / f"enem_raw_{year}.parquet"
        df.to_parquet(output, index=False)
        #ATUALIZAÇÕES IMPORTANTE:
        # O erro 404 acontecia aqui por causa do nome do bucket
        uri = upload_file(bucket, output, f"raw/{year}/enem_raw_{year}.parquet")
        print(f"[OK] Ano {year} salvo em {uri} | Linhas: {len(df)}")
        
        # Limpeza de memória para não travar o GitHub Actions
        del df

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()