
from __future__ import annotations

import argparse
import google.auth
import pandas as pd
import yaml
from pathlib import Path
from scripts.enem_pipeline.gcs_utils import upload_file

TABLE_FQN = "`basedosdados.br_inep_enem.microdados`"

# Colunas canônicas necessárias para os testes estatísticos e causalidade
REQUIRED_CANONICAL_COLUMNS = [
    "NU_INSCRICAO", "TP_SEXO", "TP_COR_RACA", "TP_ESCOLA", "TP_DEPENDENCIA_ADM_ESC",
    "TP_LOCALIZACAO_ESC", "TP_SIT_FUNC_ESC", "SG_UF_ESC", "Q001", "Q002", "Q005", 
    "Q006", "Q024", "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO",
]

# Dicionário de sinônimos para lidar com a evolução do esquema da BasedosDados
ALIASES = {
    "NU_INSCRICAO": ["nu_inscricao", "id_inscricao"],
    "TP_SEXO": ["tp_sexo", "sexo"],
    "TP_COR_RACA": ["tp_cor_raca", "cor_raca"],
    "TP_ESCOLA": ["tp_escola", "escola"],
    "TP_DEPENDENCIA_ADM_ESC": ["tp_dependencia_adm_esc", "dependencia_adm_escola"],
    "TP_LOCALIZACAO_ESC": ["tp_localizacao_esc", "localizacao_escola"],
    "TP_SIT_FUNC_ESC": ["tp_sit_func_esc", "situacao_funcionamento_escola"],
    "SG_UF_ESC": ["sg_uf_esc", "uf_escola", "sigla_uf_escola"],
    "Q006": ["q006", "renda"],
    "Q024": ["q024", "internet"],
    "NU_NOTA_CN": ["nota_cn", "nu_nota_cn", "nota_ciencias_natureza"],
    "NU_NOTA_CH": ["nota_ch", "nu_nota_ch", "nota_ciencias_humanas"],
    "NU_NOTA_LC": ["nota_lc", "nu_nota_lc", "nota_linguagens"],
    "NU_NOTA_MT": ["nota_mt", "nu_nota_mt", "nota_matematica"],
    "NU_NOTA_REDACAO": ["nota_redacao", "nu_nota_redacao", "nota_red"],

}

def _resolve_config_path(config_path: Path) -> Path:
    if config_path.exists(): return config_path
    alt = Path(str(config_path).replace("config/", "configs/", 1))
    return alt if alt.exists() else config_path

def _fetch_available_columns(project_id: str, credentials) -> list[str]:
    """Sonda o esquema real da tabela sem abrir navegador."""
    try:
        # Tenta o probe rápido via SELECT * LIMIT 1
        probe_df = pd.read_gbq(
            f"SELECT * FROM {TABLE_FQN} LIMIT 1",
            project_id=project_id,
            credentials=credentials,
            dialect="standard"
        )
        return [str(c) for c in probe_df.columns]
    except Exception as exc:
        print(f"[WARN] Falha no probe rápido: {exc}. Tentando INFORMATION_SCHEMA...")
        
    schema_query = f"SELECT column_name FROM `basedosdados.br_inep_enem.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = 'microdados'"
    cols_df = pd.read_gbq(schema_query, project_id=project_id, credentials=credentials, dialect="standard")
    return cols_df["column_name"].astype(str).tolist()


def _match_column(canonical: str, actual_cols: list[str]) -> str | None:
    
    lower_to_actual = {c.lower(): c for c in actual_cols}
    
    if canonical.lower() in lower_to_actual:
        return lower_to_actual[canonical.lower()]
    
    for alias in ALIASES.get(canonical, []):
        if alias.lower() in lower_to_actual:
            return lower_to_actual[alias.lower()]
    
    return None


def _resolve_year_column(actual_cols: list[str]) -> str:
    for cand in ["ano", "ANO", "nu_ano", "NU_ANO"]:
        for c in actual_cols:
            if c.lower() == cand.lower(): return c
    raise RuntimeError("Coluna de ano não encontrada no BigQuery.")

def _build_query(year: int, actual_cols: list[str]) -> str:
    
    select_exprs = []
    
    for canonical in REQUIRED_CANONICAL_COLUMNS:
        source_col = _match_column(canonical, actual_cols)
        if source_col:
            select_exprs.append(f"`{source_col}` AS {canonical}" if source_col != canonical else f"`{source_col}`")
    
    year_col = _resolve_year_column(actual_cols)
    return f"SELECT {', '.join(select_exprs)} FROM {TABLE_FQN} WHERE `{year_col}` = {year}"

def run(config_path: Path) -> None:
    # autenticação via Workload Identity Federation (gitHub actions)
    credentials, _ = google.auth.default()
    
    cfg_path = _resolve_config_path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

   
    years = cfg["years"]
    bucket = cfg["gcs"]["bucket"]
    billing_project_id = cfg["billing_project_id"]
    
    local_dir = Path(cfg["local"]["raw_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    actual_cols = _fetch_available_columns(billing_project_id, credentials)
    print(f"[INFO] Esquema detectado com {len(actual_cols)} colunas.")

    for year in years:
        print(f"[INFO] Baixando ENEM {year} via Service Account...")
        query = _build_query(year, actual_cols)
        
        df = pd.read_gbq(query, project_id=billing_project_id, credentials=credentials, dialect="standard")

        output = local_dir / f"enem_raw_{year}.parquet"
        df.to_parquet(output, index=False)
        
        uri = upload_file(bucket, output, f"raw/{year}/enem_raw_{year}.parquet")
        print(f"[OK] Ano {year} salvo em {uri} | Linhas: {len(df)}")
        
        del df # Limpeza crucial de memória RAM

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pipeline.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()