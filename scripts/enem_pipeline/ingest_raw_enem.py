from __future__ import annotations

import argparse
from pathlib import Path

import basedosdados as bd
import yaml

from scripts.enem_pipeline.gcs_utils import upload_file

TABLE_FQN = "`basedosdados.br_inep_enem.microdados`"

# Apenas colunas necessárias para processamento/inferência
REQUIRED_CANONICAL_COLUMNS = [
    "NU_INSCRICAO",
    "TP_SEXO",
    "TP_COR_RACA",
    "TP_ESCOLA",
    "TP_DEPENDENCIA_ADM_ESC",
    "TP_LOCALIZACAO_ESC",
    "TP_SIT_FUNC_ESC",
    "SG_UF_ESC",
    "Q001",
    "Q002",
    "Q005",
    "Q006",
    "Q024",
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
]

# aliases (inclui variações em minúsculo observadas no BD)
ALIASES = {
    "NU_INSCRICAO": ["nu_inscricao", "id_inscricao"],
    "TP_SEXO": ["tp_sexo", "sexo"],
    "TP_COR_RACA": ["tp_cor_raca", "cor_raca"],
    "TP_ESCOLA": ["tp_escola", "escola"],
    "TP_DEPENDENCIA_ADM_ESC": ["tp_dependencia_adm_esc", "dependencia_adm_escola"],
    "TP_LOCALIZACAO_ESC": ["tp_localizacao_esc", "localizacao_escola"],
    "TP_SIT_FUNC_ESC": ["tp_sit_func_esc", "situacao_funcionamento_escola"],
    "SG_UF_ESC": ["sg_uf_esc", "uf_escola"],
    "Q001": ["q001"],
    "Q002": ["q002"],
    "Q005": ["q005"],
    "Q006": ["q006", "renda"],
    "Q024": ["q024", "internet"],
    "NU_NOTA_CN": ["nu_nota_cn", "nota_cn", "NU_NOTA_CIENCIAS_NATUREZA", "nota_ciencias_natureza"],
    "NU_NOTA_CH": ["nu_nota_ch", "nota_ch", "NU_NOTA_CIENCIAS_HUMANAS", "nota_ciencias_humanas"],
    "NU_NOTA_LC": ["nu_nota_lc", "nota_lc", "NU_NOTA_LINGUAGENS", "nota_linguagens"],
    "NU_NOTA_MT": ["nu_nota_mt", "nota_mt", "NU_NOTA_MATEMATICA", "nota_matematica"],
    "NU_NOTA_REDACAO": ["nu_nota_redacao", "nota_redacao", "NU_NOTA_RED", "nota_red"],
}


def _resolve_config_path(config_path: Path) -> Path:
    if config_path.exists():
        return config_path

    # Compatibilidade: se alguém passar `config/...`, redireciona para `configs/...`.
    alt = Path(str(config_path).replace("config/", "configs/", 1))
    if alt.exists():
        print(f"[WARN] Config não encontrado em {config_path}; usando {alt}.")
        return alt

    raise FileNotFoundError(f"Config não encontrado: {config_path}")


def _fetch_available_columns(billing_project_id: str) -> list[str]:
    """Obtém colunas reais queryáveis da tabela.

    Primeiro tenta `SELECT * LIMIT 1` (fonte mais confiável para evitar divergências
    de INFORMATION_SCHEMA). Em caso de falha, cai para INFORMATION_SCHEMA.
    """
    try:
        probe_df = bd.read_sql(
            f"SELECT * FROM {TABLE_FQN} LIMIT 1",
            billing_project_id=billing_project_id,
        )
        cols = [str(c) for c in probe_df.columns]
        if cols:
            return cols
    except Exception as exc:
        print(f"[WARN] Falha no probe de colunas via SELECT * LIMIT 1: {exc}")

    schema_query = """
    SELECT column_name
    FROM `basedosdados.br_inep_enem.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = 'microdados'
    """
    cols_df = bd.read_sql(schema_query, billing_project_id=billing_project_id)
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
    for cand in ["ANO", "ano", "NU_ANO", "nu_ano"]:
        for c in actual_cols:
            if c.lower() == cand.lower():
                return c
    raise RuntimeError("Coluna de ano não encontrada (esperado algo como ANO/ano).")


def _build_query(year: int, actual_cols: list[str]) -> str:
    select_exprs = []
    missing = []

    for canonical in REQUIRED_CANONICAL_COLUMNS:
        source_col = _match_column(canonical, actual_cols)
        if source_col is None:
            missing.append(canonical)
            continue

        if source_col == canonical:
            select_exprs.append(f"`{source_col}`")
        else:
            select_exprs.append(f"`{source_col}` AS {canonical}")

    if missing:
        print(f"[WARN] Colunas ausentes e descartadas na ingestão: {missing}")

    if not select_exprs:
        raise RuntimeError("Nenhuma coluna útil encontrada para ingestão.")

    year_col = _resolve_year_column(actual_cols)

    select_list = ",\n  ".join(select_exprs)
    return f"""
SELECT
  {select_list}
FROM {TABLE_FQN}
WHERE `{year_col}` = {year}
"""


def run(config_path: Path) -> None:
    cfg_path = _resolve_config_path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    years = cfg["years"]
    bucket = cfg["gcs"]["bucket"]
    billing_project_id = cfg["billing_project_id"]

    local_dir = Path(cfg["local"]["raw_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    actual_cols = _fetch_available_columns(billing_project_id=billing_project_id)
    print(f"[INFO] Esquema detectado com {len(actual_cols)} colunas.")

    for year in years:
        print(f"[INFO] Baixando ENEM {year} do BigQuery via BasedosDados...")
        query = _build_query(year=year, actual_cols=actual_cols)
        df = bd.read_sql(query, billing_project_id=billing_project_id)

        output = local_dir / f"enem_raw_{year}.parquet"
        df.to_parquet(output, index=False)

        destination = f"raw/{year}/enem_raw_{year}.parquet"
        uri = upload_file(bucket, output, destination)
        print(f"[OK] Ano {year} salvo em {uri} | linhas={len(df)} | colunas={len(df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestão de dados brutos ENEM 2014-2019 para GCS")
    parser.add_argument("--config", default="configs/pipeline.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()