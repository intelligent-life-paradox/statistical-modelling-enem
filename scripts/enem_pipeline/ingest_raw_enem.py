from __future__ import annotations

import argparse
from pathlib import Path

import google.auth
import pandas as pd
import basedosdados as bd
import yaml

from scripts.enem_pipeline.gcs_utils import upload_file


SQL_TEMPLATE = """
SELECT 
    id_inscricao AS NU_INSCRICAO,
    ano AS NU_ANO,
    sigla_uf_escola AS SG_UF_ESC,
    nota_mt AS NU_NOTA_MT,
    nota_ch AS NU_NOTA_CH,
    nota_lc AS NU_NOTA_LC,
    nota_cn AS NU_NOTA_CN,
    nota_redacao AS NU_NOTA_REDACAO,
    q001 AS Q001,
    q002 AS Q002,
    q005 AS Q005,
    q006 AS Q006,
    q024 AS Q024,
    tp_faixa_etaria AS TP_FAIXA_ETARIA,
    tp_sexo AS TP_SEXO,
    tp_estado_civil AS TP_ESTADO_CIVIL,
    tp_cor_raca AS TP_COR_RACA,
    tp_nacionalidade AS TP_NACIONALIDADE,
    tp_st_conclusao AS TP_ST_CONCLUSAO,
    tp_ano_concluiu AS TP_ANO_CONCLUIU,
    tp_escola AS TP_ESCOLA,
    tp_ensino AS TP_ENSINO,
    tp_dependencia_adm_esc AS TP_DEPENDENCIA_ADM_ESC,
    tp_localizacao_esc AS TP_LOCALIZACAO_ESC,
    tp_sit_func_esc AS TP_SIT_FUNC_ESC
FROM `basedosdados.br_inep_enem.microdados`
WHERE ano = {year}
"""


def run(config_path: Path) -> None:
    credentials, _ = google.auth.default()

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    years = cfg["years"]
    bucket = cfg["gcs"]["bucket"]
    local_dir = Path(cfg["local"]["raw_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        print(f"[INFO] Baixando ENEM {year} do BigQuery via Service Account")
        
        df = pd.read_gbq(
            SQL_TEMPLATE.format(year=year),
            project_id=cfg["billing_project_id"],
            credentials=credentials,
            dialect="standard"
        )

        output = local_dir / f"enem_raw_{year}.parquet"
        df.to_parquet(output, index=False)
        
        destination = f"raw/{year}/enem_raw_{year}.parquet"
        uri = upload_file(bucket, output, destination)
        print(f"[OK] Ano {year} salvo em {uri}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestão de dados brutos ENEM 2014-2019 para GCS")
    parser.add_argument("--config", default="configs/pipeline.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()