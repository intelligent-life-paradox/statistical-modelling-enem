from __future__ import annotations

import argparse
from pathlib import Path

import basedosdados as bd
import pandas as pd
import yaml

from scripts.enem_pipeline.gcs_utils import upload_file


SQL_TEMPLATE = """
SELECT
  NU_INSCRICAO,
  TP_FAIXA_ETARIA,
  TP_SEXO,
  TP_ESTADO_CIVIL,
  TP_COR_RACA,
  TP_NACIONALIDADE,
  TP_ST_CONCLUSAO,
  TP_ANO_CONCLUIU,
  TP_ESCOLA,
  TP_ENSINO,
  SG_UF_ESC,
  TP_DEPENDENCIA_ADM_ESC,
  TP_LOCALIZACAO_ESC,
  TP_SIT_FUNC_ESC,
  Q001,
  Q002,
  Q005,
  Q006,
  Q024,
  NU_NOTA_CN,
  NU_NOTA_CH,
  NU_NOTA_LC,
  NU_NOTA_MT,
  NU_NOTA_REDACAO
FROM `basedosdados.br_inep_enem.microdados`
WHERE ANO = {year}
"""


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    years = cfg["years"]
    bucket = cfg["gcs"]["bucket"]
    local_dir = Path(cfg["local"]["raw_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        print(f"[INFO] Baixando ENEM {year} do BasedosDados...")
        df = bd.read_sql(SQL_TEMPLATE.format(year=year), billing_project_id=cfg["billing_project_id"])

        output = local_dir / f"enem_raw_{year}.parquet"
        df.to_parquet(output, index=False)
        destination = f"raw/{year}/enem_raw_{year}.parquet"
        uri = upload_file(bucket, output, destination)
        print(f"[OK] Ano {year} salvo em {uri}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestão de dados brutos ENEM 2014-2019 para GCS")
    parser.add_argument("--config", default="config/pipeline.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()