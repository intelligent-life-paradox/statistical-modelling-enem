from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from econml.cate_interpreter import SingleTreeCateInterpreter
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_text

from scripts.enem_pipeline.gcs_utils import download_file


FEATURES = [
    "SCORE_CULT_PAIS",
    "RENDA",
    "SCORE_CONSUMO",
    "INTERNET",
    "TP_SEXO",
    "TP_COR_RACA",
    "TP_ESCOLA",
    "TP_DEPENDENCIA_ADM_ESC",
    "TP_LOCALIZACAO_ESC",
    "TP_SIT_FUNC_ESC",
]


def _fit_causal_model(
    y: np.ndarray, t: np.ndarray, x: np.ndarray, random_seed: int
) -> CausalForestDML:
    model = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=250, min_samples_leaf=20, random_state=random_seed),
        model_t=RandomForestRegressor(n_estimators=250, min_samples_leaf=20, random_state=random_seed),
        n_estimators=400,
        min_samples_leaf=10,
        random_state=random_seed,
    )
    model.fit(y, t, X=x)
    return model


def _causal_tree_rules(
    model: CausalForestDML,
    x: np.ndarray,
    feature_names: list[str],
    max_depth: int,
) -> str:
    interpreter = SingleTreeCateInterpreter(
        max_depth=max_depth, min_samples_leaf=40, include_model_uncertainty=False
    )
    interpreter.interpret(model, x)
    return export_text(interpreter.tree_model_, feature_names=feature_names)


def estimate_effect(
    df: pd.DataFrame,
    treatment: str,
    random_seed: int,
    tree_max_depth: int,
    outcome: str = "MEDIA_CANDIDATO",
) -> dict[str, Any] | None:
    if outcome not in df.columns:
        print(f"[WARN] Coluna de outcome '{outcome}' ausente. Pulando tratamento={treatment}.")
        return None

    if treatment not in df.columns:
        print(f"[WARN] Coluna de tratamento '{treatment}' ausente. Pulando.")
        return None

    x_cols = [c for c in FEATURES if c in df.columns and c != treatment]
    if not x_cols:
        print(f"[WARN] Nenhuma feature disponível para tratamento={treatment}. Pulando.")
        return None

    x = df[x_cols].fillna(0).to_numpy()
    t = df[treatment].astype(float).to_numpy()
    y = df[outcome].astype(float).to_numpy()

    model   = _fit_causal_model(y=y, t=t, x=x, random_seed=random_seed)
    effects = model.effect(x)
    rules   = _causal_tree_rules(model, x=x, feature_names=x_cols, max_depth=tree_max_depth)

    return {
        "treatment":      treatment,
        "features_used":  x_cols,
        "tree_type":      "SingleTreeCateInterpreter",
        "tree_max_depth": tree_max_depth,
        "ate":            float(np.mean(effects)),
        "effect_std":     float(np.std(effects)),
        "effect_p10":     float(np.quantile(effects, 0.10)),
        "effect_p50":     float(np.quantile(effects, 0.50)),
        "effect_p90":     float(np.quantile(effects, 0.90)),
        "n":              int(len(df)),
        "tree_rules":     rules,
    }


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    year           = cfg["year"]
    sample_size    = cfg.get("sample_size", 100_000)
    seed           = cfg.get("random_seed", 69)
    tree_max_depth = cfg.get("tree_max_depth", 3)
    treatments     = cfg.get("treatments", ["RENDA", "SCORE_CULT_PAIS"])
    bucket         = cfg["gcs"]["bucket"]  

    
    source_blob = f"processed/enem_{year}/dados_enem_processados_{year}.parquet"
    local_file  = Path("tmp") / f"dados_enem_processados_{year}.parquet"
    local_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Baixando gs://{bucket}/{source_blob}...")
    download_file(bucket, source_blob, local_file)

    df = pd.read_parquet(local_file)
    print(f"[INFO] Parquet carregado: {len(df)} linhas | colunas: {list(df.columns)}")

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=seed)

    effects = []
    for treatment in treatments:
        print(f"[INFO] Estimando árvore causal para tratamento={treatment}...")
        result = estimate_effect(
            df=df,
            treatment=treatment,
            random_seed=seed,
            tree_max_depth=tree_max_depth,
        )
        if result is not None:
            effects.append(result)

    results = {
        "year":             year,
        "sample_size_used": int(len(df)),
        "effects":          effects,
    }

    out_dir  = Path(cfg.get("output_dir", "results/causal"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"causal_effects_{year}.json"
    
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"[OK] Resultados causais salvos em {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Roda árvore causal por tratamento (RENDA e SCORE_CULT_PAIS)")
    parser.add_argument("--config", default="configs/causal_trees.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()