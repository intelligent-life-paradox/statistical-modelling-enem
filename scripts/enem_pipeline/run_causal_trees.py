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
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_text

from scripts.enem_pipeline.gcs_utils import download_file


FEATURES = [
    "SCORE_CULT_PAIS",
    "RENDA",
    "INTERNET",
    "TP_SEXO",
    "TP_COR_RACA",
    "TP_ESCOLA",
    "TP_DEPENDENCIA_ADM_ESC",
    "TP_LOCALIZACAO_ESC",
    "TP_SIT_FUNC_ESC",
]


def _standardize(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Padroniza treatment e outcome para desvio-padrão = 1.
    Retorna df transformado e metadados de escala para interpretação.
    """
    df = df.copy()
    meta = {}

    for col in [treatment, outcome]:
        if col in df.columns:
            mu, sigma = df[col].mean(), df[col].std()
            df[col] = (df[col] - mu) / sigma
            meta[col] = {"mean": float(mu), "std": float(sigma)}
            print(f"[INFO] Padronizado '{col}': média={mu:.2f}, std={sigma:.2f}")

    return df, meta


def _fit_causal_model(
    y: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    random_seed: int,
    n_estimators: int,
    model_n_estimators: int,
    min_samples_leaf: int,
) -> CausalForestDML:
    model = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=model_n_estimators,
            min_samples_leaf=20,
            random_state=random_seed,
            n_jobs=-1,
        ),
        model_t=RandomForestRegressor(
            n_estimators=model_n_estimators,
            min_samples_leaf=20,
            random_state=random_seed,
            n_jobs=-1,
        ),
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(y, t, X=x)
    return model


def _causal_tree_rules(
    model: CausalForestDML,
    x: np.ndarray,
    feature_names: list[str],
    max_depth: int,
    min_samples_leaf: int,
) -> str:
    interpreter = SingleTreeCateInterpreter(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        include_model_uncertainty=False,
    )
    interpreter.interpret(model, x)
    return export_text(interpreter.tree_model_, feature_names=feature_names)


def estimate_effect(
    df: pd.DataFrame,
    treatment: str,
    random_seed: int,
    tree_max_depth: int,
    tree_min_samples_leaf: int,
    n_estimators: int,
    model_n_estimators: int,
    forest_min_samples_leaf: int,
    standardize: bool,
    outcome: str = "MEDIA_CANDIDATO",
) -> dict[str, Any] | None:

    if outcome not in df.columns:
        print(f"[WARN] Outcome '{outcome}' ausente. Pulando tratamento={treatment}.")
        return None
    if treatment not in df.columns:
        print(f"[WARN] Tratamento '{treatment}' ausente. Pulando.")
        return None

    x_cols = [c for c in FEATURES if c in df.columns and c != treatment]
    if not x_cols:
        print(f"[WARN] Sem features para tratamento={treatment}. Pulando.")
        return None

    scale_meta = {}
    if standardize:
        df, scale_meta = _standardize(df, treatment, outcome, x_cols)
        print(
            f"[INFO] ATE será em desvios-padrão de '{outcome}' "
            f"por desvio-padrão de '{treatment}'."
        )

    x = df[x_cols].fillna(0).to_numpy()
    t = df[treatment].astype(float).to_numpy()
    y = df[outcome].astype(float).to_numpy()

    model = _fit_causal_model(
        y=y, t=t, x=x,
        random_seed=random_seed,
        n_estimators=n_estimators,
        model_n_estimators=model_n_estimators,
        min_samples_leaf=forest_min_samples_leaf,
    )
    effects = model.effect(x)

    rules = _causal_tree_rules(
        model, x=x,
        feature_names=x_cols,
        max_depth=tree_max_depth,
        min_samples_leaf=tree_min_samples_leaf,
    )

    # Intervalo de confiança do ATE
    ate_inf, ate_point, ate_sup = model.ate_interval(x, alpha=0.05)

    result = {
        "treatment":           treatment,
        "outcome":             outcome,
        "standardized":        standardize,
        "scale_meta":          scale_meta,
        "features_used":       x_cols,
        "tree_type":           "SingleTreeCateInterpreter",
        "tree_max_depth":      tree_max_depth,
        "tree_min_samples_leaf": tree_min_samples_leaf,
        # ATE e IC 95%
        "ate":                 float(np.mean(effects)),
        "ate_ci_lower":        float(ate_inf),
        "ate_ci_upper":        float(ate_sup),
        "effect_std":          float(np.std(effects)),
        "effect_p10":          float(np.quantile(effects, 0.10)),
        "effect_p50":          float(np.quantile(effects, 0.50)),
        "effect_p90":          float(np.quantile(effects, 0.90)),
        "n":                   int(len(df)),
        "tree_rules":          rules,
    }

    # Interpretação em linguagem natural
    if standardize and scale_meta:
        t_std = scale_meta.get(treatment, {}).get("std", 1)
        y_std = scale_meta.get(outcome,   {}).get("std", 1)
        ate_original = float(np.mean(effects)) * y_std / t_std if t_std else None
        result["ate_interpretation"] = (
            f"Um aumento de 1 desvio-padrão em {treatment} "
            f"(≈ {t_std:.2f} unidades originais) causa em média "
            f"{float(np.mean(effects)):.3f} desvios-padrão em {outcome} "
            f"(≈ {ate_original:.2f} pontos no ENEM)."
        )

    return result


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    year                 = cfg["year"]
    sample_size          = cfg.get("sample_size", 100_000)
    seed                 = cfg.get("random_seed", 69)
    tree_max_depth       = cfg.get("tree_max_depth", 3)
    tree_min_samples     = cfg.get("tree_min_samples_leaf", 500)
    n_estimators         = cfg.get("n_estimators", 300)
    model_n_estimators   = cfg.get("model_n_estimators", 200)
    forest_min_samples   = cfg.get("forest_min_samples_leaf", 20)
    treatments           = cfg.get("treatments", ["RENDA", "SCORE_CULT_PAIS"])
    standardize          = cfg.get("standardize_treatment", True)
    bucket               = cfg["gcs"]["bucket"]

    source_blob = f"processed/enem_{year}/dados_enem_processados_{year}.parquet"
    local_file  = Path("tmp") / f"dados_enem_processados_{year}.parquet"
    local_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Baixando gs://{bucket}/{source_blob}...")
    download_file(bucket, source_blob, local_file)

    df = pd.read_parquet(local_file)
    print(f"[INFO] Carregado: {len(df)} linhas | colunas: {list(df.columns)}")

    if len(df) == 0:
        raise RuntimeError(f"Parquet processed/enem_{year} está vazio.")

    # Amostragem estratificada por TP_COR_RACA
    if len(df) > sample_size:
        strat = "TP_COR_RACA" if "TP_COR_RACA" in df.columns else None
        if strat:
            df = (
                df.groupby(strat, group_keys=False)
                .apply(lambda g: g.sample(
                    min(len(g), max(1, int(sample_size * len(g) / len(df)))),
                    random_state=seed,
                ))
            )
        else:
            df = df.sample(sample_size, random_state=seed)

    print(f"[INFO] Amostra: {len(df)} linhas | standardize={standardize}")

    effects = []
    for treatment in treatments:
        print(f"\n[INFO] Estimando efeito causal: treatment={treatment}")
        result = estimate_effect(
            df=df.copy(),  # copy para que cada tratamento tenha sua própria escala
            treatment=treatment,
            random_seed=seed,
            tree_max_depth=tree_max_depth,
            tree_min_samples_leaf=tree_min_samples,
            n_estimators=n_estimators,
            model_n_estimators=model_n_estimators,
            forest_min_samples_leaf=forest_min_samples,
            standardize=standardize,
        )
        if result is not None:
            effects.append(result)
            print(f"[OK] ATE={result['ate']:.4f} | IC95=[{result['ate_ci_lower']:.4f}, {result['ate_ci_upper']:.4f}]")
            if "ate_interpretation" in result:
                print(f"[OK] {result['ate_interpretation']}")

    results = {
        "year":             year,
        "sample_size_used": int(len(df)),
        "standardized":     standardize,
        "effects":          effects,
    }

    out_dir  = Path(cfg.get("output_dir", "results/causal"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"causal_effects_{year}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Resultados salvos em {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/causal_trees.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()