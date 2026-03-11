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


def _build_interpreter(
    model: CausalForestDML,
    x: np.ndarray,
    max_depth: int,
    min_samples_leaf: int,
) -> SingleTreeCateInterpreter:
    """Treina e retorna o interpretador de árvore única."""
    interpreter = SingleTreeCateInterpreter(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        include_model_uncertainty=False,
    )
    interpreter.interpret(model, x)
    return interpreter


def _causal_tree_rules(
    interpreter: SingleTreeCateInterpreter,
    feature_names: list[str],
) -> str:
    return export_text(interpreter.tree_model_, feature_names=feature_names)


def _cate_leaf_stats(
    model: CausalForestDML,
    interpreter: SingleTreeCateInterpreter,
    x: np.ndarray,
) -> list[dict]:
    """
    Para cada folha da árvore interpretativa, calcula estatísticas dos
    efeitos individuais estimados pelo Causal Forest:

      - n            : observações na folha
      - ate          : ATE médio da folha (média dos τ̂(x_i))
      - std          : desvio-padrão dos efeitos individuais dentro da folha
      - se           : erro-padrão do ATE da folha (std / √n)
      - ci_lower/upper: IC 95% do ATE da folha
      - cv           : coeficiente de variação (std / |ate|)
                       > 1.0 → heterogeneidade interna alta, folha candidata
                               a mais particionamento
                       < 0.5 → folha relativamente homogênea

    Ordenado por ATE decrescente (grupo de maior efeito primeiro).
    """
    # efeitos individuais do forest (τ̂ por observação)
    effects = model.effect(x)

    # índice de folha para cada observação segundo a árvore interpretativa
    leaf_ids = interpreter.tree_model_.apply(x)

    stats = []
    for leaf_id in np.unique(leaf_ids):
        mask = leaf_ids == leaf_id
        leaf_effects = effects[mask]
        n   = int(mask.sum())
        ate = float(leaf_effects.mean())
        std = float(leaf_effects.std())
        se  = std / np.sqrt(n) if n > 1 else float("nan")

        stats.append({
            "leaf_id":   int(leaf_id),
            "n":         n,
            "ate":       round(ate, 5),
            "std":       round(std, 5),
            "se":        round(se, 5),
            "ci_lower":  round(ate - 1.96 * se, 5),
            "ci_upper":  round(ate + 1.96 * se, 5),
            # cv > 1 → std maior que o ATE médio: heterogeneidade interna alta
            "cv":        round(std / abs(ate), 4) if ate != 0 else None,
        })

    return sorted(stats, key=lambda s: s["ate"], reverse=True)


def _refine_high_cv_leaves(
    model: CausalForestDML,
    interpreter: SingleTreeCateInterpreter,
    x: np.ndarray,
    df_features: pd.DataFrame,
    feature_names: list[str],
    tree_min_samples_leaf: int,
    cv_threshold: float = 0.5,
    standardize_meta: dict | None = None,
) -> list[dict]:
    """
    Refinamento adaptativo das folhas com CV > cv_threshold.

    Para cada folha da árvore base com coeficiente de variação alto
    (std / |ate| > cv_threshold), treina uma sub-árvore local com
    max_depth=1 usando apenas as observações daquela folha.

    Isso equivale a um split adicional localizado, sem alterar a árvore
    global (que permanece com tree_max_depth=3).

    Retorna lista unificada de folhas (base + sub-folhas refinadas),
    ordenada por ATE decrescente. Folhas com CV <= cv_threshold são
    mantidas intactas. Folhas refinadas recebem o campo
    `refined=True` e `parent_leaf_id`.
    """
    effects  = model.effect(x)
    leaf_ids = interpreter.tree_model_.apply(x)

    unified: list[dict] = []

    for leaf_id in np.unique(leaf_ids):
        mask         = leaf_ids == leaf_id
        leaf_effects = effects[mask]
        n_leaf       = int(mask.sum())
        ate          = float(leaf_effects.mean())
        std          = float(leaf_effects.std())
        se           = std / np.sqrt(n_leaf) if n_leaf > 1 else float("nan")
        cv           = std / abs(ate) if ate != 0 else None

        base_stat = {
            "leaf_id":   int(leaf_id),
            "n":         n_leaf,
            "ate":       round(ate, 5),
            "std":       round(std, 5),
            "se":        round(se, 5),
            "ci_lower":  round(ate - 1.96 * se, 5),
            "ci_upper":  round(ate + 1.96 * se, 5),
            "cv":        round(cv, 4) if cv is not None else None,
            "refined":   False,
        }

        # se CV ok, a gente  mantém folha original
        if cv is None or cv <= cv_threshold or n_leaf < tree_min_samples_leaf * 2:
            unified.append(base_stat)
            continue

        #sub-árvore local com max_depth=1 
        x_leaf      = x[mask]
        eff_leaf    = leaf_effects

        # min_samples_leaf da sub-árvore: metade do mínimo global, mas >= 100
        sub_min_leaf = max(100, tree_min_samples_leaf // 2)

        sub_interp = SingleTreeCateInterpreter(
            max_depth=1,
            min_samples_leaf=sub_min_leaf,
            include_model_uncertainty=False,
        )
        try:
            sub_interp.interpret(model, x_leaf)
            sub_leaf_ids = sub_interp.tree_model_.apply(x_leaf)
        except Exception as exc:
            print(f"       [WARN] Refinamento da folha {leaf_id} falhou ({exc}). Mantendo original.")
            unified.append(base_stat)
            continue

        n_sub_leaves = len(np.unique(sub_leaf_ids))

        # se a sub-árvore não conseguiu dividir, mantém original
        if n_sub_leaves <= 1:
            print(f"       [WARN] Folha {leaf_id}: sub-árvore não dividiu (n={n_leaf}). Mantendo.")
            unified.append(base_stat)
            continue

        print(f"       [REFINE] Folha {leaf_id} (cv={cv:.3f}) → {n_sub_leaves} sub-folhas")
        sub_rules = export_text(sub_interp.tree_model_, feature_names=feature_names)

        for sub_id in np.unique(sub_leaf_ids):
            sub_mask    = sub_leaf_ids == sub_id
            sub_effects = eff_leaf[sub_mask]
            ns          = int(sub_mask.sum())
            s_ate       = float(sub_effects.mean())
            s_std       = float(sub_effects.std())
            s_se        = s_std / np.sqrt(ns) if ns > 1 else float("nan")
            s_cv        = s_std / abs(s_ate) if s_ate != 0 else None

            flag = "  cv>1" if s_cv and s_cv > 1 else ""
            print(
                f"         sub-folha {sub_id} | n={ns:>7,} | "
                f"ate={s_ate:+.4f} | std={s_std:.4f} | cv={s_cv:.3f}{flag}"
            )

            unified.append({
                "leaf_id":        int(leaf_id) * 100 + int(sub_id),  # id único
                "parent_leaf_id": int(leaf_id),
                "sub_leaf_id":    int(sub_id),
                "n":              ns,
                "ate":            round(s_ate, 5),
                "std":            round(s_std, 5),
                "se":             round(s_se,  5),
                "ci_lower":       round(s_ate - 1.96 * s_se, 5),
                "ci_upper":       round(s_ate + 1.96 * s_se, 5),
                "cv":             round(s_cv, 4) if s_cv is not None else None,
                "refined":        True,
                "sub_tree_rules": sub_rules,
            })

    return sorted(unified, key=lambda s: s["ate"], reverse=True)



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
    cfg_adaptive_cv: float | None = 0.5,
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

    
    interpreter = _build_interpreter(
        model=model,
        x=x,
        max_depth=tree_max_depth,
        min_samples_leaf=tree_min_samples_leaf,
    )
    rules = _causal_tree_rules(interpreter, feature_names=x_cols)

    # Refinamento adaptativo: só para RENDA, folhas com cv > 0.5 recebem
    # um split adicional via sub-árvore local (max_depth=1).
    # SCORE_CULT_PAIS usa _cate_leaf_stats padrão (sem refinamento)
    #Por que isso? Bom, durante a rodagem de testes, percebi que a heterogeneidade em grupos dessa variável era muito grande.
    #Enquanto que para SCORE_CULT_PAIS, a heterogeneidade intra-grupal é bem menor. 
    cv_threshold = cfg_adaptive_cv if treatment == "RENDA" else None

    if cv_threshold is not None:
        print(f"[INFO] Refinamento adaptativo ativado para {treatment} (cv_threshold={cv_threshold})")
        leaf_stats = _refine_high_cv_leaves(
            model=model,
            interpreter=interpreter,
            x=x,
            df_features=df[x_cols],
            feature_names=x_cols,
            tree_min_samples_leaf=tree_min_samples_leaf,
            cv_threshold=cv_threshold,
            standardize_meta=scale_meta if standardize else None,
        )
    else:
        leaf_stats = _cate_leaf_stats(model, interpreter, x)

    
    print(f"[INFO] Folhas da árvore ({treatment}): {len(leaf_stats)} grupos")
    for ls in leaf_stats:
        flag = " cv>1 (heterogeneidade interna alta)" if ls["cv"] and ls["cv"] > 1 else ""
        refined_tag = " [REFINADA]" if ls.get("refined") else ""
        print(
            f"       folha {ls['leaf_id']:>5} | n={ls['n']:>7,} | "
            f"ate={ls['ate']:+.4f} | std={ls['std']:.4f} | "
            f"cv={ls['cv']:.3f}{flag}{refined_tag}"
        )

    
    ate_inf, ate_sup = model.ate_interval(x, alpha=0.05)

    result = {
        "treatment":             treatment,
        "outcome":               outcome,
        "standardized":          standardize,
        "scale_meta":            scale_meta,
        "features_used":         x_cols,
        "tree_type":             "SingleTreeCateInterpreter",
        "tree_max_depth":        tree_max_depth,
        "tree_min_samples_leaf": tree_min_samples_leaf,
        # ATE global
        "ate":                   float(np.mean(effects)),
        "ate_ci_lower":          float(ate_inf),
        "ate_ci_upper":          float(ate_sup),
        # Distribuição dos efeitos individuais (amostra inteira)
        "effect_std":            float(np.std(effects)),
        "effect_p10":            float(np.quantile(effects, 0.10)),
        "effect_p50":            float(np.quantile(effects, 0.50)),
        "effect_p90":            float(np.quantile(effects, 0.90)),
        "n":                     int(len(df)),
        # Árvore + estatísticas por folha
        "tree_rules":            rules,
        "adaptive_refinement":   cv_threshold is not None,
        "adaptive_cv_threshold": cv_threshold,
        "leaf_stats":            leaf_stats,
    }

    # Interpretação em linguagem natural
    if standardize and scale_meta:
        t_std      = scale_meta.get(treatment, {}).get("std", 1)
        y_std      = scale_meta.get(outcome,   {}).get("std", 1)
        ate_std    = float(np.mean(effects))
        ate_points = ate_std * y_std

        result["ate_interpretation"] = (
            f"Um aumento de 1 desvio-padrão em {treatment} "
            f"(≈ R$ {t_std:,.0f}) está associado a um aumento médio de "
            f"{ate_points:.1f} pontos no ENEM "
            f"({ate_std:.3f} desvios-padrão de {outcome})."
        )

        # interpretação por folha em pontos do ENEM
        for ls in result["leaf_stats"]:
            ls["ate_pts"]      = round(ls["ate"]      * y_std, 2)
            ls["std_pts"]      = round(ls["std"]      * y_std, 2)
            ls["ci_lower_pts"] = round(ls["ci_lower"] * y_std, 2)
            ls["ci_upper_pts"] = round(ls["ci_upper"] * y_std, 2)

    return result


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    year               = cfg["year"]
    sample_size        = cfg.get("sample_size", 100_000)
    seed               = cfg.get("random_seed", 69)
    tree_max_depth     = cfg.get("tree_max_depth", 4)
    tree_min_samples   = cfg.get("tree_min_samples_leaf", 200)
    n_estimators       = cfg.get("n_estimators", 2000)
    model_n_estimators = cfg.get("model_n_estimators", 500)
    forest_min_samples = cfg.get("forest_min_samples_leaf", 150)
    treatments         = cfg.get("treatments", ["RENDA", "SCORE_CULT_PAIS"])
    standardize        = cfg.get("standardize_treatment", True)
    # adaptive_cv: threshold de cv para refinamento adaptativo (só RENDA).
    # None desabilita o refinamento. Default 0.5 (folhas com cv>0.5 são subdivididas).
    adaptive_cv        = cfg.get("adaptive_cv_threshold", 0.5)
    bucket             = cfg["gcs"]["bucket"]

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

    all_effects = []
    for treatment in treatments:
        print(f"\n[INFO] ── Estimando efeito causal: treatment={treatment} ──")
        result = estimate_effect(
            df=df.copy(),
            treatment=treatment,
            random_seed=seed,
            tree_max_depth=tree_max_depth,
            tree_min_samples_leaf=tree_min_samples,
            n_estimators=n_estimators,
            model_n_estimators=model_n_estimators,
            forest_min_samples_leaf=forest_min_samples,
            standardize=standardize,
            cfg_adaptive_cv=adaptive_cv,
        )
        if result is not None:
            all_effects.append(result)
            print(f"[OK] ATE={result['ate']:.4f} | IC95=[{result['ate_ci_lower']:.4f}, {result['ate_ci_upper']:.4f}] | effect_std={result['effect_std']:.4f}")
            if "ate_interpretation" in result:
                print(f"[OK] {result['ate_interpretation']}")

    output = {
        "year":             year,
        "sample_size_used": int(len(df)),
        "standardized":     standardize,
        "effects":          all_effects,
    }

    out_dir  = Path(cfg.get("output_dir", "results/causal"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"causal_effects_{year}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Resultados salvos em {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/causal_trees.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()