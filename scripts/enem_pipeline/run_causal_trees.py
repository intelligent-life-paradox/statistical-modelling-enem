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



# O valor 25425 é um TETO DECLARATÓRIO: candidatos com renda real acima de
# 20k estão todos codificados nele. Para o fator de escala (ate_per_1k),
# usamos o desvio-padrão robusto baseado no IQR (÷ 1.3489), que é imune
# à cauda por definição — sem necessidade de definir um teto arbitrário.
RENDA_CLASSES_MIDPOINTS = [
    0, 7, 499, 1247, 1746, 2245, 2744, 2798,
    3493, 4194, 4491, 5105, 5489, 6487,
    9481, 10978, 13473, 17465,
    25425,  
]   

FEATURES = [
    "RENDA",
    "INTERNET",
    "TP_SEXO",
    "TP_COR_RACA",
    "TP_ESCOLA",
    "TP_DEPENDENCIA_ADM_ESC",
    "TP_LOCALIZACAO_ESC",
    "TP_SIT_FUNC_ESC",
    "SCORE_CULT_PAIS",   
]



def _standardize(
    df: pd.DataFrame,
    outcome: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Padroniza RENDA e o outcome.

    RENDA: z-score clássico para o forest. O fator ate_per_1k_factor usa
    o desvio-padrão robusto (IQR / 1.3489), imune à cauda censurada em
    25425 por definição — sem necessidade de definir um teto arbitrário.
    """
    df   = df.copy()
    meta = {}

    mu_y, std_y = df[outcome].mean(), df[outcome].std()
    df[outcome] = (df[outcome] - mu_y) / std_y
    meta[outcome] = {"mean": float(mu_y), "std": float(std_y)}
    print(f"[INFO] Padronizado '{outcome}': média={mu_y:.2f}, std={std_y:.2f}")

    mu_r  = float(df["RENDA"].mean())
    std_r = float(df["RENDA"].std())
    df["RENDA"] = (df["RENDA"] - mu_r) / std_r

    iqr            = float(df["RENDA"].quantile(0.75) - df["RENDA"].quantile(0.25))
    std_robust     = iqr / 1.3489
    std_robust_brl = std_robust * std_r  # de volta em R$

    meta["RENDA"] = {
        "mean":              mu_r,
        "std":               std_r,
        "std_robust_brl":    round(std_robust_brl, 2),
        "ate_per_1k_factor": round(1000 / std_robust_brl, 6),
    }
    print(
        f"[INFO] RENDA: média={mu_r:.0f} | std={std_r:.0f} | "
        f"std_robusto={std_robust_brl:.0f} | fator por R$1k = {1000/std_robust_brl:.5f}σ"
    )
    return df, meta


# ── Causal Forest ─────────────────────────────────────────────────────────────
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
    interp = SingleTreeCateInterpreter(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        include_model_uncertainty=False,
    )
    interp.interpret(model, x)
    return interp


def _causal_tree_rules(interp: SingleTreeCateInterpreter, feature_names: list[str]) -> str:
    return export_text(interp.tree_model_, feature_names=feature_names)


# Estatísticas por folh
def _leaf_stats_raw(
    effects: np.ndarray,
    leaf_ids: np.ndarray,
    refined: bool = False,
    parent_leaf_id: int | None = None,
) -> list[dict]:
    stats = []
    for lid in np.unique(leaf_ids):
        mask = leaf_ids == lid
        eff  = effects[mask]
        n    = int(mask.sum())
        ate  = float(eff.mean())
        std  = float(eff.std())
        se   = std / np.sqrt(n) if n > 1 else float("nan")
        cv   = std / abs(ate)   if ate != 0 else None
        entry = {
            "leaf_id":  int(lid),
            "n":        n,
            "ate":      round(ate, 5),
            "std":      round(std, 5),
            "se":       round(se,  5),
            "ci_lower": round(ate - 1.96 * se, 5),
            "ci_upper": round(ate + 1.96 * se, 5),
            "cv":       round(cv, 4) if cv is not None else None,
            "refined":  refined,
        }
        if parent_leaf_id is not None:
            entry["parent_leaf_id"] = parent_leaf_id
        stats.append(entry)
    return sorted(stats, key=lambda s: s["ate"], reverse=True)


# Refinamento adaptativo 
def _refine_high_cv_leaves(
    model: CausalForestDML,
    interpreter: SingleTreeCateInterpreter,
    x: np.ndarray,
    feature_names: list[str],
    tree_min_samples_leaf: int,
    cv_threshold: float = 0.5,
) -> list[dict]:
    """
    Para cada folha da árvore base com cv > cv_threshold, treina uma
    sub-árvore local (max_depth=1) usando apenas as obs daquela folha.
    Isso gera um split localizado sem alterar a árvore global.

    Folhas com cv <= cv_threshold são mantidas intactas (refined=False).
    Folhas refinadas recebem refined=True e parent_leaf_id.
    """
    effects  = model.effect(x)
    leaf_ids = interpreter.tree_model_.apply(x)
    unified  = []

    for lid in np.unique(leaf_ids):
        mask = leaf_ids == lid
        eff  = effects[mask]
        n    = int(mask.sum())
        ate  = float(eff.mean())
        std  = float(eff.std())
        se   = std / np.sqrt(n) if n > 1 else float("nan")
        cv   = std / abs(ate)   if ate != 0 else None

        base = {
            "leaf_id":  int(lid),
            "n":        n,
            "ate":      round(ate, 5),
            "std":      round(std, 5),
            "se":       round(se,  5),
            "ci_lower": round(ate - 1.96 * se, 5),
            "ci_upper": round(ate + 1.96 * se, 5),
            "cv":       round(cv, 4) if cv is not None else None,
            "refined":  False,
        }

        # folha ok ou pequena demais para subdividir, então mantém
        if cv is None or cv <= cv_threshold or n < tree_min_samples_leaf * 2:
            unified.append(base)
            continue

        x_leaf       = x[mask]
        sub_min_leaf = max(100, tree_min_samples_leaf // 2)

        sub_interp = SingleTreeCateInterpreter(
            max_depth=1,
            min_samples_leaf=sub_min_leaf,
            include_model_uncertainty=False,
        )
        try:
            sub_interp.interpret(model, x_leaf)
            sub_ids = sub_interp.tree_model_.apply(x_leaf)
        except Exception as exc:
            print(f"       [WARN] Refinamento folha {lid} falhou ({exc}). Mantendo.")
            unified.append(base)
            continue

        if len(np.unique(sub_ids)) <= 1:
            print(f"       [WARN] Folha {lid}: sub-árvore não dividiu (n={n}). Mantendo.")
            unified.append(base)
            continue

        sub_rules = export_text(sub_interp.tree_model_, feature_names=feature_names)
        print(f"       [REFINE] Folha {lid} (cv={cv:.3f}) → {len(np.unique(sub_ids))} sub-folhas")
        print(f"         Regra: {sub_rules.strip()[:120]}")

        for sub_id in np.unique(sub_ids):
            sm   = sub_ids == sub_id
            ns   = int(sm.sum())
            sate = float(eff[sm].mean())
            sstd = float(eff[sm].std())
            sse  = sstd / np.sqrt(ns) if ns > 1 else float("nan")
            scv  = sstd / abs(sate)   if sate != 0 else None
            flag = "  cv>1" if scv and scv > 1 else ""
            print(
                f"         sub-folha {sub_id} | n={ns:>7,} | "
                f"ate={sate:+.4f} | std={sstd:.4f} | cv={scv:.3f}{flag}"
            )
            unified.append({
                "leaf_id":        int(lid) * 100 + int(sub_id),
                "parent_leaf_id": int(lid),
                "sub_leaf_id":    int(sub_id),
                "n":              ns,
                "ate":            round(sate, 5),
                "std":            round(sstd, 5),
                "se":             round(sse,  5),
                "ci_lower":       round(sate - 1.96 * sse, 5),
                "ci_upper":       round(sate + 1.96 * sse, 5),
                "cv":             round(scv, 4) if scv is not None else None,
                "refined":        True,
                "sub_tree_rules": sub_rules,
            })

    return sorted(unified, key=lambda s: s["ate"], reverse=True)



def estimate_effect(
    df: pd.DataFrame,
    random_seed: int,
    tree_max_depth: int,
    tree_min_samples_leaf: int,
    n_estimators: int,
    model_n_estimators: int,
    forest_min_samples_leaf: int,
    adaptive_cv_threshold: float | None = 0.5,
    outcome: str = "MEDIA_CANDIDATO",
) -> dict[str, Any] | None:
    """
    Estima o efeito causal de RENDA sobre MEDIA_CANDIDATO.

    SCORE_CULT_PAIS permanece como feature de controle no vetor X,
    mas não é mais estimado como tratamento separado.

    ATE reportado em duas escalas:
      - ate           : σ_ENEM por 1σ_renda (escala padronizada)
      - ate_per_1k_*  : σ_ENEM (e pontos) por R$1.000 de renda
    """
    treatment = "RENDA"

    if outcome not in df.columns:
        print(f"[WARN] Outcome '{outcome}' ausente. Pulando.")
        return None
    if treatment not in df.columns:
        print(f"[WARN] Tratamento '{treatment}' ausente. Pulando.")
        return None

    x_cols = [c for c in FEATURES if c in df.columns and c != treatment]

    df, scale_meta = _standardize(df, outcome)

    x = df[x_cols].fillna(0).to_numpy()
    t = df[treatment].astype(float).to_numpy()
    y = df[outcome].astype(float).to_numpy()

    std_r             = scale_meta["RENDA"]["std"]
    std_y             = scale_meta[outcome]["std"]
    ate_per_1k_factor = scale_meta["RENDA"]["ate_per_1k_factor"]  # 1000 / std_r

    # Causal Forest
    model   = _fit_causal_model(
        y=y, t=t, x=x,
        random_seed=random_seed,
        n_estimators=n_estimators,
        model_n_estimators=model_n_estimators,
        min_samples_leaf=forest_min_samples_leaf,
    )
    effects = model.effect(x)

    # Árvore interpretativa
    interp = _build_interpreter(
        model=model, x=x,
        max_depth=tree_max_depth,
        min_samples_leaf=tree_min_samples_leaf,
    )
    rules = _causal_tree_rules(interp, feature_names=x_cols)

    # Leaf stats com refinamento adaptativo
    if adaptive_cv_threshold is not None:
        print(f"[INFO] Refinamento adaptativo RENDA (cv_threshold={adaptive_cv_threshold})")
        leaf_stats = _refine_high_cv_leaves(
            model=model,
            interpreter=interp,
            x=x,
            feature_names=x_cols,
            tree_min_samples_leaf=tree_min_samples_leaf,
            cv_threshold=adaptive_cv_threshold,
        )
    else:
        leaf_ids   = interp.tree_model_.apply(x)
        leaf_stats = _leaf_stats_raw(effects, leaf_ids)

    
    print(f"[INFO] Folhas RENDA: {len(leaf_stats)} grupos")
    for ls in leaf_stats:
        flag = "  cv>1" if ls["cv"] and ls["cv"] > 1 else ""
        ref  = " [R]"    if ls.get("refined")          else ""
        print(
            f"       folha {ls['leaf_id']:>5} | n={ls['n']:>7,} | "
            f"ate={ls['ate']:+.4f} | std={ls['std']:.4f} | "
            f"cv={ls['cv']:.3f}{flag}{ref}"
        )

    # ATE global
    ate_inf, ate_sup = model.ate_interval(x, alpha=0.05)
    ate_global       = float(np.mean(effects))

    result = {
        "treatment":             treatment,
        "outcome":               outcome,
        "standardized":          True,
        "scale_meta":            scale_meta,
        "features_used":         x_cols,
        "tree_type":             "SingleTreeCateInterpreter",
        "tree_max_depth":        tree_max_depth,
        "tree_min_samples_leaf": tree_min_samples_leaf,
        "adaptive_refinement":   adaptive_cv_threshold is not None,
        "adaptive_cv_threshold": adaptive_cv_threshold,
        # ATE em σ_ENEM por 1σ_renda
        "ate":                   round(ate_global, 6),
        "ate_ci_lower":          round(float(ate_inf), 6),
        "ate_ci_upper":          round(float(ate_sup), 6),
        # ATE em σ_ENEM por R$1.000
        "ate_per_1k_renda":      round(ate_global * ate_per_1k_factor, 6),
        # ATE em pontos ENEM por R$1.000
        "ate_per_1k_renda_pts":  round(ate_global * ate_per_1k_factor * std_y, 4),
        # Distribuição dos efeitos individuais
        "effect_std":            round(float(np.std(effects)), 6),
        "effect_p10":            round(float(np.quantile(effects, 0.10)), 6),
        "effect_p50":            round(float(np.quantile(effects, 0.50)), 6),
        "effect_p90":            round(float(np.quantile(effects, 0.90)), 6),
        "n":                     int(len(df)),
        "tree_rules":            rules,
        "leaf_stats":            leaf_stats,
    }

    result["ate_interpretation"] = (
        f"Um aumento de R$1.000 na renda está associado a +{result['ate_per_1k_renda_pts']:.2f} pts "
        f"no ENEM ({result['ate_per_1k_renda']:.5f}σ). "
        f"ATE por 1σ de renda (≈ R${std_r:,.0f}): "
        f"{ate_global:.4f}σ  IC95=[{float(ate_inf):.4f}, {float(ate_sup):.4f}]."
    )

    # rescale por folha
    for ls in leaf_stats:
        ls["ate_per_1k_pts"]      = round(ls["ate"]      * ate_per_1k_factor * std_y, 3)
        ls["std_per_1k_pts"]      = round(ls["std"]      * ate_per_1k_factor * std_y, 3)
        ls["ci_lower_per_1k_pts"] = round(ls["ci_lower"] * ate_per_1k_factor * std_y, 3)
        ls["ci_upper_per_1k_pts"] = round(ls["ci_upper"] * ate_per_1k_factor * std_y, 3)
        # em pontos por 1σ_renda (para compatibilidade com versões anteriores)
        ls["ate_pts"]             = round(ls["ate"]      * std_y, 2)
        ls["std_pts"]             = round(ls["std"]      * std_y, 2)
        ls["ci_lower_pts"]        = round(ls["ci_lower"] * std_y, 2)
        ls["ci_upper_pts"]        = round(ls["ci_upper"] * std_y, 2)

    return result


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    year               = cfg["year"]
    sample_size        = cfg.get("sample_size", 200_000)
    seed               = cfg.get("random_seed", 69)
    tree_max_depth     = cfg.get("tree_max_depth", 3)
    tree_min_samples   = cfg.get("tree_min_samples_leaf", 200)
    n_estimators       = cfg.get("n_estimators", 2000)
    model_n_estimators = cfg.get("model_n_estimators", 500)
    forest_min_samples = cfg.get("forest_min_samples_leaf", 150)
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
        raise RuntimeError(f"Parquet enem_{year} está vazio.")

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

    print(f"[INFO] Amostra: {len(df)} linhas | adaptive_cv={adaptive_cv}")

    print("\n[INFO] ── Estimando efeito causal: treatment=RENDA ──")
    result = estimate_effect(
        df=df.copy(),
        random_seed=seed,
        tree_max_depth=tree_max_depth,
        tree_min_samples_leaf=tree_min_samples,
        n_estimators=n_estimators,
        model_n_estimators=model_n_estimators,
        forest_min_samples_leaf=forest_min_samples,
        adaptive_cv_threshold=adaptive_cv,
    )

    if result is None:
        raise RuntimeError("estimate_effect retornou None.")

    print(f"[OK] {result['ate_interpretation']}")

    output = {
        "year":             year,
        "sample_size_used": int(len(df)),
        "standardized":     True,
        "effects":          [result],
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