from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import yaml

from scripts.enem_pipeline.gcs_utils import download_file

RENDA_CENSORED_VALUE = 25425  # teto declaratório, mantido para referência nos logs


def _has_min_levels(df: pd.DataFrame, col: str, min_levels: int = 2) -> bool:
    n = df[col].dropna().nunique()
    if n < min_levels:
        print(f"[WARN] '{col}' tem {n} nível(is) único(s) — excluída da fórmula.")
    return n >= min_levels


def _build_formula(df: pd.DataFrame) -> str | None:
    outcome = "MEDIA_CANDIDATO"
    if outcome not in df.columns:
        return None

    terms = []
    for c in ["RENDA", "SCORE_CULT_PAIS"]:
        if c in df.columns and df[c].notna().any():
            terms.append(c)
    if "TP_FAIXA_ETARIA" in df.columns and df["TP_FAIXA_ETARIA"].notna().any():
        terms.append("bs(TP_FAIXA_ETARIA, df=5)")
    for c in ["TP_COR_RACA", "INTERNET", "TP_ESCOLA"]:
        if c in df.columns and _has_min_levels(df, c):
            terms.append(f"C({c})")

    if not terms:
        return None
    return f"{outcome} ~ {' + '.join(terms)}"



def _fit_scaler(df: pd.DataFrame, scale_cols: list[str]) -> tuple[StandardScaler, dict]:
    """
    Ajusta o scaler na amostra completa (z-score clássico).
    Para RENDA, salva também o std robusto (IQR / 1.3489) em R$ originais,
    usado como denominador do fator ate_per_1k_factor — imune à cauda
    censurada em 25425 por definição.
    """
    scaler = StandardScaler()
    scaler.fit(df[scale_cols])

    scale_meta = {
        col: {"mean": float(scaler.mean_[i]), "std": float(scaler.scale_[i])}
        for i, col in enumerate(scale_cols)
    }

    if "RENDA" in scale_meta:
        std_r  = scale_meta["RENDA"]["std"]
        # IQR calculado nos dados originais (antes do z-score)
        iqr            = float(df["RENDA"].quantile(0.75) - df["RENDA"].quantile(0.25))
        std_robust_brl = (iqr / 1.3489)
        scale_meta["RENDA"]["std_robust_brl"]    = round(std_robust_brl, 2)
        scale_meta["RENDA"]["ate_per_1k_factor"] = round(1000 / std_robust_brl, 6)

    return scaler, scale_meta


def _apply_scaler(
    df: pd.DataFrame, scaler: StandardScaler, scale_cols: list[str]
) -> pd.DataFrame:
    df = df.copy()
    df[scale_cols] = scaler.transform(df[scale_cols])
    return df


def _backtransform_coefs(
    coefs: dict[str, float],
    pvalues: dict[str, float],
    scale_meta: dict,
    outcome: str,
) -> dict[str, dict]:
    """
    Reconverte coeficientes padronizados para escala original.

    Para variáveis contínuas padronizadas:
      β_original = β_padronizado × (std_Y / std_X)
      → em pontos de ENEM por unidade original de X

    Para RENDA especificamente, adiciona também a escala por R$1.000:
      β_per_1k = β_original × 1000

    Variáveis categóricas (C(...)) e o intercepto ficam em escala
    de σ_Y (diferença esperada em desvios-padrão da nota).
    """
    std_y   = scale_meta[outcome]["std"]
    result  = {}

    for name, b in coefs.items():
        entry = {
            "coef_std":  round(b, 6),
            "pvalue":    round(pvalues.get(name, float("nan")), 6),
        }

        # contínuas padronizadas que conhecemos
        for col in ["RENDA", "SCORE_CULT_PAIS"]:
            if name == col and col in scale_meta:
                std_x = scale_meta[col]["std"]
                b_orig = b * std_y / std_x
                entry["coef_original_units"] = round(b_orig, 6)
                entry["original_units_label"] = f"pontos ENEM por 1 unidade de {col}"
                if col == "RENDA":
                    entry["coef_per_1k_brl"] = round(b_orig * 1000, 4)
                    entry["per_1k_label"]    = "pontos ENEM por R$1.000 de renda"
                break

        result[name] = entry

    return result


def run(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    year                   = cfg["year"]
    sample_size_rlm = cfg.get("sample_size_rlm", 2000)
    seed            = cfg.get("random_seed", 69)
    bucket                 = cfg["gcs"]["bucket"]

    source_blob = f"processed/enem_{year}/dados_enem_processados_{year}.parquet"
    local_file  = Path("tmp") / f"dados_enem_processados_{year}.parquet"
    local_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Baixando gs://{bucket}/{source_blob}...")
    download_file(bucket, source_blob, local_file)

    df = pd.read_parquet(local_file)
    print(f"[INFO] Carregado: {len(df)} linhas | colunas: {list(df.columns)}")

    if len(df) == 0:
        raise RuntimeError(f"Parquet enem_{year} está vazio.")

    scale_cols = [c for c in ["RENDA", "SCORE_CULT_PAIS", "MEDIA_CANDIDATO"]
                  if c in df.columns]

    scaler, scale_meta = _fit_scaler(df, scale_cols)
    print(f"[INFO] Scaler ajustado em {len(df):,} obs | RENDA teto declaratório: {RENDA_CENSORED_VALUE}")
    for col, m in scale_meta.items():
        extra = f" | std_robusto={m.get('std_robust_brl','—')} | fator por R$1k = {m.get('ate_per_1k_factor','—')}" if col == "RENDA" else ""
        print(f"       {col}: mean={m['mean']:.2f}, std={m['std']:.2f}{extra}")

    strat = "TP_COR_RACA" if "TP_COR_RACA" in df.columns else None

    def _sample(target_size: int) -> pd.DataFrame:
        if len(df) <= target_size:
            return df.copy()
        if strat:
            return (
                df.groupby(strat, group_keys=False)
                .apply(lambda g: g.sample(
                    min(len(g), max(1, int(target_size * len(g) / len(df)))),
                    random_state=seed,
                ))
            )
        return df.sample(target_size, random_state=seed)

    
    df_rlm = _apply_scaler(_sample(sample_size_rlm), scaler, scale_cols)

    print(f"[INFO] RLM: {len(df_rlm)} obs")

    results: dict = {
        "year": year,
        "scale_meta": scale_meta,
        "sample_size_used": {
            "full": int(len(df)),
            "rlm":  int(len(df_rlm)),
        },
    }

    # RLM
    formula = _build_formula(df_rlm)
    if formula:
        print(f"[INFO] RLM: {formula}")
        rlm_fit = smf.rlm(formula, data=df_rlm).fit()
        coefs_bt = _backtransform_coefs(
            rlm_fit.params.to_dict(),
            rlm_fit.pvalues.to_dict(),
            scale_meta,
            outcome="MEDIA_CANDIDATO",
        )
        results["rlm"] = {
            "formula":      formula,
            "estimator":    "RLM (Huber M-estimator)",
            "scale":        float(rlm_fit.scale),
            "coefficients": coefs_bt,
        }
    else:
        print("[WARN] Colunas insuficientes para RLM.")
        results["rlm"] = None

    out_dir  = Path(cfg.get("output_dir", "results/statistical"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"statistical_tests_{year}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Resultados salvos em {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/statistical_tests.yml", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()