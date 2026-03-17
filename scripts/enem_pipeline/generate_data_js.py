"""
Gera docs/data.js a partir dos JSONs de resultado do pipeline.

Uso:
    python scripts/enem_pipeline/generate_data_js.py

Lê:
    docs/enem-metrics-{year}/causal/causal_effects_{year}.json
    docs/enem-metrics-{year}/statistical/statistical_tests_{year}.json

Escreve:
    docs/data.js
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

YEARS = [2015, 2016, 2017, 2018, 2019]
DOCS  = Path("docs")

YEAR_CONTEXTS = {
    2015: "Renda média R$2.974 — maior da série. ENEM consolida função SISU. Pool de candidatos ainda relativamente seletivo. Maior IC da série reflete heterogeneidade real.",
    2016: "Primeiro ano pós-recessão (PIB -3,5% em 2015). Renda média cai para R$2.723 — menor da série. ATE causal é o mais baixo (0,230σ). O RLM mostra β alto (0,261σ), possivelmente por variância residual após controles.",
    2017: "Recuperação econômica incipiente. Renda média R$2.757. O RLM mostra β mais baixo da série (0,154σ) — possivelmente porque controles absorvem mais variância. O DML captura efeito causal médio mais estável.",
    2018: "Maior média ENEM da série (531,8 pts) e maior desvio-padrão (83 pts) — sinal de polarização crescente. A codificação de TP_ESCOLA variou em 2018, o que dificulta comparação direta dos coeficientes de escola com outros anos.",
    2019: "Último ano da série. Renda média cai para R$2.685 — segunda mais baixa. ATE causal atinge máximo histórico (0,289σ ≈ 23,3 pts). A winsorização IQR pode estar influenciando mais neste ano dado o IC mais amplo.",
}

#
FEATURE_LABELS = {
    "INTERNET":               {True:  "sem internet",          False: "com internet"},
    "TP_SEXO":                {True:  "masculino",             False: "feminino"},
    "TP_COR_RACA":            {True:  "raça branca",           False: "raça não-branca"},
    "TP_DEPENDENCIA_ADM_ESC": {True:  "dep. pública",          False: "dep. federal/privada"},
    "TP_ESCOLA":              {True:  "escola pública",        False: "escola privada/federal"},
    "TP_LOCALIZACAO_ESC":     {True:  "zona rural",            False: "zona urbana"},
    "SCORE_CULT_PAIS":        {True:  "capital cultural baixo",False: "capital cultural alto"},
    "TP_SIT_FUNC_ESC":        {True:  "escola ativa",          False: "escola inativa"},
}


def parse_tree_to_paths(tree_rules_text: str) -> dict[float, str]:
    """
    Parses tree_rules text from SingleTreeCateInterpreter.
    Returns dict: rounded_ate -> descriptive path string.
    """
    lines = tree_rules_text.strip().split("\n")
    stack: list[tuple[int, str]] = []
    paths: dict[float, str] = {}

    for line in lines:
        depth = line.count("|   ")
        raw   = line.replace("|   ", "").replace("|--- ", "").strip()

        if raw.startswith("value:"):
            val = re.search(r"\[([\d.]+)\]", raw)
            if val:
                ate = float(val.group(1))
                path_parts = [c for d, c in stack if d < depth]
                paths[round(ate, 2)] = " · ".join(path_parts) if path_parts else "—"
        else:
            m = re.match(r"(\w+)\s*([<>]=?)\s*([\d.]+)", raw)
            if m:
                feat, op, _ = m.group(1), m.group(2), float(m.group(3))
                is_le  = "<=" in op
                labels = FEATURE_LABELS.get(feat, {})
                label  = labels.get(is_le, f"{feat} {op}")
                stack  = [(d, c) for d, c in stack if d < depth]
                stack.append((depth, label))

    return paths


def match_leaf_to_path(leaf_ate: float, paths: dict[float, str]) -> str:
    """Matches a leaf ATE to the closest parsed path by value."""
    if not paths:
        return f"Folha (ATE={leaf_ate:.3f})"
    closest = min(paths.keys(), key=lambda k: abs(k - leaf_ate))
    if abs(closest - leaf_ate) > 0.05:
        return f"Folha (ATE={leaf_ate:.3f})"
    return paths[closest]


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"[WARN] Arquivo não encontrado: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_causal(raw: dict) -> dict | None:
    effects = raw.get("effects", [])
    return effects[0] if effects else None


def build_year(year: int) -> dict | None:
    causal_raw = load_json(DOCS / f"enem-metrics-{year}" / "causal" / f"causal_effects_{year}.json")
    stat_raw   = load_json(DOCS / f"enem-metrics-{year}" / "statistical" / f"statistical_tests_{year}.json")

    causal = extract_causal(causal_raw)
    if not causal:
        print(f"[SKIP] {year}: sem dados causais.")
        return None

    scale      = causal.get("scale_meta", {})
    scale_y    = scale.get("MEDIA_CANDIDATO", {})
    scale_r    = scale.get("RENDA", {})
    std_enem   = scale_y.get("std", 1.0)
    std_renda  = scale_r.get("std", 1.0)
    per1k_factor = scale_r.get("ate_per_1k_factor", 1000 / std_renda)

    ate        = causal.get("ate", 0.0)
    ci_lo      = causal.get("ate_ci_lower", 0.0)
    ci_hi      = causal.get("ate_ci_upper", 0.0)
    ate_se     = (ci_hi - ci_lo) / (2 * 1.96)
    per1k      = causal.get("ate_per_1k_renda_pts", ate * per1k_factor * std_enem)
    p10        = causal.get("effect_p10", 0.0)
    p50        = causal.get("effect_p50", 0.0)
    p90        = causal.get("effect_p90", 0.0)

    # parse tree rules to get descriptive leaf names
    tree_rules = causal.get("tree_rules", "")
    paths      = parse_tree_to_paths(tree_rules)

    # RLM
    rlm       = stat_raw.get("rlm", {}) or {}
    rlm_coefs = rlm.get("coefficients", {})
    rlm_scale = rlm.get("scale", 1.0)

    def get_beta(key): return rlm_coefs.get(key, {}).get("coef_std", 0.0)
    def get_pval(key): return rlm_coefs.get(key, {}).get("pvalue", None)
    def get_per1k(key): return rlm_coefs.get(key, {}).get("coef_per_1k_brl", 0.0)
    def get_pts(key): return rlm_coefs.get(key, {}).get("coef_original_units",
                                                         get_beta(key) * std_enem)

    rlm_beta  = get_beta("RENDA")
    rlm_per1k = get_per1k("RENDA")

    def coef_entry(label, key, note):
        if key is None:
            return [label, None, None, note]
        b = get_beta(key)
        p = get_pval(key)
        if b == 0.0 and p is None:
            return [label, None, None, note]
        return [label, round(b, 4), p, note]

    rlm_coefs_display = [
        coef_entry("Renda",            "RENDA",            f"+{rlm_per1k:.3f} pts/R$1k"),
        coef_entry("Capital cultural", "SCORE_CULT_PAIS",  "+{:.3f} pts/un.".format(get_pts("SCORE_CULT_PAIS"))),
        coef_entry("Internet (sim)",   "C(INTERNET)[T.1]", "vs. sem internet"),
        coef_entry("Escola federal",   "C(TP_ESCOLA)[T.3]","vs. pública (T=3)"),
        ["Raça (ref. branca)",         None, None,          "n.s. na maioria"],
        ["Idade (spline)",             None, None,          "controlada"],
    ]

    # build leaves with descriptive names from tree_rules
    leaf_stats  = causal.get("leaf_stats", [])
    leaves_out  = []
    for ls in sorted(leaf_stats, key=lambda x: x.get("ate", 0), reverse=True):
        leaf_ate   = ls.get("ate", 0.0)
        leaf_std   = ls.get("std", 0.0)
        leaf_se    = ls.get("se",  0.0)
        leaf_cv    = ls.get("cv",  0.0) or 0.0
        leaf_n     = ls.get("n",   0)
        leaf_per1k = ls.get("ate_per_1k_pts", leaf_ate * per1k_factor * std_enem)
        group      = match_leaf_to_path(leaf_ate, paths)

        leaves_out.append({
            "group":     group,
            "ate":       round(leaf_ate,  4),
            "std":       round(leaf_std,  4),
            "se":        round(leaf_se,   5),
            "cv":        round(leaf_cv,   3),
            "n":         leaf_n,
            "per1k_pts": round(leaf_per1k, 2),
        })

    stat_meta  = stat_raw.get("scale_meta", {})
    renda_meta = stat_meta.get("RENDA", scale_r)

    return {
        "ate":       round(ate,    4),
        "ci_lo":     round(ci_lo,  4),
        "ci_hi":     round(ci_hi,  4),
        "se":        round(ate_se, 4),
        "p10":       round(p10,    4),
        "p50":       round(p50,    4),
        "p90":       round(p90,    4),
        "pts":       round(ate * std_enem, 2),
        "per1k":     round(per1k,  2),
        "mean_enem": round(scale_y.get("mean", 0.0), 1),
        "std_enem":  round(std_enem, 2),
        "renda_mean":round(renda_meta.get("mean", 0.0), 0),
        "renda_std": round(std_renda, 0),
        "rlm_beta":  round(rlm_beta,  4),
        "rlm_per1k": round(rlm_per1k, 3),
        "rlm_cult":  round(get_beta("SCORE_CULT_PAIS"), 4),
        "rlm_internet": round(get_beta("C(INTERNET)[T.1]"), 4),
        "rlm_scale": round(rlm_scale, 4),
        "rlm_coefs": rlm_coefs_display,
        "context":   YEAR_CONTEXTS.get(year, ""),
        "leaves":    leaves_out,
    }


def main() -> None:
    entries = []
    ok = 0
    for year in YEARS:
        data = build_year(year)
        if data is None:
            continue
        entries.append(f"  {year}: {json.dumps(data, ensure_ascii=False)}")
        ok += 1
        print(f"[OK]   {year}: ate={data['ate']:.4f} | folhas={len(data['leaves'])}")

    if ok == 0:
        print("[ERROR] Nenhum ano processado.")
        sys.exit(1)

    js  = "// gerado automaticamente por scripts/enem_pipeline/generate_data_js.py\n"
    js += "const D = {\n"
    js += ",\n".join(entries)
    js += "\n};\n"

    out = DOCS / "data.js"
    out.write_text(js, encoding="utf-8")
    print(f"\n[OK] docs/data.js gerado com {ok} anos ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()