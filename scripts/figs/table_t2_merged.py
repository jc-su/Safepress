#!/usr/bin/env python
"""Consolidated direct-test table for Theorem 2 (main text §7.1).

Combines, per model:
  - SSMP<->SNIP full-ranking Spearman (from raw block scores) -- the cleanest
    "the two scorers are interchangeable" measure
  - SSMP<->SNIP top-8% protect-set Jaccard (from protect maps) -- the
    deployment-relevant "same blocks protected" measure
  - Fisher<->{SSMP,SNIP} top-8% Jaccard -- shows the single-factor escape
  - escape-clause corr(Fisher, magnitude) on top-8% blocks -- the mechanism
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_CR = re.compile(r"sweep_[^/]+_(?P<m>ssmp|gradient_only|snip|magnitude)_b(?P<b>[0-9.]+)$")

MODELS = [
    ("Llama-3.1", "runs/emnlp_g2_pilot_llama31", "runs/emnlp_blockscores/llama31_scores.csv"),
    ("Qwen3",     "runs/emnlp_g2_pilot",         "runs/emnlp_blockscores/qwen3_scores.csv"),
    ("GLM-4",     "runs/emnlp_g2_pilot_glm4",    "runs/emnlp_blockscores/glm4_scores.csv"),
    ("Ministral", "runs/emnlp_g2_pilot_mistral", "runs/emnlp_blockscores/mistral_scores.csv"),
    ("DeepSeek",  "runs/emnlp_g2_pilot_deepseek","runs/emnlp_blockscores/deepseek_scores.csv"),
]
BETA = 0.08


def protect_set(root, method, b=0.08):
    for d in Path(root).glob(f"sweep_*_{method}_b{b}"):
        pm = json.loads((d / "protect_map_seed0.json").read_text()).get("protect_map", {})
        return {(mod, bi) for mod, bl in pm.items() for bi in bl}
    return None


def jac(a, b):
    if not a or not b:
        return float("nan")
    return len(a & b) / len(a | b)


def main():
    rows = []
    for model, g2root, scorecsv in MODELS:
        # Jaccards from protect maps
        ssmp, snip = protect_set(g2root, "ssmp"), protect_set(g2root, "snip")
        mag, fish = protect_set(g2root, "magnitude"), protect_set(g2root, "gradient_only")
        j_ss = jac(ssmp, snip)
        j_fish = np.nanmean([jac(fish, ssmp), jac(fish, snip)])
        # Spearman + escape from raw block scores
        df = pd.read_csv(scorecsv)
        rho_ss, _ = spearmanr(df["ssmp"], df["snip"])
        k = max(1, int(BETA * len(df)))
        top = df.nlargest(k, "ssmp")
        esc, _ = spearmanr(top["fisher"], top["magnitude"])
        rows.append((model, rho_ss, j_ss, j_fish, esc))
        print(f"{model:12s} rho(SSMP,SNIP)={rho_ss:.3f}  Jac(SSMP,SNIP)={j_ss:.2f}  Jac(Fisher,pair)={j_fish:.2f}  escape={esc:+.2f}")

    rj = BETA / (2 - BETA)
    L = ["% Consolidated T2 direct test (main text) -- auto-generated.",
         "\\begin{table}[t]", "\\centering\\small",
         "\\begin{tabular}{lcccc}", "\\toprule",
         " & \\multicolumn{2}{c}{SSMP\\,$\\equiv$\\,SNIP} & Fisher & Escape \\\\",
         "\\cmidrule(lr){2-3}",
         "Model & $\\rho_{\\text{all}}$ & Jac$_{8\\%}$ & Jac$_{8\\%}$ & corr \\\\",
         "\\midrule"]
    for model, rho_ss, j_ss, j_fish, esc in rows:
        L.append(f"{model} & \\textbf{{{rho_ss:.3f}}} & \\textbf{{{j_ss:.2f}}} & {j_fish:.2f} & {esc:+.2f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}",
          f"\\caption{{\\textbf{{Direct test of Theorem~\\ref{{thm:collapse}} (block selection).}} "
          f"SSMP and SNIP are the two gradient-weighted scorers. $\\rho_{{\\text{{all}}}}$: Spearman correlation of their block scores over \\emph{{all}} blocks (from raw scores); "
          f"Jac$_{{8\\%}}$: Jaccard overlap of their top-8\\% protect sets (random baseline ${rj:.2f}$). Both confirm SSMP and SNIP are near-interchangeable. "
          f"\\textbf{{Fisher}} Jac$_{{8\\%}}$ is the mean overlap of Fisher's protect set with the SSMP/SNIP pair --- far lower, i.e.\\ Fisher escapes. "
          f"\\textbf{{Escape}} corr is Spearman(Fisher, magnitude) on the top-8\\% blocks: negative on all models, explaining why Fisher's magnitude-free ranking selects different blocks.}}",
          "\\label{tab:direct}", "\\end{table}"]
    Path("paper/tables/table_t2_direct.tex").write_text("\n".join(L))
    print("\n[merge] wrote paper/tables/table_t2_direct.tex")


if __name__ == "__main__":
    main()
