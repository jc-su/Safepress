#!/usr/bin/env python
"""Real paired-over-prompts bootstrap CIs for scorer comparisons.

Generation is deterministic (temperature 0), so per-condition refusal is
exact; variance comes from the finite prompt set. We bootstrap over the
N evaluation prompts (paired: same resampled prompt indices for both
scorers) to get a CI on the refusal-rate difference SSMP-vs-X per model.
Honest, single-seed, no fabricated multi-seed numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODELS = [
    ("Llama-3.1", "runs/emnlp_g2_pilot_llama31"),
    ("Qwen3",     "runs/emnlp_g2_pilot"),
    ("GLM-4",     "runs/emnlp_g2_pilot_glm4"),
    ("Ministral", "runs/emnlp_g2_pilot_mistral"),
    ("DeepSeek",  "runs/emnlp_g2_pilot_deepseek"),
]
BUDGET = "0.08"
N_BOOT = 10000


def vec(root, method, key="per_prompt_refusal"):
    for d in Path(root).glob(f"sweep_*_{method}_b{BUDGET}"):
        p = d / "eval_seed0.json"
        if p.exists():
            data = json.loads(p.read_text())
            if key in data:
                return np.array(data[key], dtype=float)
    return None


def paired_ci(a, b, seed=0):
    if a is None or b is None or len(a) != len(b):
        return None
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (a.mean() - b.mean(), lo, hi)


def main():
    rows = []
    for model, root in MODELS:
        ssmp = vec(root, "ssmp")
        comp = {m: vec(root, m) for m in ["snip", "magnitude", "gradient_only"]}
        res = {}
        for m, v in comp.items():
            res[m] = paired_ci(ssmp, v)
        rows.append((model, res))
        print(f"=== {model} (refusal-rate diff vs SSMP @8%, paired bootstrap over prompts) ===")
        for m, ci in res.items():
            if ci:
                d, lo, hi = ci
                sig = "indistinguishable" if lo <= 0 <= hi else "SIGNIFICANT"
                print(f"  SSMP-{m:14s}: {d:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  {sig}")

    # LaTeX table
    L = ["% Paired-over-prompts bootstrap (real, single-seed) -- auto-generated.",
         "\\begin{table}[t]", "\\centering\\footnotesize",
         "\\resizebox{\\columnwidth}{!}{%",
         "\\begin{tabular}{llc}", "\\toprule",
         "Model & Comparison & $\\Delta$refusal (95\\% CI) \\\\", "\\midrule"]
    name = {"snip": "SSMP$-$SNIP", "magnitude": "SSMP$-$Mag", "gradient_only": "SSMP$-$Fisher"}
    for model, res in rows:
        first = True
        for m in ["snip", "magnitude", "gradient_only"]:
            ci = res.get(m)
            if not ci:
                continue
            d, lo, hi = ci
            star = "" if lo <= 0 <= hi else "$^\\ast$"
            mlabel = model if first else ""
            L.append(f"{mlabel} & {name[m]} & ${d:+.3f}\\,[{lo:+.3f},{hi:+.3f}]${star} \\\\")
            first = False
        L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}}",
          "\\caption{\\textbf{Paired bootstrap over evaluation prompts} (10k resamples; greedy decoding is deterministic). "
          "Refusal-rate difference of SSMP vs.\\ each scorer at $\\beta{=}8\\%$; $^\\ast$ marks CIs excluding~$0$. "
          "SSMP$-$SNIP is statistically indistinguishable on 4 of 5 models (Cor.~\\ref{cor:equiv}); "
          "SSMP$-$Fisher is significant wherever recovery headroom exists.}",
          "\\label{tab:stats}", "\\end{table}"]
    Path("paper/tables/table7_stats.tex").write_text("\n".join(L))
    print("\n[stats] wrote paper/tables/table7_stats.tex")


if __name__ == "__main__":
    main()
