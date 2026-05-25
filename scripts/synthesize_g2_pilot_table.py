#!/usr/bin/env python
"""Emit the consolidated G2 pilot table (Markdown + LaTeX) from on-disk files.

Reads, for each (model, method, budget) condition:
  - ``eval_seed0.json``           -> refusal_rate
  - ``harmbench_asr_seed0.json``  -> HarmBench-classifier ASR
  - ``utility_seed0.json``        -> WikiText-2 PPL + MMLU-lite accuracy

Writes:
  - ``runs/emnlp_synthesis/g2_pilot_table.md``     (paper / docs)
  - ``runs/emnlp_synthesis/g2_pilot_table.tex``    (LaTeX appendix-ready)
  - ``runs/emnlp_synthesis/g2_pilot_long.csv``     (long-form for paired stats)

Use::

    python scripts/synthesize_g2_pilot_table.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DIR_RE = re.compile(r"sweep_(?P<tag>[^/]+?)_(?P<method>fp16|int4|ssmp|gradient_only|snip|magnitude|qresafe|qresafe_noft|wanda|cwp|cwp_published|random|lastn|layer_uniform)_b(?P<budget>[0-9.]+)$")

# Print order: rank-roughly by recovery quality on the typical run.
_METHOD_PRINT_ORDER = [
    "fp16",
    "gradient_only",
    "ssmp",
    "snip",
    "magnitude",
    "int4",
]
_BUDGET_PRINT_ORDER = [0.60, 0.08, 0.04, 0.0]


def _maybe(d: dict, *keys):
    """Walk a possibly-nested dict picking the first key present."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
    return d if not isinstance(d, dict) else None


def _read_condition(cond_dir: Path) -> Optional[Dict]:
    """Aggregate all metrics for one condition. Returns None if irreparably empty."""
    eval_p = cond_dir / "eval_seed0.json"
    asr_p = cond_dir / "harmbench_asr_seed0.json"
    util_p = cond_dir / "utility_seed0.json"
    if not eval_p.exists():
        return None

    out = {"cond_dir": cond_dir.name}
    e = json.loads(eval_p.read_text())
    out["refusal_rate"] = e.get("refusal_rate")

    if asr_p.exists():
        a = json.loads(asr_p.read_text())
        out["harmbench_asr"] = a.get("asr")
    else:
        out["harmbench_asr"] = None

    if util_p.exists():
        u = json.loads(util_p.read_text())
        ppl_d = u.get("perplexity")
        out["ppl"] = ppl_d.get("perplexity") if isinstance(ppl_d, dict) else ppl_d
        mmlu_d = u.get("mmlu")
        out["mmlu"] = mmlu_d.get("accuracy") if isinstance(mmlu_d, dict) else mmlu_d
    else:
        out["ppl"] = None
        out["mmlu"] = None
    return out


def _scan_pilot(sweep_root: Path, model_label: str) -> List[Dict]:
    rows: List[Dict] = []
    for cond_dir in sorted(sweep_root.glob("sweep_*")):
        if not cond_dir.is_dir():
            continue
        m = _DIR_RE.match(cond_dir.name)
        if not m:
            continue
        data = _read_condition(cond_dir)
        if data is None:
            continue
        data.update({
            "model": model_label,
            "method": m.group("method"),
            "budget": float(m.group("budget")),
        })
        rows.append(data)
    return rows


def _fmt(v, places=3):
    if v is None:
        return "-"
    try:
        return f"{float(v):.{places}f}"
    except Exception:  # noqa: BLE001
        return str(v)


def _sort_key(row: Dict) -> Tuple[int, int]:
    method = row["method"]
    budget = row["budget"]
    try:
        mi = _METHOD_PRINT_ORDER.index(method)
    except ValueError:
        mi = len(_METHOD_PRINT_ORDER)
    try:
        bi = _BUDGET_PRINT_ORDER.index(budget)
    except ValueError:
        bi = len(_BUDGET_PRINT_ORDER)
    return (mi, bi)


def _write_md(rows: List[Dict], out_path: Path) -> None:
    """Side-by-side Qwen3 / Llama markdown table."""
    by_model: Dict[str, Dict[Tuple[str, float], Dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[(r["method"], r["budget"])] = r

    models = sorted(by_model.keys())
    # Take the union of (method, budget) tuples and sort them.
    cells = sorted(
        {(m, b) for d in by_model.values() for (m, b) in d.keys()},
        key=lambda mb: _sort_key({"method": mb[0], "budget": mb[1]}),
    )

    lines: List[str] = []
    lines.append("# G2 pilot — final consolidated table")
    lines.append("")
    lines.append("Per-condition refusal_rate (higher safer), HarmBench-classifier "
                 "ASR (lower safer), WikiText-2 PPL (lower better), MMLU-lite "
                 "(higher better; random chance = 0.25 for 4-way MCQ).")
    lines.append("")
    for model in models:
        lines.append(f"## {model}")
        lines.append("")
        header_cols = ["method", "budget", "refusal", "HB ASR", "PPL", "MMLU"]
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "|".join(["---:" if i > 1 else "---" for i in range(len(header_cols))]) + "|")
        for mb in cells:
            row = by_model[model].get(mb)
            if row is None:
                continue
            lines.append(
                "| " + " | ".join([
                    row["method"],
                    f"{row['budget']:.2f}",
                    _fmt(row["refusal_rate"]),
                    _fmt(row["harmbench_asr"]),
                    _fmt(row["ppl"], places=2),
                    _fmt(row["mmlu"]),
                ]) + " |"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[synth] wrote {out_path}")


def _write_tex(rows: List[Dict], out_path: Path) -> None:
    """LaTeX booktabs table (one per model, side by side)."""
    by_model: Dict[str, Dict[Tuple[str, float], Dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[(r["method"], r["budget"])] = r

    models = sorted(by_model.keys())
    cells = sorted(
        {(m, b) for d in by_model.values() for (m, b) in d.keys()},
        key=lambda mb: _sort_key({"method": mb[0], "budget": mb[1]}),
    )

    lines: List[str] = []
    lines.append("% G2 pilot consolidated table -- auto-generated. Do not edit by hand.")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lr" + ("rrrr" * len(models)) + r"}")
    lines.append(r"\toprule")
    lines.append("Method & Budget" + "".join([f" & \\multicolumn{{4}}{{c}}{{{m}}}" for m in models]) + r" \\")
    # second header row
    lines.append(
        " & "
        + " & ".join(["refusal & ASR & PPL & MMLU"] * len(models))
        + r" \\"
    )
    lines.append(r"\midrule")

    for mb in cells:
        method, budget = mb
        # build a row across all models
        row_parts = [method.replace("_", r"\_"), f"{budget:.2f}"]
        for model in models:
            r = by_model[model].get(mb)
            if r is None:
                row_parts.extend(["-"] * 4)
            else:
                row_parts.extend([
                    _fmt(r["refusal_rate"]),
                    _fmt(r["harmbench_asr"]),
                    _fmt(r["ppl"], places=2),
                    _fmt(r["mmlu"]),
                ])
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{G2 pilot: scoring ablation at 3-bit simulated symmetric group-wise PTQ. "
                 r"Refusal rate (higher safer); HarmBench-classifier ASR (lower safer); "
                 r"WikiText-2 perplexity (lower better); MMLU-lite accuracy "
                 r"(higher better; 0.25 is random for 4-way MCQ).}")
    lines.append(r"\label{tab:g2_pilot}")
    lines.append(r"\end{table*}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[synth] wrote {out_path}")


def _write_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "method", "budget", "refusal_rate", "harmbench_asr", "ppl", "mmlu", "cond_dir"])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["model"], _sort_key(x))):
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"[synth] wrote {out_path}")


def main() -> None:
    rows: List[Dict] = []
    # G2 pilot (4 scorers x 2 budgets x 5 models + fp16/int4 references)
    rows.extend(_scan_pilot(Path("runs/emnlp_g2_pilot"), "Qwen3-8B"))
    rows.extend(_scan_pilot(Path("runs/emnlp_g2_pilot_llama31"), "Llama-3.1-8B-Instruct"))
    rows.extend(_scan_pilot(Path("runs/emnlp_g2_pilot_mistral"), "Ministral-8B"))
    rows.extend(_scan_pilot(Path("runs/emnlp_g2_pilot_glm4"), "GLM-4-9B"))
    rows.extend(_scan_pilot(Path("runs/emnlp_g2_pilot_deepseek"), "DeepSeek-R1-Distill-Llama-8B"))
    # Fisher@60% (CWP-equivalent under Theorem 2)
    rows.extend(_scan_pilot(Path("runs/emnlp_fisher60/qwen3"), "Qwen3-8B"))
    rows.extend(_scan_pilot(Path("runs/emnlp_fisher60/llama31"), "Llama-3.1-8B-Instruct"))
    rows.extend(_scan_pilot(Path("runs/emnlp_fisher60/mistral"), "Ministral-8B"))
    rows.extend(_scan_pilot(Path("runs/emnlp_fisher60/glm4"), "GLM-4-9B"))
    rows.extend(_scan_pilot(Path("runs/emnlp_fisher60/deepseek"), "DeepSeek-R1-Distill-Llama-8B"))
    if not rows:
        raise SystemExit("[synth] no conditions found")
    out_dir = Path("runs/emnlp_synthesis")
    _write_md(rows, out_dir / "g2_pilot_table.md")
    _write_tex(rows, out_dir / "g2_pilot_table.tex")
    _write_csv(rows, out_dir / "g2_pilot_long.csv")
    print(f"[synth] {len(rows)} conditions consolidated")


if __name__ == "__main__":
    main()
