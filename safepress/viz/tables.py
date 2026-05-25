"""Generate LaTeX and markdown tables for paper."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Generic DataFrame -> LaTeX
# ---------------------------------------------------------------------------

def results_to_latex(
    results_df: pd.DataFrame,
    *,
    caption: str = "",
    label: str = "",
    bold_best: bool = True,
    higher_is_better: Optional[Dict[str, bool]] = None,
    float_fmt: str = ".2f",
) -> str:
    """Convert a comparison DataFrame to a LaTeX ``tabular`` environment.

    Parameters
    ----------
    results_df:
        The data to render.  The first column is typically a row label
        (e.g. method name); the remaining columns are numeric results.
    caption:
        LaTeX table caption.
    label:
        LaTeX label for ``\\label{}``.
    bold_best:
        If ``True``, the best value in each numeric column is
        rendered in ``\\textbf{}``.
    higher_is_better:
        Optional mapping ``column_name -> bool`` indicating whether
        higher values are better for that column.  Columns not present
        default to ``True``.
    float_fmt:
        Format specifier for floating-point numbers.

    Returns
    -------
    str
        A complete ``\\begin{table}...\\end{table}`` LaTeX string.
    """
    if higher_is_better is None:
        higher_is_better = {}

    df = results_df.copy()
    columns = list(df.columns)

    # Identify numeric columns (skip the first column if it looks like a label).
    numeric_cols: List[str] = []
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    # Find best value per numeric column.
    best_vals: Dict[str, float] = {}
    if bold_best:
        for col in numeric_cols:
            hib = higher_is_better.get(col, True)
            if hib:
                best_vals[col] = float(df[col].max())
            else:
                best_vals[col] = float(df[col].min())

    # Build LaTeX body rows.
    rows_latex: List[str] = []
    for _, row in df.iterrows():
        cells: List[str] = []
        for col in columns:
            val = row[col]
            if col in numeric_cols:
                formatted = f"{float(val):{float_fmt}}"
                if bold_best and col in best_vals:
                    if abs(float(val) - best_vals[col]) < 1e-9:
                        formatted = "\\textbf{" + formatted + "}"
                cells.append(formatted)
            else:
                cells.append(_latex_escape(str(val)))
        rows_latex.append(" & ".join(cells) + " \\\\")

    # Header.
    header_cells = [_latex_escape(str(c)) for c in columns]
    header = " & ".join(header_cells) + " \\\\"
    col_spec = "l" + "c" * (len(columns) - 1)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
    ]
    if caption:
        lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")
    lines.extend(rows_latex)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in *text*."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------------------
# 2. Generic DataFrame -> Markdown
# ---------------------------------------------------------------------------

def results_to_markdown(results_df: pd.DataFrame) -> str:
    """Convert a DataFrame to a GitHub-flavored Markdown table.

    Parameters
    ----------
    results_df:
        The data to render.

    Returns
    -------
    str
        A GFM Markdown table string.
    """
    df = results_df.copy()
    columns = list(df.columns)

    # Compute column widths for alignment.
    col_widths: List[int] = []
    str_cols: List[List[str]] = []
    for col in columns:
        vals = [str(col)]
        for _, row in df.iterrows():
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        max_w = max(len(v) for v in vals)
        col_widths.append(max_w)
        str_cols.append(vals)  # first entry is header

    # Transpose str_cols to rows.
    n_rows = len(df) + 1  # +1 for header
    table_rows: List[List[str]] = []
    for r in range(n_rows):
        row_cells = []
        for c in range(len(columns)):
            cell = str_cols[c][r]
            row_cells.append(cell.ljust(col_widths[c]))
        table_rows.append(row_cells)

    # Build output.
    lines: List[str] = []

    # Header.
    lines.append("| " + " | ".join(table_rows[0]) + " |")

    # Separator -- right-align numeric columns, left-align others.
    sep_parts: List[str] = []
    for i, col in enumerate(columns):
        w = col_widths[i]
        if pd.api.types.is_numeric_dtype(df[col]):
            sep_parts.append("-" * (w - 1) + ":")
        else:
            sep_parts.append("-" * w)
    lines.append("| " + " | ".join(sep_parts) + " |")

    # Data rows.
    for row_cells in table_rows[1:]:
        lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Causal experiment results table
# ---------------------------------------------------------------------------

def causal_results_table(
    targeted: Dict[str, Dict[str, float]],
    rollback: Dict[str, Dict[str, float]],
    control: Dict[str, Dict[str, float]],
    *,
    fmt: str = "latex",
) -> str:
    """Generate the main causal-experiment results table.

    Parameters
    ----------
    targeted:
        Dict ``condition_name -> {"refusal_rate": float, ...}``
        for the targeted-quantization experiment.
    rollback:
        Same structure for the rollback experiment.
    control:
        Same structure for the scoring-method comparison.
    fmt:
        ``"latex"`` or ``"markdown"``.

    Returns
    -------
    str
        Formatted table string.
    """
    # Merge all three experiments into a single DataFrame.
    rows: List[Dict[str, Any]] = []

    def _add_section(
        label: str,
        data: Dict[str, Dict[str, float]],
    ) -> None:
        for condition, metrics in data.items():
            row: Dict[str, Any] = {
                "Experiment": label,
                "Condition": condition,
            }
            row.update(metrics)
            rows.append(row)

    _add_section("Targeted", targeted)
    _add_section("Rollback", rollback)
    _add_section("Control", control)

    df = pd.DataFrame(rows)

    # Ensure a consistent column order: Experiment, Condition, then sorted metrics.
    fixed = ["Experiment", "Condition"]
    metric_cols = sorted([c for c in df.columns if c not in fixed])
    df = df[fixed + metric_cols]

    # Rename metric columns for nicer display.
    rename_map: Dict[str, str] = {}
    for col in metric_cols:
        rename_map[col] = col.replace("_", " ").title()
    df = df.rename(columns=rename_map)

    if fmt == "latex":
        return results_to_latex(
            df,
            caption="Causal experiment results",
            label="tab:causal",
            bold_best=True,
            higher_is_better={
                rename_map.get(c, c): (c == "refusal_rate")
                for c in metric_cols
            },
        )
    else:
        return results_to_markdown(df)
