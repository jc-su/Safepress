from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class ProtectPlan:
    """
    Map: original_module_name -> list of protected out-block indices.
    """
    block_size: int
    budget_ratio: float
    total_params: int
    protected_params: int
    protect_map: Dict[str, List[int]]


def select_top_blocks(
    scores: pd.DataFrame,
    *,
    budget_ratio: float,
    block_size: int,
    min_blocks_per_module: int = 0,
) -> ProtectPlan:
    """
    Select blocks by descending score until we reach the budget.

    budget_ratio: fraction of total Linear params to keep in higher precision.
    """
    if not (0.0 < budget_ratio < 1.0):
        raise ValueError(f"budget_ratio must be in (0,1), got {budget_ratio}")

    required_cols = {"module", "block_idx", "num_params", "score"}
    missing = required_cols - set(scores.columns)
    if missing:
        raise ValueError(f"Scores df missing columns: {missing}")

    # Total params in all blocks (i.e., all Linear weights)
    total_params = int(scores["num_params"].sum())
    budget_params = int(total_params * float(budget_ratio))

    protect_map: Dict[str, List[int]] = {}
    protected_params = 0

    # optional: ensure each module has some minimum blocks protected
    if min_blocks_per_module > 0:
        by_mod = scores.groupby("module")
        for mod, dfm in by_mod:
            topm = dfm.sort_values("score", ascending=False).head(min_blocks_per_module)
            protect_map[mod] = sorted(topm["block_idx"].astype(int).tolist())
            protected_params += int(topm["num_params"].sum())

    # now fill remaining budget globally
    remaining = scores.sort_values("score", ascending=False)
    for _, row in remaining.iterrows():
        mod = str(row["module"])
        b = int(row["block_idx"])
        n = int(row["num_params"])
        if mod in protect_map and b in protect_map[mod]:
            continue
        if protected_params + n > budget_params:
            continue
        protect_map.setdefault(mod, []).append(b)
        protected_params += n
        if protected_params >= budget_params:
            break

    # sort indices
    for mod in list(protect_map.keys()):
        protect_map[mod] = sorted(set(int(x) for x in protect_map[mod]))

    return ProtectPlan(
        block_size=block_size,
        budget_ratio=float(budget_ratio),
        total_params=total_params,
        protected_params=protected_params,
        protect_map=protect_map,
    )


def save_protect_plan(plan: ProtectPlan, path: str | Path) -> None:
    obj = dict(
        block_size=plan.block_size,
        budget_ratio=plan.budget_ratio,
        total_params=plan.total_params,
        protected_params=plan.protected_params,
        protect_map=plan.protect_map,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_protect_plan(path: str | Path) -> ProtectPlan:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProtectPlan(
        block_size=int(obj["block_size"]),
        budget_ratio=float(obj["budget_ratio"]),
        total_params=int(obj["total_params"]),
        protected_params=int(obj["protected_params"]),
        protect_map={str(k): [int(x) for x in v] for k, v in obj["protect_map"].items()},
    )
