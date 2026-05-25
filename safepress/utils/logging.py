from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict


class _NaNSafeEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Inf to None for JSON compliance."""

    def default(self, obj):
        return super().default(obj)

    def encode(self, o):
        return super().encode(_sanitize_for_json(o))


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def init_run_dir(out_dir: str | Path, allow_overwrite: bool = False) -> Path:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not allow_overwrite:
        raise FileExistsError(
            f"Output directory exists and is not empty: {out_dir}. "
            f"Pass --overwrite to reuse it."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_for_json(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
