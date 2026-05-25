from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_prompts_jsonl(path: str | Path, key: str = "prompt") -> List[str]:
    rows = read_jsonl(path)
    prompts: List[str] = []
    for r in rows:
        if key not in r:
            raise KeyError(f"JSONL row missing key='{key}'. Row keys={list(r.keys())}")
        prompts.append(str(r[key]))
    return prompts


def read_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml  # lazy

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: str | Path, data: Dict[str, Any]) -> None:
    import yaml  # lazy

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
