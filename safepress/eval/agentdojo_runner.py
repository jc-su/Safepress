from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class AgentDojoRun:
    ok: bool
    cmd: List[str]
    returncode: int
    stdout_path: str
    stderr_path: str


def run_agentdojo(
    *,
    model: str,
    out_dir: str | Path,
    defense: str = "none",
    attack: str = "none",
    openai_base_url: str = "http://localhost:8000/v1",
    openai_api_key: str = "EMPTY",
    extra_args: Optional[List[str]] = None,
) -> AgentDojoRun:
    """
    Runs AgentDojo's benchmark script using an OpenAI-compatible endpoint (e.g., vLLM server).

    Example:
      OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
      python -m agentdojo.scripts.benchmark --model <model> --defense none --attack none

    You must have `agentdojo` installed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "agentdojo_stdout.txt"
    stderr_path = out_dir / "agentdojo_stderr.txt"

    cmd = [
        "python",
        "-m",
        "agentdojo.scripts.benchmark",
        "--model",
        model,
        "--defense",
        defense,
        "--attack",
        attack,
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = openai_base_url
    env["OPENAI_API_KEY"] = openai_api_key

    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
        proc = subprocess.run(cmd, env=env, stdout=out_f, stderr=err_f)

    return AgentDojoRun(
        ok=proc.returncode == 0,
        cmd=cmd,
        returncode=proc.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )
