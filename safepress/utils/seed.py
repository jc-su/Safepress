from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Best-effort determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # torch not installed or no CUDA, ignore.
        pass
