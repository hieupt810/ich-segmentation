import os
import random

import numpy as np
import torch


# --- Mixed precision utilities ---
def resolve_amp_dtype(name: str) -> torch.dtype:
    """Resolves a string name to a torch.dtype for automatic mixed precision (AMP)."""

    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(
            f"amp_dtype must be one of {list(mapping.keys())}, got '{name!r}'"
        )
    return mapping[name]


# --- Reproducibility utilities ---
def set_seed(seed: int) -> None:
    """Sets the random seed for reproducibility across various libraries and environments."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_: int) -> None:
    """Sets the random seed for a worker process, ensuring reproducibility when using DataLoader with multiple workers."""

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
