import os
import random

import numpy as np
import torch


# --- Mixed precision utilities ---
def autocast_context(device: torch.device) -> torch.amp.autocast:
    """Returns an autocast context manager for the specified device. Uses float16 for CUDA devices and float32 for others."""

    dtype: torch.dtype = torch.float16 if device.type == "cuda" else torch.float32
    enabled: bool = device.type == "cuda"
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


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
