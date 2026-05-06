from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_int(key: str, default: int) -> int:
    value: str | None = os.getenv(key)
    return int(value) if value is not None else default


def _env_float(key: str, default: float) -> float:
    value: str | None = os.getenv(key)
    return float(value) if value is not None else default


def _env_str(key: str, default: str) -> str:
    value: str | None = os.getenv(key)
    return value if value is not None else default


@dataclass
class MAEConfig:
    """Hyperparameters and configuration for training the MAE model."""

    # Paths
    data_dir: Path = Path(_env_str("DATA_DIR", "data/rsna"))
    output_dir: Path = Path("checkpoints")

    # ViT-B/16 architecture (encoder)
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072

    # MAE decoder
    decoder_hidden_size: int = 512
    decoder_num_hidden_layers: int = 8
    decoder_num_attention_heads: int = 16
    decoder_intermediate_size: int = 2048

    # MedMAE
    mask_ratio: float = 0.75
    norm_pix_loss: bool = True

    # Optimization
    epochs: int = 1000
    batch_size: int = 64
    base_learning_rate: float = 1e-3
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_epochs: int = 40

    # Runtime
    num_workers: int = 4
    save_every: int = 50
    log_every: int = 50
    seed: int = 42
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
