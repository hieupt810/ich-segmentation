import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MAEConfig:
    vit_name: str = os.environ.get("VIT_NAME", "vit_small_patch16_224")
    image_size: int = 224

    data_dir: Path = Path(os.environ.get("DATA_DIR", "data/rsna"))
    output_dir: Path = Path("output")
    checkpoint_path: Path = output_dir / f"mae_{vit_name}.pt"

    # Hyperparameters for training
    seed: int = 42
    subset_size: int = 50000
    batch_size: int = 2048
    num_workers: int = 32
    epochs: int = 400
    warmup_epochs: int = 10

    # Hyperparameters for the MAE model
    in_chans: int = 1
    attn_drop_rate: float = 0.0
    decoder_depth: int = 1
    decoder_dim: int = 512
    decoder_num_heads: int = 16
    mask_ratio: float = 0.75
    mlp_ratio: float = 4.0
    proj_drop_rate: float = 0.0
