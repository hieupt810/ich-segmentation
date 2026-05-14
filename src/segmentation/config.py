import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class SegmentationConfig:
    vit_name: str = os.environ.get("VIT_NAME", "vit_small_patch16_224")
    image_size: int = 224
    in_chans: int = 1

    # Segmentation head
    num_classes: int = 1
    feature_size: int = 16
    skip_block_indices: tuple[int, int, int] = (2, 5, 8)

    # Pretrained backbone
    output_dir: Path = Path("output")
    mae_checkpoint_path: Path = output_dir / "best_model_vit-16.pth"
    freeze_encoder: bool = True

    seed: int = 42
