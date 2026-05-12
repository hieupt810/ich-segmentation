import random
from pathlib import Path

from lightly.transforms import MAETransform
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .config import MAEConfig


class RSNADataset(Dataset):
    def __init__(self, cfg: MAEConfig) -> None:
        super().__init__()

        paths: list[Path] = [
            p for p in Path(cfg.data_dir).rglob("*") if p.suffix.lower() == ".png"
        ]

        if cfg.subset_size < len(paths):
            paths = random.Random(cfg.seed).sample(paths, cfg.subset_size)

        self.image_paths: list[Path] = sorted(paths)
        self.transform = MAETransform()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index]).convert("RGB")
        views: list[Tensor] = self.transform(image)
        return views[0]
