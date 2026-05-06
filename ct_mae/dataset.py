from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class RSNADataset(Dataset):
    """Loads PNG brain-CT slices for MAE pretraining."""

    IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png"})

    def __init__(
        self, data_dir: str | Path, *, transform: Callable | None = None
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform

        self.image_paths: list[Path] = [
            p
            for p in self.data_dir.rglob("*")
            if p.suffix.lower() in self.IMAGE_EXTENSIONS
        ]
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG files found under {self.data_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.image_paths[idx]).convert("L")
        if self.transform is not None:
            return self.transform(image)
        return image
