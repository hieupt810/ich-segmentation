import random
from pathlib import Path

from lightly.transforms import MAETransform
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class RSNADataset(Dataset):
    def __init__(
        self, data_dir: str | Path, subsample_size: int = 25000, seed: int = 42
    ) -> None:
        super().__init__()

        paths: list[Path] = [
            p for p in Path(data_dir).rglob("*") if p.suffix.lower() == ".png"
        ]

        if subsample_size < len(paths):
            paths = random.Random(seed).sample(paths, subsample_size)

        self.image_paths: list[Path] = sorted(paths)
        self.transform = MAETransform()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index]).convert("RGB")
        views: list[Tensor] = self.transform(image)
        return views[0]
