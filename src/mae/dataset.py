import logging
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils import setup_logging
from .config import MAEConfig

setup_logging()


class RSNADataset(Dataset):
    def __init__(self, cfg: MAEConfig) -> None:
        super().__init__()

        paths: list[Path] = [
            p for p in Path(cfg.data_dir).rglob("*") if p.suffix.lower() == ".png"
        ]
        logging.info(f"Found {len(paths)} images in {cfg.data_dir}.")

        if cfg.subset_size < len(paths):
            paths = random.Random(cfg.seed).sample(paths, cfg.subset_size)
            logging.info(f"Using a subset of {len(paths)} images for training.")

        self.image_paths: list[Path] = sorted(paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)
