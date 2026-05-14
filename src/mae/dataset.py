import logging
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils import setup_logging
from .config import MAEConfig

setup_logging()


def build_transform(cfg: MAEConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (cfg.image_size, cfg.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


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

            # Save the selected subset for reproducibility
            subset_file = cfg.output_dir / "subset_paths.txt"
            subset_file.parent.mkdir(parents=True, exist_ok=True)
            with subset_file.open("w") as f:
                for p in paths:
                    f.write(f"{p}\n")
            logging.info(f"Subset paths saved to {subset_file}.")

        self.image_paths: list[Path] = sorted(paths)
        self.transform = build_transform(cfg)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index]).convert("L")
        return self.transform(image)
