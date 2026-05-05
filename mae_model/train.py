from pathlib import Path

from torch.utils.data import Dataset


class RSNADataset(Dataset):
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
