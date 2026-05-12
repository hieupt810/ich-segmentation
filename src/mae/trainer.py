import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .config import MAEConfig
from .dataset import RSNADataset
from .model import MAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_scheduler(
    optimizer: optim.Optimizer, cfg: MAEConfig
) -> lr_scheduler.LRScheduler:
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=cfg.warmup_epochs
    )
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-6
    )
    return lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_epochs],
    )


def _set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.default_rng(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % 2**32

    random.seed(seed + worker_id)
    np.random.default_rng(seed + worker_id)


def train_mae(cfg: MAEConfig):
    _set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAE(cfg).to(device)

    dataset = RSNADataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_worker_init_fn,
    )

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.05
    )
    scheduler = _get_scheduler(optimizer, cfg)
    scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))

    logger.info("Starting training...")
    best_loss = float("inf")
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        for images in dataloader:
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                predictions, targets = model(images)
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach()
            scheduler.step()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), cfg.output_dir / "best_model.pth")
            logger.info(f"New best model saved with loss: {best_loss:.5f}")

    logger.info("Training completed.")
