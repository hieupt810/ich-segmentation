import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import set_seed, setup_logging, worker_init_fn
from .config import MAEConfig
from .dataset import RSNADataset
from .model import MAE

setup_logging()


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


def train_mae(cfg: MAEConfig):
    set_seed(cfg.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAE(cfg).to(device)

    # --- Data ---
    dataset = RSNADataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn,
    )

    test_batch = next(iter(dataloader))
    logging.info(f"Test batch shape: {test_batch.shape}")

    # --- Training Setup ---
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.05
    )
    scheduler = _get_scheduler(optimizer, cfg)
    scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))

    # --- Training Loop ---
    logging.info("Starting training...")
    best_loss = float("inf")
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        for images in pbar:
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

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        avg_loss = total_loss / len(dataloader)
        pbar.set_postfix({"loss": f"{avg_loss:.5f}"})
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), cfg.output_dir / "best_model.pth")
            logging.info(f"New best model saved with loss: {best_loss:.5f}")

    logging.info("Training completed.")
