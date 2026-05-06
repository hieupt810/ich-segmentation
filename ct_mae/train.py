from __future__ import annotations

import logging
import math

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import ViTMAEConfig, ViTMAEForPreTraining

from .config import MAEConfig
from .dataset import RSNADataset
from .transform import build_transform
from .utils import resolve_amp_dtype, seed_worker, set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_model(cfg: MAEConfig) -> ViTMAEForPreTraining:
    """Instantiate a ViT-B/16 MAE configured to MedMAE specifications."""

    vit_mae_config = ViTMAEConfig(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        num_channels=cfg.num_channels,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        decoder_hidden_size=cfg.decoder_hidden_size,
        decoder_num_hidden_layers=cfg.decoder_num_hidden_layers,
        decoder_num_attention_heads=cfg.decoder_num_attention_heads,
        decoder_intermediate_size=cfg.decoder_intermediate_size,
        mask_ratio=cfg.mask_ratio,
        norm_pix_loss=cfg.norm_pix_loss,
    )
    return ViTMAEForPreTraining(vit_mae_config)


def cosine_with_warmup(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """Linear warmup followed by cosine decay to zero."""
    if epoch < warmup_epochs:
        return float(epoch) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train(cfg: MAEConfig) -> None:
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    transform = build_transform(cfg.image_size)
    dataset = RSNADataset(cfg.data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=seed_worker,
        persistent_workers=(cfg.num_workers > 0),
    )
    logger.info(f"Dataset size: {len(dataset)} | steps/epoch: {len(loader)}")

    # --- Model ---
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params / 1e6:.2f} M")

    # --- Optimizer / Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.base_learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: cosine_with_warmup(e, cfg.warmup_epochs, cfg.epochs),
    )

    # --- AMP ---
    amp_dtype: torch.dtype = resolve_amp_dtype(cfg.amp_dtype)
    amp_enabled: bool = cfg.use_amp and device.type == "cuda"
    scaler = GradScaler(device=device.type, enabled=amp_enabled)

    # --- Training Loop ---
    model.train()
    for epoch in range(cfg.epochs):
        running_loss = 0.0
        n_batches = 0
        for step, pixel_values in enumerate(loader):
            pixel_values = pixel_values.to(device, non_blocking=True)

            with autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                outputs = model(pixel_values=pixel_values)
                loss: torch.Tensor = outputs.loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

            if step % cfg.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"epoch {epoch + 1:04d}/{cfg.epochs} "
                    f"step {step:04d}/{len(loader)} "
                    f"loss {loss.item():.4f} lr {lr:.2e}"
                )

        scheduler.step()
        avg = running_loss / max(1, n_batches)
        logger.info(f"[epoch {epoch + 1:04d}] avg_loss={avg:.4f}")

        if (epoch + 1) % cfg.save_every == 0 or (epoch + 1) == cfg.epochs:
            ckpt = cfg.output_dir / f"mae_epoch_{epoch + 1:04d}"
            model.save_pretrained(ckpt)
            logger.info(f"Saved checkpoint to {ckpt}")

    logger.info("Training complete!")
