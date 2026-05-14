"""Structural smoke test for the MAE-pretrained UNETR segmentation model.

Runs on CPU. Verifies model construction, forward-pass output shape, and
the freeze/unfreeze contract. Tolerates a missing MAE checkpoint by falling
back to random init.

Run:
    source .venv/bin/activate
    python tools/smoke_test_segmentation.py
"""

import sys
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation import MAESegmenter, SegmentationConfig  # noqa: E402
from src.utils import set_seed, setup_logging  # noqa: E402

setup_logging()


def _count(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


def _trainable(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params if p.requires_grad)


def run_one(freeze: bool) -> None:
    cfg = replace(SegmentationConfig(), freeze_encoder=freeze)
    model = MAESegmenter(cfg).eval()

    if cfg.mae_checkpoint_path.exists():
        info = model.load_mae_checkpoint()
        print(
            f"[freeze={freeze}] loaded MAE ckpt epoch={info['epoch']} "
            f"missing={len(info['missing_keys'])} "
            f"unexpected={len(info['unexpected_keys'])}"
        )
        if info["unexpected_keys"]:
            raise AssertionError(
                f"unexpected keys when loading MAE checkpoint: "
                f"{info['unexpected_keys']}"
            )
    else:
        print(
            f"[freeze={freeze}] MAE checkpoint not found at "
            f"{cfg.mae_checkpoint_path}; using random init."
        )

    x = torch.randn(2, cfg.in_chans, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        y = model(x)

    expected = (2, cfg.num_classes, cfg.image_size, cfg.image_size)
    if tuple(y.shape) != expected:
        raise AssertionError(f"output shape {tuple(y.shape)} != expected {expected}")

    enc_total = _count(model.encoder_parameters())
    dec_total = _count(model.decoder_parameters())
    enc_train = _trainable(model.encoder_parameters())
    dec_train = _trainable(model.decoder_parameters())
    print(
        f"[freeze={freeze}] enc params={enc_total:,} (trainable={enc_train:,})  "
        f"dec params={dec_total:,} (trainable={dec_train:,})  "
        f"total={enc_total + dec_total:,}"
    )

    if freeze:
        if enc_train != 0:
            raise AssertionError(
                f"frozen encoder must have 0 trainable params, got {enc_train:,}"
            )
    else:
        if enc_train != enc_total:
            raise AssertionError(
                f"unfrozen encoder should be fully trainable: "
                f"{enc_train:,} / {enc_total:,}"
            )
    if dec_train != dec_total:
        raise AssertionError(
            f"decoder must always be trainable: {dec_train:,} / {dec_total:,}"
        )


def main() -> int:
    set_seed(42)
    print("=== freeze_encoder=True ===")
    run_one(freeze=True)
    print("=== freeze_encoder=False ===")
    run_one(freeze=False)
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
