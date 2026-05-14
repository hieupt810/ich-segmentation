from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn as nn
from lightly.models.modules import MaskedVisionTransformerTIMM
from timm.models import vision_transformer

from .config import SegmentationConfig


class _ConvNormAct(nn.Sequential):
    def __init__(
        self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1
    ) -> None:
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )


class UnetrBasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, residual: bool = True) -> None:
        super().__init__()
        self.conv1 = _ConvNormAct(in_c, out_c)
        self.conv2 = _ConvNormAct(out_c, out_c)
        self.residual = residual and (in_c == out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        return x


class UnetrPrUpBlock(nn.Module):
    """Project ViT tokens-as-feature-map to a target spatial resolution.

    `num_steps` controls the number of 2x bilinear upsampling stages applied
    before the final channel projection. `num_steps=0` means channel projection
    only (no spatial change), useful for the bottleneck.
    """

    def __init__(self, in_c: int, out_c: int, num_steps: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_c
        for _ in range(num_steps):
            layers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
            layers.append(_ConvNormAct(current, current))
        layers.append(_ConvNormAct(current, out_c))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UnetrUpBlock(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = UnetrBasicBlock(in_c + skip_c, out_c, residual=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class MAESegmenter(nn.Module):
    """UNETR-style 2D segmenter with a ViT backbone pre-trained by MAE."""

    def __init__(self, cfg: SegmentationConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Mirror src/mae/model.py:14,18 so encoder state_dict keys match the
        # MAE trainer's checkpoint format exactly (under the `backbone.` prefix).
        vit = getattr(vision_transformer, cfg.vit_name)(in_chans=cfg.in_chans)
        self.encoder = MaskedVisionTransformerTIMM(vit=vit)

        # Defensive check: forward_intermediates was added in lightly>=1.5.23.
        if not hasattr(self.encoder, "forward_intermediates"):
            raise RuntimeError(
                "Installed lightly is missing MaskedVisionTransformerTIMM."
                "forward_intermediates; require lightly>=1.5.23"
            )

        embed_dim: int = self.encoder.vit.embed_dim
        patch_size: int = self.encoder.vit.patch_embed.patch_size[0]
        if cfg.image_size % patch_size != 0:
            raise ValueError(
                f"image_size {cfg.image_size} not divisible by patch_size {patch_size}"
            )
        self.grid_size: int = cfg.image_size // patch_size

        if len(cfg.skip_block_indices) != 3:
            raise ValueError("skip_block_indices must be a 3-tuple")
        num_blocks = len(self.encoder.vit.blocks)  # ty:ignore[invalid-argument-type]
        if max(cfg.skip_block_indices) >= num_blocks:
            raise ValueError(
                f"skip_block_indices {cfg.skip_block_indices} out of range "
                f"for encoder with {num_blocks} blocks"
            )

        # NOTE: in_chans=1 and image_size=224 match MAE pre-training, so no
        # patch_embed adaptation or pos_embed interpolation is required here.
        # If image_size ever changes, call
        # `timm.layers.resample_abs_pos_embed(self.encoder.vit.pos_embed,
        # (new_g, new_g), num_prefix_tokens=1)` after loading weights.

        fs = cfg.feature_size
        self.stem = UnetrBasicBlock(cfg.in_chans, fs, residual=False)
        self.proj_skip1 = UnetrPrUpBlock(embed_dim, fs * 2, num_steps=3)
        self.proj_skip2 = UnetrPrUpBlock(embed_dim, fs * 4, num_steps=2)
        self.proj_skip3 = UnetrPrUpBlock(embed_dim, fs * 8, num_steps=1)
        self.proj_bottleneck = UnetrPrUpBlock(embed_dim, fs * 16, num_steps=0)

        self.up3 = UnetrUpBlock(fs * 16, fs * 8, fs * 8)
        self.up2 = UnetrUpBlock(fs * 8, fs * 4, fs * 4)
        self.up1 = UnetrUpBlock(fs * 4, fs * 2, fs * 2)
        self.up0 = UnetrUpBlock(fs * 2, fs, fs)

        self.head = nn.Conv2d(fs, cfg.num_classes, kernel_size=1)

        self._encoder_frozen = False
        if cfg.freeze_encoder:
            self.freeze_encoder()
        else:
            # Lightly's MaskedVisionTransformerTIMM disables grads on pos_embed
            # (sincos init is intentional for MAE pretraining). For downstream
            # fine-tuning we want every encoder param trainable; normalize here.
            self.unfreeze_encoder()

    def _tokens_to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, c = tokens.shape
        g = self.grid_size
        if n != 1 + g * g:
            raise RuntimeError(
                f"expected {1 + g * g} tokens (1 CLS + {g * g} patches), got {n}"
            )
        patch = tokens[:, 1:, :]
        return patch.transpose(1, 2).reshape(b, c, g, g)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Requires lightly>=1.5.23 for forward_intermediates.
        final_tokens, intermediates = self.encoder.forward_intermediates(
            images, idx_keep=None, norm=False
        )

        i1, i2, i3 = self.cfg.skip_block_indices
        z_skip1 = self._tokens_to_grid(intermediates[i1])
        z_skip2 = self._tokens_to_grid(intermediates[i2])
        z_skip3 = self._tokens_to_grid(intermediates[i3])
        z_bottleneck = self._tokens_to_grid(final_tokens)

        s_stem = self.stem(images)
        s1 = self.proj_skip1(z_skip1)
        s2 = self.proj_skip2(z_skip2)
        s3 = self.proj_skip3(z_skip3)
        bn = self.proj_bottleneck(z_bottleneck)

        d3 = self.up3(bn, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s_stem)

        return self.head(d0)

    def load_mae_checkpoint(
        self, path: Path | None = None, *, strict: bool = False
    ) -> dict:
        ckpt_path = Path(path) if path is not None else self.cfg.mae_checkpoint_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MAE checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model_state_dict" not in ckpt:
            raise KeyError(
                f"checkpoint at {ckpt_path} has no 'model_state_dict' key; "
                f"keys present: {list(ckpt.keys())}"
            )
        full_sd = ckpt["model_state_dict"]

        prefix = "backbone."
        encoder_sd = {
            k[len(prefix) :]: v for k, v in full_sd.items() if k.startswith(prefix)
        }
        result = self.encoder.load_state_dict(encoder_sd, strict=strict)
        return {
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "epoch": ckpt.get("epoch"),
        }

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self._encoder_frozen = True

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True
        self._encoder_frozen = False
        self.encoder.train(self.training)

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def decoder_parameters(self) -> Iterator[nn.Parameter]:
        encoder_ids = {id(p) for p in self.encoder.parameters()}
        for p in self.parameters():
            if id(p) not in encoder_ids:
                yield p

    def train(self, mode: bool = True) -> "MAESegmenter":
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        return self
