import torch
import torch.nn as nn
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from timm.models.vision_transformer import vit_small_patch32_224

from .config import MAEConfig


class MAE(nn.Module):
    def __init__(self, cfg: MAEConfig) -> None:
        super().__init__()

        vit = vit_small_patch32_224(in_chans=cfg.in_chans)
        self.mask_ratio = cfg.mask_ratio
        self.patch_size = vit.patch_embed.patch_size[0]

        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=cfg.decoder_dim,
            decoder_depth=cfg.decoder_depth,
            decoder_num_heads=cfg.decoder_num_heads,
            mlp_ratio=cfg.mlp_ratio,
            proj_drop_rate=cfg.proj_drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
        )

    def forward_encoder(
        self, images: torch.Tensor, idx_keep: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode only the *visible* (unmasked) patches."""
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(
        self, x_encoded: torch.Tensor, idx_keep: torch.Tensor, idx_mask: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct pixel values for the masked patches."""
        batch_size = x_encoded.shape[0]

        # Project encoder output to decoder embedding space
        x_decode = self.decoder.embed(x_encoded)

        # Build full-length sequence filled with the learnable mask token
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )

        # Place the encoded visible tokens back at their original positions
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # Full transformer decoder pass over the complete sequence
        x_decoded = self.decoder.decode(x_masked)

        # Extract and project only the masked positions → pixel predictions
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.shape[0]

        # Sample random mask: idx_keep = visible indices, idx_mask = masked indices
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )

        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # Ground-truth: raw pixel patches for the masked positions
        patches = utils.patchify(images, self.patch_size)

        # idx_mask is offset by 1 to skip the CLS token that the decoder prepends
        target = utils.get_at_index(patches, idx_mask - 1)

        return x_pred, target
