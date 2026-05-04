import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Calculate query, key, value matrices
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute the output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        *,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        superpatch_size: int = 128,
        patch_size: int = 8,
        in_channels: int = 1,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.superpatch_size = superpatch_size
        self.patch_size = patch_size
        self.grid_size = superpatch_size // patch_size
        self.num_patches = self.grid_size**3

        # Use a 3D convolution to extract patches and project to the embedding dimension
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MaskedAutoEncoder3D(nn.Module):
    def __init__(
        self,
        superpatch_size: int = 128,
        patch_size: int = 8,
        in_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 384,
        decoder_depth: int = 4,
        decoder_num_heads: int = 6,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio

        # --- MAE Encoder ---
        self.patch_embed = PatchEmbed3D(
            superpatch_size=superpatch_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # --- MAE Decoder ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # --- Reconstruction prediction head ---
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**3 * in_channels, bias=True
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones(B, L, device=x.device)
        mask[:, :len_keep] = 0

        # Unshuffle to get the binary mask aligned with the original sequence
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)

        # Add positional embedding excluding the cls token
        x = x + self.pos_embed[:, 1:, :]

        # Apply spatial masking mechanism
        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # Append the cls token to the masked tokens
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_masked.size(0), -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)

        # Apply the Transformer blocks
        for block in self.blocks:
            x_masked = block(x_masked)
        x_masked = self.norm(x_masked)

        return x_masked, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # Map the encoder dim to the decoder dim
        x = self.decoder_embed(x)

        # Prepare the mask tokens
        mask_tokens = self.mask_token.expand(
            x.size(0), ids_restore.size(1) + 1 - x.size(1), -1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)

        # Unshuffle to get the original token order
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.size(2))
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # Add positional embedding
        x = x + self.decoder_pos_embed

        # Apply the Transformer blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        # Reconstruct the original patches
        x = self.decoder_pred(x)
        x = x[:, 1:, :].reshape(x.size(0), self.patch_embed.num_patches, -1)
        return x

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_embed.patch_size
        c = imgs.size(1)
        b, _, d, h, w = imgs.shape
        grid_d, grid_h, grid_w = d // p, h // p, w // p

        x = imgs.reshape(b, c, grid_d, p, grid_h, p, grid_w, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
            b, grid_d * grid_h * grid_w, (p**3) * c
        )
        return x

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        target = self.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred


def smoke_test() -> None:
    model = MaskedAutoEncoder3D()

    input_tensor = torch.randn(2, 1, 128, 128, 128)
    loss, pred = model(input_tensor)

    print(f"Loss: {loss.item():.4f}, Pred shape: {pred.shape}")


if __name__ == "__main__":
    smoke_test()
