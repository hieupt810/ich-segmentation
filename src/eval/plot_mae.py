from pathlib import Path

import matplotlib.pyplot as plt
import torch
from lightly.models import utils
from PIL import Image

from src.mae import MAE, MAEConfig, build_transform


def plot_reconstruction(
    image_paths: list[str],
    seed: int = 42,
    show: bool = True,
) -> None:
    cfg = MAEConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MAE(cfg).to(device)
    checkpoint_path = cfg.output_dir / "best_model.pth"
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()

    transform = build_transform(cfg)
    tensors = [transform(Image.open(p).convert("L")) for p in image_paths]
    images = torch.stack(tensors).to(device)

    with torch.no_grad():
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        idx_keep, idx_mask = utils.random_token_mask(
            size=(images.shape[0], model.sequence_length),
            mask_ratio=model.mask_ratio,
            device=device,
        )
        x_encoded = model.forward_encoder(images, idx_keep)
        x_pred = model.forward_decoder(x_encoded, idx_keep, idx_mask)

        patches = utils.patchify(images, model.patch_size)
        recon_patches = utils.set_at_index(patches, idx_mask - 1, x_pred)
        recon = utils.unpatchify(recon_patches, model.patch_size, channels=cfg.in_chans)

    originals = (images * 0.5 + 0.5).clamp(0, 1).squeeze(1).cpu().numpy()
    reconstructions = (recon * 0.5 + 0.5).clamp(0, 1).squeeze(1).cpu().numpy()

    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n), squeeze=False)
    for row, path in enumerate(image_paths):
        axes[row, 0].imshow(originals[row], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(reconstructions[row], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(Path(path).name, fontsize=9)
        for col in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, 0].set_title("Original")
    axes[0, 1].set_title("Reconstruction")
    fig.tight_layout()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / "reconstruction.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
