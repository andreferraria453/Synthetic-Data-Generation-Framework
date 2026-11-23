"""Reconstruct a single image with the official 512px RAE."""

import argparse
from pathlib import Path

from rae_experiment.models import RAE_ROOT, RECON_CONFIG, load_config
from rae_experiment.representations import load_rae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=RAE_ROOT / "assets/pixabay_cat.png")
    parser.add_argument("--output", type=Path, default=Path("outputs/reconstruction.png"))
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

    import torch

    if not torch.cuda.is_available():
        parser.error("A CUDA runtime is required.")
    config = load_config(RECON_CONFIG)
    from stage1_sample import load_image, reconstruct, save_image

    with torch.inference_mode():
        image = load_image(args.image).to("cuda")
        rae = load_rae(config, "cuda")
        latent, reconstructed = reconstruct(rae, image)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_image(reconstructed.clamp(0, 1), args.output)
    print(f"{tuple(latent.shape)} -> {tuple(reconstructed.shape)}: {args.output}")


if __name__ == "__main__":
    main()
