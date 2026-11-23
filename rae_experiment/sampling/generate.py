"""Generate one 512px image with the official ImageNet DiT and autoguidance."""

import argparse
import math
from pathlib import Path

from rae_experiment.models import GEN_CONFIG, load_config, load_model
from rae_experiment.representations import load_rae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-id", type=int, default=207, help="ImageNet class index (0–999)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/generated.png"))
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        parser.error("A CUDA runtime is required.")
    config = load_config(GEN_CONFIG)
    if not 0 <= args.class_id < config.misc.num_classes:
        parser.error(f"--class-id must be in [0, {config.misc.num_classes - 1}].")

    from stage2.transport import Sampler, create_transport
    from torchvision.utils import save_image

    torch.manual_seed(args.seed)
    guidance = config.guidance
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    with torch.inference_mode():
        dit = load_model(config.stage_2).eval().to(device="cuda", dtype=precision)
        guide = load_model(guidance.guidance_model).eval().to(device="cuda", dtype=precision)
        shift = math.sqrt(config.misc.time_dist_shift_dim / config.misc.time_dist_shift_base)
        transport = create_transport(**config.transport.params, time_dist_shift=shift)
        sample = Sampler(transport).sample_ode(**config.sampler.params)
        noise = torch.randn(1, *config.misc.latent_size, device="cuda")
        labels = torch.tensor([args.class_id], device="cuda")
        with torch.autocast("cuda", dtype=precision):
            latent = sample(
                noise,
                dit.forward_with_autoguidance,
                y=labels,
                cfg_scale=guidance.scale,
                additional_model_forward=guide.forward,
                cfg_interval=(guidance["t-min"], guidance["t-max"]),
            )[-1]
        del dit, guide
        rae = load_rae(config, "cuda")
        image = rae.decode(latent.float())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_image(image.clamp(0, 1), args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
