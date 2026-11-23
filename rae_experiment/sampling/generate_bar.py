"""Generate images with the official BAR-L-res512 ImageNet model."""

import argparse
import sys
from pathlib import Path

from .performance import add_run_arguments, measure_generation

BAR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "BAR"
COLLECTION = "FAR-Amazon/BAR-collections"
GENERATOR = "generator/bar_l_res512.bin"
TOKENIZER = "tokenizer/bar_sfq_10bits_res512.bin"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--class-id", type=int, default=207)
    args = parser.parse_args()
    if not 0 <= args.class_id < 1000:
        parser.error("--class-id must be in [0, 999].")
    config_path = BAR_ROOT / "configs" / "generator" / "bar_l_res512.yaml"
    if not config_path.is_file():
        parser.error("Run git submodule update --init --recursive first.")

    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    settings = config.model.generator

    def load():
        import torch
        from huggingface_hub import hf_hub_download
        from PIL import Image

        sys.path.insert(0, str(BAR_ROOT))
        from modeling.generator import BAR
        from utils.train_utils import get_pretrained_tokenizer

        config.experiment.tokenizer_checkpoint = hf_hub_download(COLLECTION, TOKENIZER)
        generator_path = hf_hub_download(COLLECTION, GENERATOR)
        tokenizer = get_pretrained_tokenizer(config).to("cuda")
        generator = BAR(config)
        generator.load_state_dict(torch.load(generator_path, map_location="cpu"))
        generator.eval().requires_grad_(False).to("cuda")

        def generate():
            labels = torch.full((args.batch_size,), args.class_id, device="cuda", dtype=torch.long)
            tokens = generator.generate(
                condition=labels,
                guidance_scale=4.2,
                randomize_temperature=2.8,
                kv_cache=True,
                tokens_allocation=list(settings.mbm_head.tokens_allocation),
            )
            pixels = ((tokenizer.decode_tokens(tokens) + 1) / 2).clamp(0, 1)
            pixels = (pixels * 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            return [Image.fromarray(image) for image in pixels]

        return generate

    measure_generation(
        args, load, model="BAR-L-res512",
        checkpoint=f"{COLLECTION}/{GENERATOR}; {COLLECTION}/{TOKENIZER}",
        precision="fp32", resolution="512x512", steps=settings.image_seq_len,
    )


if __name__ == "__main__":
    main()
