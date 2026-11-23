"""Prepare trainable DiT attention adapters for a future adaptation loop."""

import argparse

from rae_experiment.models import GEN_CONFIG, load_config, load_model
from rae_experiment.representations import load_rae


def prepare_lora(config, rank, device):
    from peft import LoraConfig, get_peft_model

    rae = load_rae(config, device)
    dit = load_model(config.stage_2).requires_grad_(False)
    dit = get_peft_model(
        dit,
        LoraConfig(r=rank, lora_alpha=rank, target_modules=["qkv"], bias="none"),
    ).to(device)
    return rae, dit.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()
    if args.rank < 1:
        parser.error("--rank must be positive.")

    import torch

    if not torch.cuda.is_available():
        parser.error("A CUDA runtime is required.")
    rae, dit = prepare_lora(load_config(GEN_CONFIG), args.rank, "cuda")
    assert not any(parameter.requires_grad for parameter in rae.parameters())
    dit.print_trainable_parameters()


if __name__ == "__main__":
    main()
