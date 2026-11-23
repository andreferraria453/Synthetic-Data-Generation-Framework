"""Load the frozen DINOv2 encoder and ViT-XL decoder together."""

from rae_experiment.models import load_model


def load_rae(config, device):
    return load_model(config.stage_1).requires_grad_(False).eval().to(device)
