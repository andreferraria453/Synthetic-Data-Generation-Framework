"""Load the pinned upstream configs and only the checkpoints needed for each command."""

import sys
from pathlib import Path

RAE_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "RAE"
RECON_CONFIG = "configs/stage1/pretrained/DINOv2-B_512.yaml"
GEN_CONFIG = "configs/stage2/sampling/ImageNet512/DiTDH-XL_DINOv2-B_decXL_AG.yaml"
CHECKPOINT_REVISION = "1be4f03273523431f099a934da4cf1940dc6039f"


def load_config(relative_path):
    if not (RAE_ROOT / relative_path).is_file():
        raise SystemExit("Run git submodule update --init --recursive from the repository root.")
    from omegaconf import OmegaConf

    sys.path.insert(0, str(RAE_ROOT / "src"))
    return OmegaConf.load(RAE_ROOT / relative_path)


def load_model(config):
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from utils.model_utils import instantiate_from_config

    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    params = config.params
    if "decoder_config_path" in params:
        params.decoder_config_path = str(RAE_ROOT / params.decoder_config_path)

    for section, key in (
        (params, "pretrained_decoder_path"),
        (params, "normalization_stat_path"),
        (config, "ckpt"),
    ):
        if key in section:
            relative_path = section[key].removeprefix("models/")
            section[key] = hf_hub_download(
                "nyu-visionx/RAE-collections",
                filename=relative_path,
                revision=CHECKPOINT_REVISION,
                local_dir=RAE_ROOT / "models",
            )

    return instantiate_from_config(config)
