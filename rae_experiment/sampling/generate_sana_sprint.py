"""Generate images with SANA-Sprint-0.6B at 1024 pixels."""

from ._sana import run_sana


if __name__ == "__main__":
    run_sana(
        "SanaSprintPipeline",
        model="SANA-Sprint-0.6B-1024",
        checkpoint="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        resolution=1024,
        precision="bf16",
        default_steps=2,
    )
