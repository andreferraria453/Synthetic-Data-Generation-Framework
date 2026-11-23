"""Generate images with SANA-0.6B at 512 pixels."""

from ._sana import run_sana


if __name__ == "__main__":
    run_sana(
        "SanaPipeline",
        model="SANA-0.6B-512",
        checkpoint="Efficient-Large-Model/Sana_600M_512px_diffusers",
        resolution=512,
        precision="fp16",
        default_steps=20,
    )
