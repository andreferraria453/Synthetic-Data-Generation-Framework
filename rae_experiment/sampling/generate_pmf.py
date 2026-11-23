"""Generate ImageNet samples with the published pMF-L/32-512 Diffusers pipeline."""

import argparse
from pathlib import Path

from .performance import add_run_arguments, measure_generation

CHECKPOINT = "BiliSakura/pMF-diffusers"
REVISION = "ee50c8763200f26b0656c396315909b762be6191"
VARIANT = "pMF-L-32"
STEPS = 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--class-id", type=int, default=207)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    args = parser.parse_args()
    if not 0 <= args.class_id < 1000:
        parser.error("--class-id must be in [0, 999].")

    def load():
        import torch
        from diffusers.utils.dynamic_modules_utils import get_class_from_dynamic_module
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            CHECKPOINT,
            revision=REVISION,
            allow_patterns=[
                f"{VARIANT}/pipeline.py",
                f"{VARIANT}/scheduler/scheduling_pmf.py",
                f"{VARIANT}/transformer/*",
            ],
        )
        model_dir = Path(snapshot) / VARIANT
        # Load the pinned classes explicitly. Diffusers 0.35 cannot infer this
        # pipeline's constructor types from its postponed annotations.
        pipeline_class = get_class_from_dynamic_module(
            str(model_dir), "pipeline.py", class_name="PMFPipeline", local_files_only=True,
        )
        transformer_dir = str(model_dir / "transformer")
        transformer_class = get_class_from_dynamic_module(
            transformer_dir, "transformer_pmf.py",
            class_name="PMFTransformer2DModel", local_files_only=True,
        )
        transformer = transformer_class.from_pretrained(
            transformer_dir, torch_dtype=torch.float32, local_files_only=True,
        )
        # The pipeline reads the next timestep, including the terminal zero.
        scheduler_class = get_class_from_dynamic_module(
            str(model_dir / "scheduler"), "scheduling_pmf.py",
            class_name="PMFScheduler", local_files_only=True,
        )
        pipeline = pipeline_class(transformer=transformer, scheduler=scheduler_class()).to("cuda")
        pipeline.set_progress_bar_config(disable=True)

        def generate():
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.precision == "fp16"):
                return pipeline(
                    class_labels=[args.class_id] * args.batch_size,
                    num_inference_steps=STEPS,
                    guidance_scale=7.5,
                    guidance_interval_min=0.2,
                    guidance_interval_max=0.6,
                    noise_scale=4.0,
                ).images

        return generate

    measure_generation(
        args, load, model="pMF-L/32-512",
        checkpoint=f"{CHECKPOINT}/{VARIANT}@{REVISION}",
        precision=args.precision, resolution="512x512", steps=STEPS,
    )


if __name__ == "__main__":
    main()
