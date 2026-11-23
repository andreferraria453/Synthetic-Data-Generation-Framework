"""Run an official SANA Diffusers pipeline with the shared CUDA measurements."""

import argparse

from .performance import add_run_arguments, measure_generation


def run_sana(pipeline_name, *, model, checkpoint, resolution, precision, default_steps):
    parser = argparse.ArgumentParser(description=f"Generate images with {model}.")
    add_run_arguments(parser)
    parser.add_argument("--prompt", default="A photograph of a golden retriever in a park")
    parser.add_argument("--steps", type=int, default=default_steps)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive.")

    def load():
        import diffusers
        import torch

        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        pipeline_class = getattr(diffusers, pipeline_name)
        pipeline = pipeline_class.from_pretrained(checkpoint, torch_dtype=dtype).to("cuda")
        pipeline.set_progress_bar_config(disable=True)

        def generate():
            return pipeline(
                prompt=[args.prompt] * args.batch_size,
                num_inference_steps=args.steps,
                height=resolution,
                width=resolution,
                use_resolution_binning=False,
            ).images

        return generate

    measure_generation(
        args, load, model=model, checkpoint=checkpoint, precision=precision,
        resolution=f"{resolution}x{resolution}", steps=args.steps,
    )
