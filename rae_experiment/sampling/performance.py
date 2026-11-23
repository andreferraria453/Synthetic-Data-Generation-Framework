"""Shared CUDA timing and CSV output for the generation scripts."""

import csv
from pathlib import Path
from statistics import pstdev
from time import perf_counter


def add_run_arguments(parser):
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batches", type=int, default=5, help="Number of measured batches")
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=Path("outputs/generation_performance.csv"))
    parser.add_argument("--output-dir", type=Path, help="Optional directory for measured images")


def measure_generation(args, load, *, model, checkpoint, precision, resolution, steps):
    import torch

    if args.batch_size < 1 or args.batches < 1 or args.warmup_batches < 0:
        raise SystemExit("Batch size and batches must be positive; warm-up batches must be nonnegative.")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA runtime is required.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit("This configuration requires a GPU with native BF16 support.")

    def timed(operation):
        torch.cuda.synchronize()
        start = perf_counter()
        result = operation()
        torch.cuda.synchronize()
        return result, perf_counter() - start

    with torch.inference_mode():
        generate, load_seconds = timed(load)
        torch.manual_seed(args.seed)
        for _ in range(args.warmup_batches):
            generate()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(args.seed)

        batch_seconds = []
        peak_bytes = 0
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)

        for batch in range(args.batches):
            images, elapsed = timed(generate)
            peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated())
            if len(images) != args.batch_size:
                raise RuntimeError(f"Expected {args.batch_size} images, got {len(images)}.")
            batch_seconds.append(elapsed)
            if args.output_dir is not None:
                for index, image in enumerate(images, start=batch * args.batch_size):
                    image.save(args.output_dir / f"{index:06d}.png")

    count = args.batch_size * args.batches
    generation_seconds = sum(batch_seconds)
    row = {
        "model": model,
        "checkpoint": checkpoint,
        "gpu": torch.cuda.get_device_name(),
        "precision": precision,
        "resolution": resolution,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "steps": steps,
        "warmup_batches": args.warmup_batches,
        "measured_batches": args.batches,
        "images": count,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "seconds_per_image": generation_seconds / count,
        "seconds_per_image_std": pstdev(batch_seconds) / args.batch_size,
        "images_per_second": count / generation_seconds,
        "peak_vram_mib": peak_bytes / (1024 ** 2),
    }

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("a+", newline="", encoding="utf-8") as file:
        file.seek(0)
        header = next(csv.reader(file), None)
        if header is not None and header != list(row):
            raise ValueError(f"Unexpected CSV columns in {args.csv}")
        writer = csv.DictWriter(file, fieldnames=list(row))
        if header is None:
            writer.writeheader()
        writer.writerow(row)
    print(
        f"{model}: load {load_seconds:.3f}s, generation {generation_seconds:.3f}s, "
        f"{row['seconds_per_image']:.3f} ± {row['seconds_per_image_std']:.3f}s/image, "
        f"{row['images_per_second']:.3f} images/s, "
        f"{row['peak_vram_mib']:.0f} MiB peak. CSV: {args.csv}"
    )
