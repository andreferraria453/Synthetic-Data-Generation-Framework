"""Run validation, a bounded preflight, or the RF100 adaptation experiment."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

import torch
from torch.utils.data import DataLoader, Subset

from .data import CorrosionDataset, build_manifest
from .flow import Task, choose_task, edit_region, generate, loss
from .model import AnyToAnyPMF, load_backbone


def signature(config, manifest):
    stable = json.dumps([config, manifest], sort_keys=True).encode()
    return hashlib.sha256(stable).hexdigest()


def move(item, device):
    return {name: value.to(device) for name, value in item.items()}


def save_atomic(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def trainable_state(model):
    return {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}


def precision_context(device, use_bf16):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16)


def smoke(model, train, valid, optimizer, output, experiment_id, use_bf16, device, seed):
    """Exercise training gradients and joint/conditional sampling on the real batch."""
    model.train()
    keys = model.modalities
    batch = move(next(iter(train)), device)
    val_batch = move(next(iter(valid)), device)
    rng = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tasks = [Task((), tuple(keys))]
    for name in keys:
        rest = tuple(key for key in keys if key != name)
        if rest:
            tasks.extend((Task(rest, (name,)), Task((name,), rest)))
    if "rgb" in keys:
        tasks.extend((Task((), ("rgb",)), Task((), ("rgb",), edit=True)))
    rows = []
    for task in tasks:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with precision_context(device, use_bf16):
            total, _ = loss(model, batch, task, generator=rng)
        if not torch.isfinite(total):
            raise FloatingPointError(f"Nonfinite smoke loss for {task}")
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True,
        )
        if grad.item() <= 0:
            raise RuntimeError(f"No gradient for {task}")
        optimizer.step()
        model.eval()
        edit = edit_region(val_batch["rgb"], val_batch["boxes"], val_batch["presence"]) if task.edit else None
        outputs, _ = generate(
            model, {name: val_batch[name] for name in task.conditions},
            {name: val_batch[name].shape[1:] for name in task.targets},
            batch_size=len(val_batch["rgb"]), generator=rng, metadata=val_batch,
            edit=edit, original_rgb=val_batch["rgb"] if task.edit else None,
        )
        for name, value in outputs.items():
            if value.shape != val_batch[name].shape or not value.isfinite().all():
                raise RuntimeError(f"Invalid smoke output for {name}: {task}")
        if edit is not None and not torch.equal(
            torch.where(edit, val_batch["rgb"], outputs["rgb"]), val_batch["rgb"],
        ):
            raise RuntimeError("Editing changed known pixels.")
        rows.append({"conditions": task.conditions, "targets": task.targets, "edit": task.edit,
                     "loss": total.item(), "gradient_norm": grad.item()})
    report = {"signature": experiment_id, "device": str(device), "bf16": use_bf16,
              "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
              "tasks": rows}
    write_json(report, output / "smoke.json")
    print(json.dumps(report, indent=2))


def epoch_loss(model, loader, rng, task_rng, device, use_bf16, *, optimizer=None, accumulation=1):
    training = optimizer is not None
    model.train(training)
    values, counts, usage = defaultdict(float), Counter(), Counter()
    params = [param for param in model.parameters() if param.requires_grad]
    for index, item in enumerate(loader):
        batch = move(item, device)
        task = choose_task(model.modalities, task_rng)
        with torch.set_grad_enabled(training), precision_context(device, use_bf16):
            total, components = loss(model, batch, task, generator=rng)
        if not torch.isfinite(total):
            raise FloatingPointError(f"Nonfinite loss at batch {index} for {task}")
        size = len(batch["rgb"])
        for name, score in {"global": total} | components.items():
            values[name] += score.detach().item() * size
            counts[name] += size
        for name in task.conditions:
            usage[f"condition_{name}"] += size
        for name in task.targets:
            usage[f"target_{name}"] += size
        if task.edit:
            usage["condition_rgb_visible"] += size
        if training:
            group_size = min(accumulation, len(loader) - index // accumulation * accumulation)
            (total / group_size).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    result = {f"loss_{name}": values[name] / counts[name] for name in counts}
    for role in ("condition", "target"):
        for name in model.modalities:
            key = f"{role}_{name}"
            result[f"{key}_frequency"] = usage[key] / counts["global"]
    if "rgb" in model.modalities:
        result["condition_rgb_visible_frequency"] = usage["condition_rgb_visible"] / counts["global"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, help="Local pinned Diffusers transformer directory.")
    parser.add_argument("--detector", type=Path, help="Local train-only RF100 detector.")
    parser.add_argument("--offline", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = build_manifest(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    if args.train and device.type != "cuda":
        parser.error("The full experiment requires CUDA.")
    experiment_id = signature(config | {"backbone": str(args.backbone.resolve()) if args.backbone else None},
                              manifest)
    if args.train:
        receipt = args.output / "smoke.json"
        report = json.loads(receipt.read_text()) if receipt.is_file() else {}
        if report.get("signature") != experiment_id or report.get("device") != "cuda":
            parser.error("Complete --smoke on CUDA with this config, checkpoint and dataset first.")
        if args.resume is None and (args.output / "epochs").exists():
            parser.error("Use --resume or choose a new run directory.")
    torch.manual_seed(config["seed"])
    backbone = load_backbone(args.backbone, args.offline)
    if (backbone.sample_size, backbone.patch_size, backbone.hidden_size, backbone.depth) != (512, 32, 1024, 32):
        parser.error("Expected the pMF-L/32-512 transformer.")
    model = AnyToAnyPMF(backbone, **config["model"]).to(device)
    model.pmf.gradient_checkpointing = config["gradient_checkpointing"]
    data = {split: CorrosionDataset(
        args.dataset, manifest[split], size=512, slots=model.slots, cache=args.cache,
        modalities=model.modalities, depth_size=model.depth_size,
    ) for split in ("train", "valid")}
    microbatch = config["microbatch"]
    if microbatch < 1 or config["effective_batch_size"] % microbatch:
        parser.error("microbatch must divide effective_batch_size.")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"],
    )
    if args.smoke:
        split_loaders = [DataLoader(Subset(data[split], range(microbatch)), batch_size=microbatch)
                         for split in ("train", "valid")]
        smoke(model, *split_loaders, optimizer, args.output, experiment_id, use_bf16, device, config["seed"])
        return
    from .metrics import Validator

    training = DataLoader(data["train"], batch_size=microbatch, shuffle=True,
                          generator=torch.Generator().manual_seed(config["seed"]), drop_last=True)
    validation = DataLoader(data["valid"], batch_size=1)
    reference_ids = random.Random(config["validation_seed"]).sample(
        range(len(data["train"])), len(data["valid"]),
    )
    reference = DataLoader(Subset(data["train"], reference_ids), batch_size=1)
    validator = Validator(model, reference, device, offline=args.offline,
                          detector_path=args.detector, seed=config["validation_seed"])
    rng = torch.Generator(device=device).manual_seed(config["seed"])
    tasks = random.Random(config["seed"])
    first_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=True, map_location="cpu")
        if checkpoint["signature"] != experiment_id or checkpoint["weights"].keys() != trainable_state(model).keys():
            raise ValueError("Checkpoint does not match the dataset, model and configuration.")
        model.load_state_dict(checkpoint["weights"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.set_state(checkpoint["flow_rng"])
        tasks.setstate(checkpoint["task_rng"])
        training.generator.set_state(checkpoint["shuffle_rng"])
        first_epoch = checkpoint["epoch"] + 1

    for epoch in range(first_epoch, config["epochs"] + 1):
        torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        train_scores = epoch_loss(model, training, rng, tasks, device, use_bf16, optimizer=optimizer,
                                  accumulation=config["effective_batch_size"] // microbatch)
        torch.cuda.synchronize(device)
        timing = {"train_seconds": perf_counter() - start,
                  "train_peak_vram_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        torch.cuda.reset_peak_memory_stats(device)
        val_seed = config["validation_seed"]
        val_scores = epoch_loss(model, validation, torch.Generator(device=device).manual_seed(val_seed),
                                random.Random(val_seed), device, use_bf16)
        scores = validator.evaluate(model, validation, args.output / "examples" / f"{epoch:03}")
        timing["validation_peak_vram_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        row = {"epoch": epoch, "train": train_scores, "validation": val_scores, "metrics": scores} | timing
        write_json(row, args.output / "epochs" / f"{epoch:03}.json")
        if epoch % config["checkpoint_every"] == 0 or epoch == config["epochs"]:
            save_atomic({
                "epoch": epoch, "signature": experiment_id, "weights": trainable_state(model),
                "optimizer": optimizer.state_dict(), "flow_rng": rng.get_state(),
                "task_rng": tasks.getstate(), "shuffle_rng": training.generator.get_state(),
            }, args.output / "checkpoints" / f"epoch-{epoch:03}.pt")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
