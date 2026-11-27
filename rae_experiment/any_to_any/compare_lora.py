"""Compare shared LoRA and UniVidX-style DGL on 16 annotated RF100 images."""

import argparse
from collections import defaultdict
from pathlib import Path
import random
from time import perf_counter

import torch
from torchvision.ops import box_iou

from .data import CorrosionDataset, build_manifest
from .flow import RGB_BOX_TASKS, generate, loss
from .metrics import box_outputs
from .model import AnyToAnyPMF, load_backbone
from .preflight import audit_rgb
from .rgb_boxes_metrics import panel
from .train import write_json


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model, batches, output, device, seed):
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    scores = {}
    empty = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
    for task_name, task in RGB_BOX_TASKS.items():
        elapsed, true, count, expected, overlap = 0.0, 0, 0, 0, 0.0
        for index, batch in enumerate(batches):
            rgb = batch["rgb"]
            targets = {name: batch[name].shape[1:] for name in task.targets}
            conditions = {name: batch[name] for name in task.conditions}

            def sample():
                rng = torch.Generator(device=device).manual_seed(seed + index)
                return generate(model, conditions, targets, batch_size=len(rgb), generator=rng, metadata=batch)

            if index == 0:
                sample()  # Warm up this task; the measured run repeats its seed.
            synchronize(device)
            start = perf_counter()
            prediction, logits = sample()
            synchronize(device)
            elapsed += perf_counter() - start
            if not all(value.isfinite().all() for value in (*prediction.values(), *logits.values())):
                raise FloatingPointError(f"Nonfinite inference results: {task_name}")
            decoded = (box_outputs(prediction["boxes"], logits["boxes"], model.classes)
                       if "boxes" in targets else [empty] * len(rgb))
            for i, predicted in enumerate(decoded):
                mask = batch["presence"][i]
                actual = {"boxes": batch["boxes"][i, mask].cpu(),
                          "labels": batch["box_classes"][i, mask].cpu()}
                if task_name == "rgb_to_boxes":
                    count += len(predicted["boxes"])
                    expected += len(actual["boxes"])
                    overlaps = box_iou(predicted["boxes"], actual["boxes"])
                    overlaps *= predicted["labels"][:, None] == actual["labels"][None, :]
                    matched = torch.zeros(len(actual["boxes"]), dtype=torch.bool)
                    for slot in predicted["scores"].argsort(descending=True).tolist():
                        if matched.numel() == 0:
                            break
                        quality, target = overlaps[slot].masked_fill(matched, -1).max(0)
                        if quality >= 0.5:
                            matched[target] = True
                            true += 1
                            overlap += quality.item()
                if index == 0 and i == 0:
                    panel(rgb[i], actual, empty, "RF100 treino | referência").save(output / "reference.png")
                    generated = prediction.get("rgb", rgb)
                    overlay = actual if task_name in ("rgb_to_boxes", "boxes_to_rgb") else empty
                    panel(generated[i], overlay, predicted, task_name).save(output / f"{task_name}.png")
        row = {"inference_seconds": elapsed,
               "images_per_second": sum(len(batch["rgb"]) for batch in batches) / elapsed}
        if task_name == "rgb_to_boxes":
            row.update(precision50=true / max(count, 1), recall50=true / max(expected, 1),
                       mean_iou_matched=overlap / max(true, 1),
                       predictions=count, annotations=expected)
        scores[task_name] = row
    return scores


def prototype(args):
    if not (1 <= args.interval <= args.steps and 2 <= args.batch_size <= 4):
        raise ValueError("Use a batch size of 2–4 and a reporting interval within the step budget.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Run the production comparison on a CUDA runtime.")
    torch.set_num_threads(args.threads)
    manifest = build_manifest(args.dataset)
    annotated = [row for row in manifest["train"] if (args.dataset / row[1]).read_text().strip()]
    selected = random.Random(args.seed).sample(annotated, 16)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json({"seed": args.seed, "batch_size": args.batch_size, "steps": args.steps,
                "backbone": str(args.backbone) if args.backbone else "pMF-B-32 (pinned cache)",
                "resolution": "fixture" if args.fixture else 512,
                "split": "train (overfit)", "records": selected}, args.output / "subset.json")
    for label, adapter, rank in (("A_shared", "shared", 16), ("B_dgl", "dgl", 8)):
        backbone = load_backbone(args.backbone, offline=True, strict=True, variant="pMF-B-32")
        if not args.fixture and (backbone.sample_size, backbone.patch_size, backbone.hidden_size, backbone.depth) != (
            512, 32, 768, 16,
        ):
            raise ValueError("Expected a pretrained pMF-B/32-512 inference backbone.")
        if backbone.v_final_layer is not None:
            raise ValueError("Expected the pMF inference checkpoint.")
        torch.manual_seed(args.seed)
        model = AnyToAnyPMF(backbone, ("rgb", "boxes"), rank=rank, adapter=adapter).to(device)
        model.rgb_noise_scale = 2.0
        model.pmf.gradient_checkpointing = True
        dataset = CorrosionDataset(args.dataset, selected, size=backbone.sample_size, slots=32)
        batches = [{key: value.to(device) for key, value in batch.items()}
                   for batch in torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)]
        initial = audit_rgb(model, {key: value[:1] for key, value in batches[0].items()},
                            omega=6.5, guidance_min=0.1, guidance_max=0.7)
        trainable = [p for p in model.parameters() if p.requires_grad]
        metadata = {"adapter": adapter, "rank": rank, "trainable": sum(p.numel() for p in trainable),
                    "total": sum(p.numel() for p in model.parameters()), "rgb_audit": initial}
        directory = args.output / label
        write_json(metadata, directory / "protocol.json")
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        rng = torch.Generator(device=device).manual_seed(args.seed)
        order = random.Random(args.seed)
        tasks = tuple(RGB_BOX_TASKS.values())
        loss_sums, loss_counts = defaultdict(float), defaultdict(int)
        gradient_norms = []
        train_seconds = 0.0
        indices = list(range(len(batches)))
        for step in range(args.steps + 1):
            if step:
                if (step - 1) % len(indices) == 0:
                    order.shuffle(indices)
                batch = batches[indices[(step - 1) % len(indices)]]
                task = tasks[(step - 1) % len(tasks)]
                model.train()
                synchronize(device)
                start = perf_counter()
                optimizer.zero_grad(set_to_none=True)
                total, parts = loss(model, batch, task, generator=rng, balanced=True)
                if not all(value.isfinite().item() for value in (total, *parts.values())):
                    raise FloatingPointError(f"{label} step {step}: nonfinite loss")
                total.backward()
                grad = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True).item()
                optimizer.step()
                synchronize(device)
                train_seconds += perf_counter() - start
                gradient_norms.append(grad)
                for name, value in ({"global": total} | parts).items():
                    loss_sums[name] += value.item()
                    loss_counts[name] += 1
            if step % args.interval == 0 or step == args.steps:
                row = {"step": step, "train_seconds": train_seconds,
                       "loss": {name: value / loss_counts[name] for name, value in loss_sums.items()},
                       "gradient_norm_mean": sum(gradient_norms) / len(gradient_norms) if gradient_norms else None,
                       "gradient_norm_max": max(gradient_norms, default=None),
                       "inference": evaluate(model, batches, directory / f"{step:04}", device, args.seed)}
                write_json(row, directory / f"{step:04}.json")
                print(f"{label}: {step}/{args.steps} | {row['loss']} | {train_seconds:.2f} s", flush=True)
                loss_sums.clear()
                loss_counts.clear()
                gradient_norms.clear()
        del model, backbone, optimizer, trainable, batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, help="Local transformer directory.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fixture", action="store_true", help="Allow a reduced test backbone.")
    prototype(parser.parse_args())


if __name__ == "__main__":
    main()
