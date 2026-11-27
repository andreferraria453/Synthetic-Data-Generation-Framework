"""Read the immutable RF100 YOLO splits and cache aligned model features."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

SPLITS = {"train": 840, "valid": 304, "test": 105}
CLASSES = ("Slippage", "corrosion", "crack")
SUFFIXES = {".jpg", ".jpeg", ".png"}
DINO = "facebook/dinov2-small"
DEPTH = "depth-anything/Depth-Anything-V2-Small-hf"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(root):
    root = Path(root)
    extended = root / "extended_test_manifest.json"
    original = json.loads(extended.read_text())["original_test_files"] if extended.is_file() else {}
    manifest, seen = {}, {}
    for split, count in SPLITS.items():
        records = []
        for image in sorted((root / split / "images").iterdir()):
            if image.suffix.lower() not in SUFFIXES:
                continue
            relative = image.relative_to(root).as_posix()
            if split == "test" and original and relative not in original:
                continue
            label = root / split / "labels" / f"{image.stem}.txt"
            image_hash, label_hash = digest(image), digest(label)
            if split == "test" and original and (
                image_hash != original[relative] or label_hash != original[label.relative_to(root).as_posix()]
            ):
                raise ValueError(f"Original test checksum changed: {relative}")
            prior = seen.setdefault(image_hash, split)
            if prior != split:
                raise ValueError(f"Exact duplicate across {prior}/{split}: {relative}")
            records.append((relative, label.relative_to(root).as_posix(), image_hash, label_hash))
        if len(records) != count:
            raise ValueError(f"Expected {count} original {split} images, found {len(records)}.")
        manifest[split] = records
    return manifest


def read_boxes(path, slots):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cls, x, y, width, height = map(float, line.split())
        if cls not in range(len(CLASSES)) or not all(np.isfinite((x, y, width, height))):
            raise ValueError(f"Invalid YOLO label: {path}")
        box = np.clip((x - width / 2, y - height / 2, x + width / 2, y + height / 2), 0, 1)
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f"Degenerate box: {path}")
        rows.append((int(cls), *box))
    rows.sort(key=lambda row: (*row[1:], row[0]))
    if len(rows) > slots:
        raise ValueError(f"{path} has {len(rows)} objects. Increase the slot count ({slots}).")
    boxes = torch.zeros(slots, 4)
    classes = torch.full((slots,), len(CLASSES), dtype=torch.long)
    for index, (cls, *box) in enumerate(rows):
        boxes[index] = torch.tensor(box)
        classes[index] = cls
    return boxes, classes


class CorrosionDataset(Dataset):
    def __init__(self, root, records, *, size, slots, cache=None, modalities=(), depth_size=64):
        self.root = Path(root)
        self.records = records
        self.size = size
        self.slots = slots
        self.cache = Path(cache) if cache is not None else None
        self.modalities = tuple(modalities)
        self.depth_size = depth_size

    def cache_key(self, index):
        _, _, image_hash, label_hash = self.records[index]
        key = (1, image_hash, label_hash, self.size, self.slots, self.depth_size, DINO, DEPTH)
        return hashlib.sha256(repr(key).encode()).hexdigest()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, label, *_ = self.records[index]
        with Image.open(self.root / image) as source:
            rgb = np.array(source.convert("RGB").resize((self.size, self.size), Image.Resampling.BICUBIC))
        boxes, classes = read_boxes(self.root / label, self.slots)
        item = {
            "rgb": torch.from_numpy(rgb).permute(2, 0, 1).float().div(127.5).sub(1),
            "boxes": boxes, "box_classes": classes, "presence": classes != len(CLASSES),
        }
        features = set(self.modalities) & {"dino", "depth"}
        if features:
            path = self.cache / f"{self.cache_key(index)}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Precompute DINO/depth first. Missing {path}")
            cached = torch.load(path, weights_only=True)
            item.update({name: cached[name] for name in features})
        return item


def load_encoders(device, modalities, offline):
    from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation

    result = {}
    for name, identifier, cls in (("dino", DINO, AutoModel), ("depth", DEPTH, AutoModelForDepthEstimation)):
        if name in modalities:
            processor = AutoImageProcessor.from_pretrained(identifier, local_files_only=offline)
            model = cls.from_pretrained(identifier, local_files_only=offline)
            result[name] = processor, model.requires_grad_(False).eval().to(device)
    return result


@torch.inference_mode()
def extract(item, encoders, device, depth_size):
    from torchvision.ops import roi_align

    rgb = item["rgb"].add(1).mul(127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    result = {}
    for name, (processor, encoder) in encoders.items():
        if name == "dino":
            inputs = processor(images=rgb, do_center_crop=False, size={"height": 518, "width": 518},
                               return_tensors="pt").to(device)
            tokens = encoder(**inputs).last_hidden_state
            patch = encoder.config.patch_size
            height, width = (dim // patch for dim in inputs.pixel_values.shape[-2:])
            features = tokens[:, -height * width:].transpose(1, 2).reshape(1, -1, height, width)
            keep = item["presence"]
            boxes = item["boxes"][keep].to(device) * features.new_tensor((width, height, width, height))
            local = roi_align(features.float(), [boxes], output_size=2, aligned=True).mean((-1, -2))
            aligned = torch.zeros((len(keep), features.shape[1]))
            aligned[keep] = F.normalize(local, dim=-1).cpu() * features.shape[1] ** 0.5
            result["dino"] = aligned
        else:
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            raw = encoder(**inputs).predicted_depth[:, None].float()
            depth = F.interpolate(raw, (depth_size, depth_size), mode="bilinear", align_corners=False)[0]
            scale = depth.amax() - depth.amin()
            result["depth"] = ((depth - depth.amin()) / scale.clamp_min(1e-6)).cpu()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--depth-size", type=int, default=64)
    parser.add_argument("--modalities", nargs="+", choices=("dino", "depth"), default=("dino", "depth"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Maximum samples per split for precompute smoke.")
    args = parser.parse_args()
    manifest = build_manifest(args.dataset)
    for split, records in manifest.items():
        for _, label, *_ in records:
            read_boxes(args.dataset / label, args.slots)
        print(f"{split}: {len(records)} original images")
    if args.audit_only:
        return
    if args.cache is None:
        parser.error("--cache is required to precompute.")
    args.cache.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = load_encoders(device, args.modalities, args.offline)
    for split in ("train", "valid"):
        dataset = CorrosionDataset(args.dataset, manifest[split], size=512, slots=args.slots,
                                   depth_size=args.depth_size)
        for index in range(min(len(dataset), args.limit or len(dataset))):
            destination = args.cache / f"{dataset.cache_key(index)}.pt"
            cached = torch.load(destination, weights_only=True) if destination.is_file() else {}
            missing = {name: encoder for name, encoder in encoders.items() if name not in cached}
            if missing:
                cached.update(extract(dataset[index], missing, device, args.depth_size))
                temporary = destination.with_suffix(".tmp")
                torch.save(cached, temporary)
                temporary.replace(destination)


if __name__ == "__main__":
    main()
