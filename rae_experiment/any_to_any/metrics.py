"""Deterministic validation protocols for the original RF100 validation split."""

from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from .flow import edit_region, generate


def frechet(left, right):
    """Bures term via the cross-Gram matrix, including rank-deficient samples."""
    left, right = left.double(), right.double()
    mean = (left.mean(0) - right.mean(0)).square().sum()
    x = (left - left.mean(0)) / (len(left) - 1) ** 0.5
    y = (right - right.mean(0)) / (len(right) - 1) ** 0.5
    return (mean + x.square().sum() + y.square().sum()
            - 2 * torch.linalg.svdvals(x @ y.T).sum()).clamp_min(0).item()


def kid(left, right):
    left, right = left.double(), right.double()
    k_left = (left @ left.T / left.shape[1] + 1).pow(3)
    k_right = (right @ right.T / left.shape[1] + 1).pow(3)
    mixed = (left @ right.T / left.shape[1] + 1).pow(3)
    return ((k_left.sum() - k_left.trace()) / (len(left) * (len(left) - 1))
            + (k_right.sum() - k_right.trace()) / (len(right) * (len(right) - 1))
            - 2 * mixed.mean()).item()


def pixels(rgb):
    return rgb.add(1).mul(127.5).round().clamp(0, 255).byte()


def box_outputs(boxes, logits, no_object):
    label_probs = logits.float().softmax(-1)
    scores, labels = label_probs[..., :no_object].max(-1)
    keep = label_probs.argmax(-1) != no_object
    boxes = boxes.float().clamp(0, 1)
    boxes = torch.cat((torch.minimum(boxes[..., :2], boxes[..., 2:]),
                       torch.maximum(boxes[..., :2], boxes[..., 2:])), dim=-1)
    return [dict(boxes=coords[mask].cpu(), scores=score[mask].cpu(), labels=label[mask].cpu())
            for coords, score, label, mask in zip(boxes, scores, labels, keep)]


class Validator:
    def __init__(self, model, train_reference, device, *, offline, detector_path, seed):
        self.modalities = model.modalities
        self.device = device
        self.seed = seed
        self.real_inception = []
        self.real_dino = []
        if "boxes" in self.modalities:
            from torchmetrics.detection import MeanAveragePrecision

            self.ap = MeanAveragePrecision(box_format="xyxy")
        if "rgb" in self.modalities:
            import lpips
            from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
            from transformers import AutoModel

            from .data import DINO

            self.inception = FeatureExtractorInceptionV3(
                "inception-v3-compat", ["2048"],
            ).eval().to(device)
            self.dino = AutoModel.from_pretrained(DINO, local_files_only=offline).eval().to(device)
            self.lpips = lpips.LPIPS(net="alex", spatial=True).eval().to(device)
            self.reference_dino = torch.cat([self.dino_features(item["rgb"].to(device)) for item in train_reference])
        if {"rgb", "boxes"} <= set(self.modalities) and detector_path is not None:
            from torchmetrics.detection import MeanAveragePrecision
            from ultralytics import YOLO

            if not Path(detector_path).is_file():
                raise FileNotFoundError(detector_path)
            self.detector = YOLO(str(detector_path))
            self.consistency = MeanAveragePrecision(box_format="xyxy")

    @torch.inference_mode()
    def dino_features(self, rgb):
        value = F.interpolate(rgb.float().add(1).mul(0.5), (224, 224), mode="bilinear",
                              align_corners=False, antialias=True)
        mean = value.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = value.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return self.dino(pixel_values=(value - mean) / std).last_hidden_state[:, 0].cpu()

    @torch.inference_mode()
    def evaluate(self, model, loader, example_dir):
        from torchvision.utils import save_image

        rng = torch.Generator(device=self.device).manual_seed(self.seed)
        sums, counts = {}, {}
        generated_inception, generated_dino = [], []

        def accumulate(name, values):
            values = values.detach().flatten()
            sums[name] = sums.get(name, 0) + values.sum().item()
            counts[name] = counts.get(name, 0) + values.numel()

        for index, item in enumerate(loader):
            batch = {key: value.to(self.device) for key, value in item.items()}
            rgb = batch["rgb"]
            size = len(rgb)
            shapes = {name: batch[name].shape[1:] for name in self.modalities}
            kwargs = dict(batch_size=size, generator=rng, metadata=batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = perf_counter()
            outputs, logits = generate(model, {}, shapes, **kwargs)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            accumulate("inference_seconds_per_image", rgb.new_full((size,), (perf_counter() - start) / size))
            if "rgb" in shapes:
                unconditional, _ = generate(model, {}, {"rgb": shapes["rgb"]}, **kwargs)
                generated = unconditional["rgb"].clamp(-1, 1)
                generated_inception.append(self.inception(pixels(generated))[0].cpu())
                generated_dino.append(self.dino_features(generated))
                if len(self.real_inception) <= index:
                    self.real_inception.append(self.inception(pixels(rgb))[0].cpu())
                    self.real_dino.append(self.dino_features(rgb))
                region = edit_region(rgb, batch["boxes"], batch["presence"])
                edited, edit_details = generate(model, {}, {"rgb": shapes["rgb"]}, **kwargs,
                                                edit=region, original_rgb=rgb)
                edited = edited["rgb"].clamp(-1, 1)
                spatial = self.lpips(edit_details["rgb_edit_raw"].clamp(-1, 1), rgb)
                for name, mask in (("inside", region), ("outside", ~region)):
                    count = mask.flatten(1).sum(1)
                    numer = (spatial * mask).flatten(1).sum(1)
                    accumulate(f"lpips_{name}", numer[count > 0] / count[count > 0])
                if index == 0:
                    example_dir.mkdir(parents=True, exist_ok=True)
                    montage = torch.cat((rgb[:1], generated[:1], edited[:1])).add(1).mul(0.5)
                    save_image(montage, example_dir / "rgb_reference_generated_edit.png", nrow=3)

            if hasattr(self, "detector"):
                predicted = box_outputs(outputs["boxes"], logits["boxes"], model.classes)
                images = [image.permute(1, 2, 0).cpu().numpy()[..., ::-1].copy()
                          for image in pixels(outputs["rgb"])]
                results = self.detector.predict(images, conf=0.001, verbose=False, device=str(self.device))
                if tuple(str(results[0].names[i]).lower() for i in range(model.classes)) != (
                    "slippage", "corrosion", "crack",
                ):
                    raise ValueError("Detector class order must match the RF100 export.")
                detected = [dict(boxes=result.boxes.xyxyn.cpu(), labels=result.boxes.cls.long().cpu(),
                                 scores=result.boxes.conf.cpu()) for result in results]
                self.consistency.update(detected, [{k: v for k, v in sample.items() if k != "scores"}
                                                   for sample in predicted])

            paired = {name: shape for name, shape in shapes.items() if name != "rgb"}
            if paired:
                context = {"rgb": rgb} if "rgb" in shapes else {}
                recovered, recovered_logits = generate(model, context, paired, **kwargs)
                has_objects = batch["presence"]
                if "boxes" in paired:
                    expected = [dict(boxes=box[mask].cpu(), labels=cls[mask].cpu())
                                for box, cls, mask in zip(batch["boxes"], batch["box_classes"], has_objects)]
                    self.ap.update(box_outputs(recovered["boxes"], recovered_logits["boxes"], model.classes), expected)
                if "dino" in paired:
                    pred, target = recovered["dino"][has_objects], batch["dino"][has_objects]
                    accumulate("dino_cosine", F.cosine_similarity(pred, target, dim=-1))
                    accumulate("dino_reconstruction_mse", (pred - target).square().mean(-1))
                if "depth" in paired:
                    target = batch["depth"]
                    delta = recovered["depth"].clamp(0, 1) - target
                    accumulate("depth_absrel", (delta.abs() / target.clamp_min(1e-3))[target > 1e-3])
                    accumulate("depth_mse", delta.square())
        scores = {name: total / counts[name] for name, total in sums.items() if counts[name]}
        if "depth_mse" in scores:
            scores["depth_rmse"] = scores.pop("depth_mse") ** 0.5
        if "rgb" in self.modalities:
            true_features, fake_features = torch.cat(self.real_inception), torch.cat(generated_inception)
            scores.update(fid=frechet(true_features, fake_features), kid=kid(true_features, fake_features))
            true_dino = torch.cat(self.real_dino)
            denominator = frechet(self.reference_dino, true_dino)
            numerator = frechet(torch.cat(generated_dino), true_dino)
            scores["fd_dino"] = numerator
            scores["fdr_dino"] = numerator / denominator if denominator > 0 else None
        for prefix, metric in (("boxes", getattr(self, "ap", None)),
                               ("box_rgb_consistency", getattr(self, "consistency", None))):
            if metric is not None:
                results = metric.compute()
                scores[f"{prefix}_map50"] = results["map_50"].item() if results["map_50"].item() >= 0 else None
                scores[f"{prefix}_map50_95"] = results["map"].item() if results["map"].item() >= 0 else None
                metric.reset()
        if {"rgb", "boxes"} <= set(self.modalities) and not hasattr(self, "detector"):
            scores.update(box_rgb_consistency_map50=None, box_rgb_consistency_map50_95=None)
        return scores | {"inference_steps": 1}
