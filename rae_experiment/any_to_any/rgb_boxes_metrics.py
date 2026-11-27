"""Fixed RF100 validation and qualitative samples for four tasks."""

from time import perf_counter

import torch
from PIL import Image, ImageDraw

from .flow import RGB_BOX_TASKS, generate
from .metrics import box_outputs, frechet, kid, pixels


def panel(image, real, predicted, title):
    result = Image.fromarray(pixels(image).permute(1, 2, 0).cpu().numpy())
    draw = ImageDraw.Draw(result)
    for boxes, color in ((real, "#45ed76"), (predicted, "#ff5aa5")):
        for box, label in zip(boxes["boxes"], boxes["labels"]):
            rect = (box * result.width).tolist()
            draw.rectangle(rect, outline=color, width=3)
            draw.text((rect[0], rect[1]), ("Slippage", "corrosion", "crack")[label.item()], fill=color)
    draw.rectangle((0, 0, result.width, 24), fill="black")
    draw.text((5, 5), title, fill="white")
    return result


class RGBBoxesValidator:
    def __init__(self, device, seed):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        from torchmetrics.detection import MeanAveragePrecision

        self.device = device
        self.seed = seed
        self.inception = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).eval().to(device)
        self.ap = MeanAveragePrecision(box_format="xyxy")
        self.reference = []

    @torch.inference_mode()
    def evaluate(self, model, loader, example_dir):
        model.eval()
        fake = {name: [] for name, task in RGB_BOX_TASKS.items() if "rgb" in task.targets}
        times = {name: [] for name in RGB_BOX_TASKS}
        empty = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
        example_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(loader):
            batch = {name: value.to(self.device) for name, value in item.items()}
            rgb = batch["rgb"]
            keep = batch["presence"][0]
            annotations = {"boxes": batch["boxes"][0, keep].cpu(),
                           "labels": batch["box_classes"][0, keep].cpu()}
            if len(self.reference) <= index:
                self.reference.append(self.inception(pixels(rgb))[0].cpu())
            for job, (name, task) in enumerate(RGB_BOX_TASKS.items()):
                rng = torch.Generator(device=self.device).manual_seed(self.seed + 4 * index + job)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                start = perf_counter()
                outputs, logits = generate(
                    model, {key: batch[key] for key in task.conditions},
                    {key: batch[key].shape[1:] for key in task.targets},
                    batch_size=1, generator=rng, metadata=batch,
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                if index:
                    times[name].append(perf_counter() - start)
                if not all(torch.isfinite(value).all() for value in (*outputs.values(), *logits.values())):
                    raise FloatingPointError(f"Nonfinite validation sample: {name}/{index}")
                if "rgb" in outputs:
                    fake[name].append(self.inception(pixels(outputs["rgb"]))[0].cpu())
                predicted = empty
                if "boxes" in outputs:
                    predicted = box_outputs(outputs["boxes"], logits["boxes"], model.classes)[0]
                    if name == "rgb_to_boxes":
                        self.ap.update([predicted], [annotations])
                if index < 2:
                    reference = panel(rgb[0], annotations, empty, "Validação: caixas reais em verde")
                    comparison = panel(
                        outputs.get("rgb", rgb)[0],
                        annotations if name in ("rgb_to_boxes", "boxes_to_rgb") else empty,
                        predicted, name,
                    )
                    image = Image.new("RGB", (rgb.shape[-1] * 2, rgb.shape[-2]))
                    image.paste(reference, (0, 0))
                    image.paste(comparison, (rgb.shape[-1], 0))
                    image.save(example_dir / f"{index:02}_{name}.png")
            if (index + 1) % 25 == 0 or index + 1 == len(loader):
                print(f"Métricas e amostras: {index + 1}/{len(loader)}", flush=True)
        true = torch.cat(self.reference)
        scores = {}
        for name, features in fake.items():
            prediction = torch.cat(features)
            scores[f"{name}_fid"] = frechet(true, prediction)
            scores[f"{name}_kid"] = kid(true, prediction)
        scores["fid"] = scores["noise_to_rgb_fid"]
        scores["kid"] = scores["noise_to_rgb_kid"]
        ap = self.ap.compute()
        scores["map50"] = ap["map_50"].item()
        scores["map50_95"] = ap["map"].item()
        self.ap.reset()
        for name, latency in times.items():
            scores[f"{name}_images_per_second"] = len(latency) / sum(latency)
        return scores | {"inference_steps": 1}
