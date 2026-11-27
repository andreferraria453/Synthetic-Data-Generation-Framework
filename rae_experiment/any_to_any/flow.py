"""Shared-interval MeanFlow, task selection and joint one-step inference."""

import random
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Task:
    conditions: tuple[str, ...]
    targets: tuple[str, ...]
    edit: bool = False


def choose_task(modalities, rng: random.Random):
    active = set(modalities)
    if "rgb" in active:
        draw = rng.random()
        if draw < 0.15:
            return Task((), ("rgb",))
        if draw < 0.30:
            context = tuple(name for name in modalities if name != "rgb" and rng.random() < 0.5)
            return Task(context, ("rgb",), edit=True)
    roles = [rng.randrange(3) for _ in modalities]
    roles[rng.randrange(len(roles))] = 2
    return Task(
        tuple(name for name, role in zip(modalities, roles) if role == 1),
        tuple(name for name, role in zip(modalities, roles) if role == 2),
    )


def expand_time(value, tensor):
    return value.reshape((len(value),) + (1,) * (tensor.ndim - 1))


def edit_region(rgb, boxes, presence):
    """Use annotated rectangles, with a central fallback for empty images."""
    height, width = rgb.shape[-2:]
    x = (torch.arange(width, device=rgb.device) + 0.5) / width
    y = (torch.arange(height, device=rgb.device) + 0.5) / height
    left, top, right, bottom = boxes.unbind(-1)
    inside_x = (x >= left[..., None]) & (x <= right[..., None])
    inside_y = (y >= top[..., None]) & (y <= bottom[..., None])
    mask = (inside_y[..., :, None] & inside_x[..., None, :] & presence[..., None, None]).any(1)
    center = ((y >= 0.25) & (y < 0.75))[:, None] & ((x >= 0.25) & (x < 0.75))[None, :]
    return torch.where(mask.flatten(1).any(1)[:, None, None], mask, center)[:, None]


def loss(model, batch, task, *, generator=None, epsilon=1e-6):
    keys = task.targets
    rgb = batch["rgb"]
    random = torch.rand((3, len(rgb)), device=rgb.device, generator=generator)
    t = 0.05 + 0.94 * random[0]
    r = torch.where(random[1] < 0.25, t, t * random[2])
    edit = edit_region(rgb, batch["boxes"], batch["presence"]) if task.edit else None
    clean = {name: batch[name].float() for name in keys}
    noisy, velocity = {}, {}
    for name, value in clean.items():
        noise = torch.randn(value.shape, device=value.device, generator=generator)
        noise = noise * (4.0 if name == "rgb" else 1.0)
        if name == "rgb" and edit is not None:
            noise = torch.where(edit, noise, value)
        noisy[name] = torch.lerp(value, noise, expand_time(t, value))
        velocity[name] = noise - value
    fixed = {name: batch[name] for name in task.conditions}

    def velocity_prediction(time, *states, auxiliary=False):
        inputs = dict(zip(keys, states))
        predictions, extras = model(fixed | inputs, keys, time, r, edit=edit, metadata=batch,
                                    auxiliary=auxiliary)
        flows = tuple((inputs[name] - predictions[name].float()) /
                      expand_time(time.clamp_min(model.pmf.t_clip_min), inputs[name]) for name in keys)
        return flows, extras

    # Fixed conditions and r have zero tangent. Only noisy target states and t move.
    # If the conversion ships v weights, retain the pretrained RGB v head and its loss.
    directions = dict(velocity)
    if "rgb" in keys and model.pmf.v_final_layer is not None:
        with torch.no_grad():
            _, auxiliary = model(fixed | noisy, keys, t, t, edit=edit, metadata=batch, auxiliary=True)
            directions["rgb"] = (noisy["rgb"] - auxiliary["rgb_v"].float()) / expand_time(t, rgb)
            if edit is not None:
                directions["rgb"] = torch.where(edit, directions["rgb"], 0.0)
    with torch.no_grad():
        _, derivative = torch.func.jvp(
            lambda *args: velocity_prediction(*args)[0],
            (t, *(noisy[name] for name in keys)),
            (torch.ones_like(t), *(directions[name] for name in keys)),
        )
    predictions, extras = velocity_prediction(t, *(noisy[name] for name in keys), auxiliary=True)
    components = {}
    for name, average, tangent in zip(keys, predictions, derivative):
        compound = average + expand_time(t - r, average) * tangent.detach()
        error = (compound - velocity[name]).square()
        weight = torch.ones_like(error)
        if name in ("boxes", "dino"):
            weight = weight * batch["presence"][..., None]
        if name == "rgb" and edit is not None:
            weight = weight * edit
        squared = (error * weight).flatten(1).sum(1)
        if name == "rgb":
            component = (squared / (squared.detach() + epsilon).pow(0.5)).mean()
            if "rgb_v" in extras:
                v = (noisy[name] - extras["rgb_v"].float()) / expand_time(t, rgb)
                residual = ((v - velocity[name]).square() * weight).flatten(1).sum(1)
                component = component + (residual / (residual.detach() + epsilon).pow(0.5)).mean()
        else:
            component = (squared / weight.flatten(1).sum(1).clamp_min(1)).mean()
        if name == "boxes":
            component = component + F.cross_entropy(extras[name].transpose(1, 2), batch["box_classes"])
        elif name == "dino":
            component = component + F.cross_entropy(extras[name].transpose(1, 2), batch["presence"].long())
        components[name] = component
    return torch.stack(tuple(components.values())).mean(), components


@torch.no_grad()
def generate(model, conditions, target_shapes, *, batch_size, generator, metadata=None,
             edit=None, original_rgb=None, labels=None):
    """Generate all requested outputs in one shared Transformer evaluation."""
    targets = tuple(target_shapes)
    if not targets or set(conditions) & set(targets):
        raise ValueError("Targets must be nonempty and disjoint from conditions.")
    if (edit is None) != (original_rgb is None) or (edit is not None and "rgb" not in targets):
        raise ValueError("Editing requires an original image, mask and RGB target.")
    device = next(model.parameters()).device
    inputs = dict(conditions)
    for name, shape in target_shapes.items():
        noise = torch.randn((batch_size, *shape), device=device, generator=generator)
        inputs[name] = noise * (4.0 if name == "rgb" else 1.0)
    if edit is not None:
        inputs["rgb"] = torch.where(edit, inputs["rgb"], original_rgb)
    t = torch.ones(batch_size, device=device)
    predictions, logits = model(inputs, targets, t, torch.zeros_like(t), edit=edit,
                                labels=labels, metadata=metadata)
    if edit is not None:
        logits["rgb_edit_raw"] = predictions["rgb"]
        predictions["rgb"] = torch.where(edit, predictions["rgb"], original_rgb)
    return predictions, logits
