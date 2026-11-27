"""Numerical preflight for the RGB/boxes protocol."""

import torch

from .flow import RGB_BOX_TASKS, generate, loss


@torch.no_grad()
def audit_rgb(model, batch):
    """Check the untouched RGB one-step path and the velocity derivative."""
    model.eval()
    rgb = batch["rgb"]
    rng = torch.Generator(device=rgb.device).manual_seed(413)
    state = rng.get_state()
    noise = 4 * torch.randn(rgb.shape, device=rgb.device, generator=rng)
    labels = torch.full((len(rgb),), 207, dtype=torch.long, device=rgb.device)
    t = torch.ones(len(rgb), device=rgb.device)
    original = model.pmf(noise, t, labels, h=t, omega=t * 7.5,
                         guidance_interval_min=t * 0.2, guidance_interval_max=t * 0.6).u
    rng.set_state(state)
    outputs, _ = generate(model, {}, {"rgb": rgb.shape[1:]}, batch_size=len(rgb),
                          labels=labels, generator=rng, omega=7.5, guidance_min=0.2, guidance_max=0.6)
    expected = noise - original
    torch.testing.assert_close(outputs["rgb"], expected, rtol=1e-4, atol=2e-5)
    noisy, direction = torch.lerp(rgb, noise, 0.6), noise - rgb
    r = t * 0.2
    t = t * 0.6

    def velocity(value, instant):
        x, _ = model({"rgb": value}, ("rgb",), instant, r, labels=labels)
        return (value - x["rgb"]) / instant[:, None, None, None]

    reference, tangent = torch.func.jvp(velocity, (noisy, t), (direction, torch.ones_like(t)))
    step = 0.001
    approximate = (velocity(noisy + step * direction, t + step)
                   - velocity(noisy - step * direction, t - step)) / (2 * step)
    scale = torch.maximum(tangent.norm(), reference.norm() * 0.01).clamp_min(1e-6)
    relative = (tangent - approximate).norm() / scale
    if not torch.isfinite(relative) or relative.item() > 0.08:
        raise RuntimeError(f"RGB JVP disagrees with finite differences: relative error={relative.item():.5g}")
    return {"rgb_sampler_max_abs": (outputs["rgb"] - expected).abs().max().item(),
            "rgb_jvp_relative_error": relative.item()}


def overfit(model, dataset, optimizer, device, *, images, updates, seed):
    """Fit fixed training images and fixed noise draws before the real run."""
    fixed = [{key: value[None].to(device) for key, value in dataset[index].items()}
             for index in range(min(images, len(dataset)))]
    tasks = tuple(RGB_BOX_TASKS.items())
    cases = [(name, task, batch) for batch in fixed for name, task in tasks]
    if not cases or updates < len(cases):
        raise ValueError("Overfit requires at least one complete update cycle.")
    params = [param for param in model.parameters() if param.requires_grad]

    def evaluate(index):
        name, task, batch = cases[index]
        rng = torch.Generator(device=device).manual_seed(seed + index)
        score, parts = loss(model, batch, task, generator=rng, balanced=True)
        if not torch.isfinite(score) or not all(torch.isfinite(value) for value in parts.values()):
            raise FloatingPointError(f"Nonfinite overfit loss: {name}")
        raw = [parts[key] for key in ("rgb_unweighted", "boxes_unweighted") if key in parts]
        monitor = torch.stack(raw).mean()
        return name, score, monitor

    def assess():
        model.eval()
        with torch.no_grad():
            values = [evaluate(index) for index in range(len(cases))]
        return {name: sum(raw.item() for label, _, raw in values if label == name) / len(fixed)
                for name, _ in tasks}

    before = assess()
    model.train()
    peak_gradient = 0.0
    for index in range(updates):
        optimizer.zero_grad(set_to_none=True)
        _, score, _ = evaluate(index % len(cases))
        score.backward()
        gradient = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True).item()
        peak_gradient = max(peak_gradient, gradient)
        optimizer.step()
        if (index + 1) % 8 == 0 or index + 1 == updates:
            print(f"Overfit: {index + 1}/{updates} updates | loss {score.item():.5g} "
                  f"| grad {gradient:.5g}", flush=True)
    after = assess()
    return {"passed": peak_gradient > 0 and all(after[name] < before[name] for name, _ in tasks),
            "before": before, "after": after, "peak_gradient_norm": peak_gradient,
            "images": len(fixed), "updates": updates}
