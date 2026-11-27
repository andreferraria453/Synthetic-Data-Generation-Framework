"""Add modality tokens around the original pMF transformer blocks."""

from contextvars import ContextVar
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rae_experiment.sampling.generate_pmf import CHECKPOINT, REVISION, VARIANT

MODALITIES = ("rgb", "boxes", "dino", "depth")
_TARGET_SLICES = ContextVar("pmf_target_slices", default=())


def load_backbone(local_dir=None, offline=False, *, strict=False, variant=VARIANT):
    from diffusers.utils.dynamic_modules_utils import get_class_from_dynamic_module
    from huggingface_hub import snapshot_download

    if local_dir is None:
        root = snapshot_download(
            CHECKPOINT, revision=REVISION, local_files_only=offline,
            allow_patterns=[f"{variant}/transformer/*"],
        )
        local_dir = Path(root) / variant / "transformer"
    local_dir = str(Path(local_dir).resolve())
    model_class = get_class_from_dynamic_module(
        local_dir, "transformer_pmf.py", class_name="PMFTransformer2DModel",
        local_files_only=True,
    )
    if not strict:
        return model_class.from_pretrained(local_dir, local_files_only=True)
    backbone, info = model_class.from_pretrained(local_dir, local_files_only=True, output_loading_info=True)
    problems = {key: entries for key, entries in info.items() if entries}
    if problems:
        raise RuntimeError(f"Could not load the full pMF checkpoint: {problems}")
    return backbone


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        return self.base(value) + self.up(self.down(value))


class GatedLoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(base.in_features, rank, bias=False),
                                nn.Linear(rank, base.out_features, bias=False))
            for name in ("rgb", "boxes")
        })
        for adapter in self.adapters.values():
            nn.init.zeros_(adapter[1].weight)

    def forward(self, value):
        output = self.base(value)
        routes = _TARGET_SLICES.get()
        if not routes:
            return output
        pieces = []
        start = 0
        for name, span in routes:
            pieces.extend((output[:, start:span.start],
                           output[:, span] + self.adapters[name](value[:, span])))
            start = span.stop
        pieces.append(output[:, start:])
        return torch.cat(pieces, dim=1)


class AnyToAnyPMF(nn.Module):
    def __init__(self, pmf, modalities, slots=32, dino_dim=384, depth_size=64,
                 depth_patch=8, classes=3, rank=8, adapter="legacy"):
        super().__init__()
        self.modalities = tuple(modalities)
        if not self.modalities or len(set(self.modalities)) != len(self.modalities):
            raise ValueError("Specify distinct, nonempty modalities.")
        if set(self.modalities) - set(MODALITIES) or slots < 1 or depth_size % depth_patch:
            raise ValueError("Invalid modalities, slot count or depth grid.")
        if adapter not in ("legacy", "shared", "dgl"):
            raise ValueError(f"Unknown adapter: {adapter}")
        if adapter == "dgl" and set(self.modalities) - {"rgb", "boxes"}:
            raise ValueError("DGL is configured for RGB and boxes.")
        self.adapter = adapter
        self.pmf = pmf.requires_grad_(False)
        projections = {"q_proj", "k_proj", "v_proj", "out_proj"}
        wrapper = GatedLoRALinear if adapter == "dgl" else LoRALinear
        if adapter != "legacy":
            projections.update(("w1", "w2", "w3"))  # pMF SwiGLU
        # Keep initialization of the modality heads identical across adapter layouts.
        with torch.random.fork_rng(devices=[], enabled=adapter != "legacy"):
            for name, layer in list(pmf.named_modules()):
                if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in projections:
                    parent_name, attribute = name.rsplit(".", 1)
                    setattr(pmf.get_submodule(parent_name), attribute, wrapper(layer, rank))
        self.slots = slots
        self.classes = classes
        self.dino_dim = dino_dim
        self.depth_size = depth_size
        self.depth_patch = depth_patch
        self.null_class = pmf.config.num_classes
        width = pmf.hidden_size
        pixels = pmf.x_embedder.num_patches
        depth_tokens = (depth_size // depth_patch) ** 2
        counts = dict(rgb=pixels, boxes=slots, dino=slots, depth=depth_tokens)
        fields = dict(boxes=4, dino=dino_dim, depth=depth_patch ** 2)
        self.inputs = nn.ModuleDict()
        self.outputs = nn.ModuleDict()
        self.position = nn.ParameterDict()
        self.role = nn.ModuleDict()
        self.kind = nn.ParameterDict()
        for name in self.modalities:
            self.position[name] = nn.Parameter(torch.zeros(1, counts[name], width))
            self.kind[name] = nn.Parameter(torch.zeros(1, 1, width))
            self.role[name] = nn.Embedding(2, width)
            nn.init.zeros_(self.role[name].weight)
            if name != "rgb":
                self.inputs[name] = nn.Linear(fields[name], width)
                self.outputs[name] = nn.Linear(width, fields[name])
        if "rgb" in self.modalities:
            self.edit_embed = nn.Linear(pmf.patch_size ** 2, width, bias=False)
            nn.init.zeros_(self.edit_embed.weight)
            self.rgb_output = nn.Linear(width, pmf.patch_size ** 2 * pmf.out_channels, bias=False)
            nn.init.zeros_(self.rgb_output.weight)
        if "boxes" in self.modalities:
            self.box_class_input = nn.Embedding(classes + 1, width)
            self.box_class_output = nn.Linear(width, classes + 1)
        if "dino" in self.modalities:
            self.dino_presence_input = nn.Embedding(2, width)
            self.dino_presence_output = nn.Linear(width, 2)
        if "boxes" in self.modalities and "dino" in self.modalities:
            self.object_slots = nn.Parameter(torch.zeros(1, slots, width))

    def forward(self, values, targets, t, r, *, edit=None, labels=None, omega=1.0,
                guidance_min=0.0, guidance_max=1.0, metadata=None, auxiliary=False):
        """Predict x0 in each active modality, sharing all attention blocks."""
        pmf = self.pmf
        batch_size = t.shape[0]
        labels = labels if labels is not None else torch.full(
            (batch_size,), self.null_class, device=t.device, dtype=torch.long,
        )
        filled = torch.ones_like(t)
        rgb = values.get("rgb")
        sample = rgb if rgb is not None else torch.zeros(
            batch_size, 3, pmf.sample_size, pmf.sample_size, device=t.device, dtype=t.dtype,
        )
        sequence = pmf._build_sequence(
            sample, t - r, filled * omega, filled * guidance_min, filled * guidance_max, labels,
        )
        pieces = [sequence[:, :pmf.prefix_tokens]]
        sections = {}
        start = pmf.prefix_tokens
        # RGB stays last so the upstream RoPE still rotates precisely the RGB lattice.
        for name in ("boxes", "dino", "depth", "rgb"):
            if name not in values:
                continue
            value = values[name]
            if name == "rgb":
                tokens = sequence[:, pmf.prefix_tokens:]
                if edit is not None:
                    mask_patches = F.unfold(edit.float(), pmf.patch_size, stride=pmf.patch_size)
                    tokens = tokens + self.edit_embed(mask_patches.transpose(1, 2))
            else:
                if name == "depth":
                    value = F.unfold(value, self.depth_patch, stride=self.depth_patch).transpose(1, 2)
                tokens = self.inputs[name](value)
                if name in ("boxes", "dino") and hasattr(self, "object_slots"):
                    tokens = tokens + self.object_slots
                if name not in targets and metadata is not None:
                    if name == "boxes":
                        tokens = tokens + self.box_class_input(metadata["box_classes"])
                    elif name == "dino":
                        tokens = tokens + self.dino_presence_input(metadata["presence"].long())
            tokens = tokens + self.kind[name] + self.position[name] + self.role[name].weight[int(name in targets)]
            pieces.append(tokens)
            sections[name] = slice(start, start + tokens.shape[1])
            start += tokens.shape[1]

        hidden = torch.cat(pieces, dim=1)
        routes = tuple((name, span) for name, span in sections.items() if name in targets) if self.adapter == "dgl" else ()
        # An empty lattice needs a separate identity rotation. The upstream
        # implementation interprets -0 as the start of the entire sequence.
        rope = pmf.rope_freqs if rgb is not None else torch.ones(
            start, pmf.rope_freqs.shape[-1], dtype=pmf.rope_freqs.dtype, device=t.device,
        )
        for block in pmf.shared_blocks:
            hidden = self.run_block(block, hidden, rope, routes)
        shared = hidden
        for block in pmf.u_heads:
            hidden = self.run_block(block, hidden, rope, routes)
        predictions, logits = {}, {}
        for name in targets:
            tokens = hidden[:, sections[name]]
            if name == "rgb":
                predictions[name] = pmf._unpatchify(pmf.u_final_layer(tokens) + self.rgb_output(tokens))
            else:
                prediction = self.outputs[name](tokens)
                if name == "depth":
                    prediction = F.fold(
                        prediction.transpose(1, 2), (self.depth_size, self.depth_size),
                        self.depth_patch, stride=self.depth_patch,
                    )
                predictions[name] = prediction
                if name == "boxes":
                    logits[name] = self.box_class_output(tokens)
                elif name == "dino":
                    logits[name] = self.dino_presence_output(tokens)
        if "rgb" in targets and pmf.v_final_layer is not None and auxiliary:
            for block in pmf.v_heads:
                shared = self.run_block(block, shared, rope, routes)
            logits["rgb_v"] = pmf._unpatchify(pmf.v_final_layer(shared[:, sections["rgb"]]))
        return predictions, logits

    def run_block(self, block, hidden, rope, routes=()):
        def routed(value, frequencies):
            token = _TARGET_SLICES.set(routes)
            try:
                return block(value, frequencies)
            finally:
                _TARGET_SLICES.reset(token)

        operation = routed if routes else block
        if self.training and self.pmf.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(operation, hidden, rope, use_reentrant=False)
        return operation(hidden, rope)
