"""Unit and CPU integration tests using the locally prepared tiny pMF conversion."""

import itertools
import os
from pathlib import Path
import random
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, Subset

from rae_experiment.any_to_any.data import CorrosionDataset, read_boxes
from rae_experiment.any_to_any.flow import Task, edit_region, generate, loss
from rae_experiment.any_to_any.metrics import frechet, kid
from rae_experiment.any_to_any.model import AnyToAnyPMF, MODALITIES, load_backbone
from rae_experiment.any_to_any.preflight import audit_rgb
from rae_experiment.any_to_any.train import epoch_loss, smoke


class UnitTests(unittest.TestCase):
    def test_box_slots_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "labels.txt"
            label.write_text("1 0.75 0.5 0.25 0.25\n0 0.25 0.5 0.25 0.25\n")
            boxes, labels = read_boxes(label, 3)
            self.assertEqual(labels.tolist(), [0, 1, 3])
            self.assertTrue(torch.allclose(boxes[0], torch.tensor([0.125, 0.375, 0.375, 0.625])))
            with self.assertRaisesRegex(ValueError, "Increase"):
                read_boxes(label, 1)

    def test_metrics(self):
        first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertAlmostEqual(frechet(first, first), 0.0, places=9)
        self.assertAlmostEqual(frechet(first, first + 1), 2.0, places=9)
        self.assertAlmostEqual(kid(first, first + 1), kid(first + 1, first))

    def test_spatial_edit(self):
        rgb = torch.zeros(1, 3, 8, 8)
        boxes = torch.tensor([[[0, 0, 0.25, 0.25]]])
        mask = edit_region(rgb, boxes, torch.ones(1, 1, dtype=torch.bool))
        self.assertEqual(mask.sum().item(), 4)


@unittest.skipUnless(os.getenv("PMF_TINY_BACKBONE"), "set PMF_TINY_BACKBONE for pMF integration")
class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        cls.backbone = Path(os.environ["PMF_TINY_BACKBONE"])
        cls.model = AnyToAnyPMF(load_backbone(cls.backbone), MODALITIES, slots=3,
                               dino_dim=6, depth_size=8, depth_patch=4, rank=2)
        cls.model.pmf.gradient_checkpointing = True

    def sample(self):
        root = os.getenv("RF100_DATASET")
        if root:
            image = sorted((Path(root) / "valid" / "images").glob("*.jpg"))[0]
            records = [(image.relative_to(root).as_posix(), f"valid/labels/{image.stem}.txt", "", "")]
            dataset = CorrosionDataset(root, records, size=self.model.pmf.sample_size, slots=32)
            item = dataset[0]
            item = {key: val[:3] if key in ("boxes", "box_classes", "presence") else val
                    for key, val in item.items()}
        else:
            item = dict(rgb=torch.rand(3, 32, 32).mul(2).sub(1),
                        boxes=torch.tensor([[0.1, 0.1, 0.7, 0.7], [0.] * 4, [0.] * 4]),
                        box_classes=torch.tensor([1, 3, 3]), presence=torch.tensor([True, False, False]))
        return item | dict(dino=torch.randn(3, 6), depth=torch.rand(1, 8, 8))

    def test_rgb_matches_unmodified_backbone_at_initialization(self):
        model = AnyToAnyPMF(load_backbone(self.backbone), ("rgb",), slots=3, rank=2).eval()
        rgb = self.sample()["rgb"][None]
        t = torch.tensor([0.6])
        labels = torch.tensor([17])
        with torch.no_grad():
            baseline = load_backbone(self.backbone).eval()(rgb, t, labels).u
            prediction, _ = model({"rgb": rgb}, ("rgb",), t, t * 0, labels=labels)
        torch.testing.assert_close((rgb - prediction["rgb"]) / t, baseline)

    def test_all_65_condition_target_partitions(self):
        batch = {key: val[None] for key, val in self.sample().items()}
        shapes = {key: batch[key].shape[1:] for key in MODALITIES}
        rng = torch.Generator().manual_seed(17)
        self.model.eval()
        count = 0
        for roles in itertools.product(range(3), repeat=4):
            targets = tuple(name for name, role in zip(MODALITIES, roles) if role == 2)
            if not targets:
                continue
            conditions = {name: batch[name] for name, role in zip(MODALITIES, roles) if role == 1}
            result, _ = generate(self.model, conditions, {name: shapes[name] for name in targets},
                                 batch_size=1, generator=rng, metadata=batch)
            self.assertEqual(result.keys(), set(targets))
            self.assertTrue(all(result[name].shape[1:] == shapes[name] and result[name].isfinite().all()
                                for name in targets))
            with torch.no_grad():
                total, _ = loss(self.model, batch, Task(tuple(conditions), targets), generator=rng)
            self.assertTrue(total.isfinite())
            count += 1
        self.assertEqual(count, 65)

    def test_optimizer_smoke_with_checkpointing(self):
        loader = DataLoader([self.sample()])
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            smoke(self.model, loader, loader, optimizer, Path(directory), "tiny", False,
                  torch.device("cpu"), 17)
            self.assertTrue((Path(directory) / "smoke.json").exists())
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()
                            if not parameter.requires_grad))

    def test_epoch_aggregates_global_and_modality_losses(self):
        loader = DataLoader([self.sample()])
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-4)
        for options in ({"optimizer": optimizer}, {}):
            scores = epoch_loss(
                self.model, loader, torch.Generator().manual_seed(17), random.Random(1),
                torch.device("cpu"), False, **options,
            )
            self.assertGreater(scores["loss_global"], 0)
            self.assertAlmostEqual(scores["loss_global"], scores["loss_rgb"])
            self.assertEqual(scores["target_rgb_frequency"], 1)

    def test_rgb_boxes_audit_and_validation_protocol(self):
        model = AnyToAnyPMF(load_backbone(self.backbone), ("rgb", "boxes"), slots=3, rank=2)
        item = self.sample()
        audit = audit_rgb(model, {key: val[None] for key, val in item.items()})
        self.assertLess(audit["rgb_jvp_relative_error"], 0.08)
        scores = epoch_loss(
            model, DataLoader([item] * 4), torch.Generator().manual_seed(17), random.Random(1),
            torch.device("cpu"), False, rgb_boxes=True,
        )
        self.assertEqual(scores["task_noise_to_rgb_boxes_frequency"], 0.25)
        self.assertEqual(scores["task_rgb_to_boxes_frequency"], 0.25)
        self.assertEqual(scores["task_boxes_to_rgb_frequency"], 0.25)
        self.assertEqual(scores["task_noise_to_rgb_frequency"], 0.25)
        self.assertGreater(scores["loss_rgb_unweighted"], 0)
        self.assertLessEqual(scores["loss_rgb"], 1.0)
        self.assertGreater(scores["loss_boxes_unweighted"], 0)
