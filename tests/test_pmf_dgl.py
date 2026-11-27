import unittest

import torch
from torch import nn

from rae_experiment.any_to_any.model import GatedLoRALinear, _TARGET_SLICES


class GatedLoRATest(unittest.TestCase):
    def test_target_slices_and_jvp(self):
        layer = GatedLoRALinear(nn.Linear(1, 1, bias=False).requires_grad_(False), 1)
        x = torch.ones(2, 3, 1)
        routes = (("boxes", slice(1, 2)), ("rgb", slice(2, 3)))
        token = _TARGET_SLICES.set(routes)
        try:
            torch.testing.assert_close(layer(x), layer.base(x))
            with torch.no_grad():
                layer.base.weight.fill_(1)
                for name, residual in (("boxes", 2), ("rgb", 4)):
                    layer.adapters[name][0].weight.fill_(1)
                    layer.adapters[name][1].weight.fill_(residual)
            output, tangent = torch.func.jvp(layer, (x,), (x,))
            expected = torch.tensor([1.0, 3.0, 5.0]).reshape(1, 3, 1).expand_as(x)
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(tangent, expected)
        finally:
            _TARGET_SLICES.reset(token)

        token = _TARGET_SLICES.set((("boxes", slice(1, 2)),))
        try:
            layer(x).sum().backward()
            self.assertIsNone(layer.adapters["rgb"][1].weight.grad)
            self.assertIsNone(layer.base.weight.grad)
            self.assertEqual(layer.adapters["boxes"][1].weight.grad.item(), 2)
        finally:
            _TARGET_SLICES.reset(token)
