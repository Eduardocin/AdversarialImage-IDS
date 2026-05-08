import importlib.util
import unittest

from src.models import target_models


class TargetModelLoaderTests(unittest.TestCase):
    def test_imagenet_backbone_names_are_explicit(self):
        names = set(target_models.ImageNetBackboneName.__args__)

        self.assertEqual(names, {"alexnet", "googlenet", "inception_v3"})


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class TorchScriptTargetLoaderTests(unittest.TestCase):
    def test_load_torchscript_target_returns_eval_model(self):
        import tempfile
        from pathlib import Path

        import torch
        from torch import nn

        class TinyTarget(nn.Module):
            def forward(self, inputs):
                return torch.cat([1.0 - inputs, inputs], dim=1)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            model_path = Path(temp_dir) / "target.pt"
            scripted = torch.jit.script(TinyTarget())
            scripted.save(str(model_path))

            loaded = target_models.load_torchscript_target(model_path)

        self.assertFalse(loaded.training)


if __name__ == "__main__":
    unittest.main()
