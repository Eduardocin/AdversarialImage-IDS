import importlib.util
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class FGSMAttackTests(unittest.TestCase):
    def test_fgsm_attack_returns_clipped_batch_with_same_shape(self):
        import torch
        from torch import nn

        from src.attacks.fgsm import fgsm_attack

        model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        images = torch.full((2, 1, 2, 2), 0.5)
        labels = torch.tensor([0, 1])

        adversarial = fgsm_attack(model, images, labels, epsilon=0.1)

        self.assertEqual(adversarial.shape, images.shape)
        self.assertTrue(torch.all(adversarial >= 0.0))
        self.assertTrue(torch.all(adversarial <= 1.0))
        self.assertFalse(adversarial.requires_grad)


if __name__ == "__main__":
    unittest.main()
