from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.attacks.fgsm_imagenet import generate_fgsm_imagenet


def test_generate_fgsm_imagenet_preserves_shape_and_range() -> None:
    """Check ImageNet FGSM output shape, dtype and clipping."""
    tf = pytest.importorskip("tensorflow")
    tf.compat.v1.reset_default_graph()

    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 4, 4, 1), name="x")
    summed = tf.reduce_sum(x, axis=[1, 2, 3])
    logits = tf.stack([summed, -summed], axis=1)

    images = np.full((3, 4, 4, 1), 0.5, dtype=np.float32)
    with tf.compat.v1.Session() as sess:
        adv = generate_fgsm_imagenet(
            sess=sess,
            x_placeholder=x,
            logits=logits,
            images=images,
            eps=4.0 / 255.0,
            batch_size=2,
        )

    assert adv.shape == images.shape
    assert adv.dtype == np.float32
    assert adv.min() >= 0.0
    assert adv.max() <= 1.0
