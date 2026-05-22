from pathlib import Path
import builtins
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.models.imagenet_wrappers import GoogLeNetCaffeWrapper


def test_googlenet_caffe_wrapper_reports_missing_caffe(monkeypatch) -> None:
    """Check that the Caffe dependency failure is explicit."""
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "caffe":
            raise ImportError("no caffe")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="Caffe is required"):
        GoogLeNetCaffeWrapper(
            model_dir="models/googlenet",
            deploy_prototxt="models/googlenet/deploy.prototxt",
            caffemodel="models/googlenet/bvlc_googlenet.caffemodel",
        )
