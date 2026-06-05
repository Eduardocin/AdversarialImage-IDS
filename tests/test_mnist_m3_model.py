from pathlib import Path
import sys
import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.models.mnist_m3 import build_mnist_m3_model  # noqa: E402


class FakeSequential:
    def __init__(self) -> None:
        self.layers = []

    def add(self, layer) -> None:
        self.layers.append(layer)

    def __call__(self, x_placeholder):
        return {"x": x_placeholder, "num_layers": len(self.layers)}


def _fake_layer(name):
    def factory(*args, **kwargs):
        return {"name": name, "args": args, "kwargs": kwargs}

    return factory


def test_mnist_m3_model_declares_fashion_mnist_cnn_contract(monkeypatch) -> None:
    """M3 should expose the configured Fashion-MNIST CNN shape and class contract."""
    keras_module = types.ModuleType("keras")
    layers_module = types.ModuleType("keras.layers")
    models_module = types.ModuleType("keras.models")
    layers_module.Activation = _fake_layer("Activation")
    layers_module.Dense = _fake_layer("Dense")
    layers_module.Dropout = _fake_layer("Dropout")
    layers_module.Flatten = _fake_layer("Flatten")
    layers_module.Convolution2D = _fake_layer("Convolution2D")
    layers_module.MaxPooling2D = _fake_layer("MaxPooling2D")
    models_module.Sequential = FakeSequential

    monkeypatch.setitem(sys.modules, "keras", keras_module)
    monkeypatch.setitem(sys.modules, "keras.layers", layers_module)
    monkeypatch.setitem(sys.modules, "keras.models", models_module)

    model, predictions = build_mnist_m3_model("x")

    layer_names = [layer["name"] for layer in model.layers]
    assert layer_names == [
        "Convolution2D",
        "Activation",
        "Convolution2D",
        "Activation",
        "MaxPooling2D",
        "Dropout",
        "Convolution2D",
        "Activation",
        "Convolution2D",
        "Activation",
        "MaxPooling2D",
        "Dropout",
        "Flatten",
        "Dense",
        "Activation",
        "Dropout",
        "Dense",
        "Activation",
    ]
    assert model.layers[0]["kwargs"]["input_shape"] == (28, 28, 1)
    assert model.layers[-2]["args"] == (10,)
    assert model.layers[-1]["args"] == ("softmax",)
    assert model.layers[5]["args"] == (0.25,)
    assert model.layers[11]["args"] == (0.25,)
    assert model.layers[15]["args"] == (0.5,)
    assert predictions == {"x": "x", "num_layers": len(model.layers)}
