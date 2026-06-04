from pathlib import Path
import sys
import types

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from deepdetector.evaluation.article_reproduction import (  # noqa: E402
    evaluate_filter_predictions,
    interval_size,
)
from deepdetector.models import mnist_m2  # noqa: E402
from scripts.article_reproduction import mnist_table_10_m1_fgsm as table_10  # noqa: E402
from scripts.article_reproduction import mnist_table_10_m2_cw as table_10_m2  # noqa: E402


def test_interval_size_uses_article_mapping() -> None:
    """Check interval-count to interval-size mapping."""
    assert interval_size(2) == 128
    assert interval_size(6) == 43
    assert interval_size(10) == 26


def test_interval_size_rejects_unsupported_count() -> None:
    """Unsupported interval counts should fail explicitly."""
    with pytest.raises(ValueError):
        interval_size(11)


def test_evaluate_filter_predictions_matches_article_count_semantics() -> None:
    """Check #F, TP, FN, FP and RTP on a small deterministic example."""
    y_true = np.array([1, 2, 3, 4])
    clean_pred = np.array([1, 2, 3, 4])
    adv_pred = np.array([1, 0, 0, 0])
    filtered_clean_pred = np.array([1, 9, 3, 4])
    filtered_adv_pred = np.array([1, 2, 0, 4])

    metrics = evaluate_filter_predictions(
        y_true=y_true,
        clean_pred=clean_pred,
        adv_pred=adv_pred,
        filtered_clean_pred=filtered_clean_pred,
        filtered_adv_pred=filtered_adv_pred,
    )

    assert metrics["F"] == 1
    assert metrics["TP"] == 2
    assert metrics["FN"] == 1
    assert metrics["FP"] == 1
    assert metrics["RTP"] == 2
    assert metrics["recall_percent"] == pytest.approx(66.6666667)
    assert metrics["precision_percent"] == pytest.approx(66.6666667)


def test_evaluate_filter_predictions_can_skip_invalid_table_3_pairs() -> None:
    """Table 3 skips clean errors and failed attacks before metric counting."""
    y_true = np.array([1, 2, 3, 4])
    clean_pred = np.array([1, 9, 3, 4])
    adv_pred = np.array([0, 0, 3, 0])
    filtered_clean_pred = np.array([1, 8, 0, 9])
    filtered_adv_pred = np.array([1, 2, 0, 0])

    metrics = evaluate_filter_predictions(
        y_true=y_true,
        clean_pred=clean_pred,
        adv_pred=adv_pred,
        filtered_clean_pred=filtered_clean_pred,
        filtered_adv_pred=filtered_adv_pred,
        exclude_invalid_pairs=True,
    )

    assert metrics["clean_errors"] == 1
    assert metrics["F"] == 1
    assert metrics["TP"] == 1
    assert metrics["FN"] == 1
    assert metrics["FP"] == 1
    assert metrics["recall_percent"] == pytest.approx(50.0)
    assert metrics["precision_percent"] == pytest.approx(50.0)


def test_table_3_config_documents_experiment_parameters() -> None:
    """Table 3 should record the parameters needed to reproduce the run."""
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    table3 = config["experiments"]["table_3"]

    assert table3["kind"] == "filter_grid"
    assert table3["dataset"] == {
        "name": "mnist",
        "split": "test",
        "start": 0,
        "end": 100,
    }
    assert table3["attack"]["name"] == "fgsm"
    assert table3["attack"]["epsilon"] == 0.2
    assert table3["model"]["checkpoint_dir"] == (
        "artifacts/models/mnist/m1/clean_baseline/checkpoints"
    )
    assert table3["evaluation"]["exclude_invalid_pairs"] is True
    assert table3["evaluation"]["include_filter_time"] is True
    assert table3["output"]["include_filter_name"] is False

    filters = table3["filters"]
    assert filters[0]["type"] == "scalar_quantization"
    assert filters[0]["intervals"] == 2
    assert filters[1]["type"] == "nonuniform_quantization"
    assert len(filters) == 2


def test_table_4_config_documents_experiment_parameters() -> None:
    """Table 4 should record the scalar interval sweep parameters."""
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    table4 = config["experiments"]["table_4"]
    table4_mnist = config["experiments"]["table_4_mnist"]

    assert table4["kind"] == "composite"
    assert table4["components"] == ["table_4_mnist", "table_4_imagenet"]
    assert table4["output_dir"] == "results/table_4"
    assert table4_mnist["kind"] == "filter_grid"
    assert table4_mnist["output_dir"] == "results/table_4/mnist"
    assert table4_mnist["dataset"] == {
        "name": "mnist",
        "split": "test",
        "start": 0,
        "end": 4500,
    }
    assert table4_mnist["attack"]["name"] == "fgsm"
    assert table4_mnist["attack"]["epsilon"] == 0.2
    assert table4_mnist["model"]["checkpoint_dir"] == (
        "artifacts/models/mnist/m1/clean_baseline/checkpoints"
    )
    assert [row["intervals"] for row in table4_mnist["filters"]] == [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
    assert {row["type"] for row in table4_mnist["filters"]} == {"scalar_quantization"}


def test_imagenet_table_4_config_documents_spec_parameters() -> None:
    """ImageNet Table 4 should record the GoogLeNet/FGSM scalar sweep."""
    experiments = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    config = experiments["experiments"]["table_4_imagenet"]

    assert config["kind"] == "imagenet_table_4"
    assert config["output_dir"] == "results/table_4/imagenet"
    assert config["dataset"]["name"] == "imagenet"
    assert config["dataset"]["split"] == "train"
    assert config["dataset"]["images_dir"] == "data/imagenet/train"
    assert config["dataset"]["class_indices"] == {
        "goldfish": 1,
        "pineapple": 953,
        "digital_clock": 530,
    }
    assert config["model"]["name"] == "googlenet_caffe"
    assert config["model"]["reference"] == "BVLC GoogLeNet"
    assert config["model"]["mean_file"] is None
    assert config["attack"]["name"] == "fgsm"
    assert config["attack"]["epsilon_255"] == 1.0
    assert config["quantization"]["intervals"] == [2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert config["quantization"]["interval_sizes"][6] == 43
    assert config["output"]["csv"] == "table_4_imagenet.csv"
    assert "diagnostics_csv" not in config["output"]
    assert config["output"]["status_json"] == "table_4_status.json"


def test_table_6_config_documents_experiment_parameters() -> None:
    """Table 6 should record adaptive quantization validation parameters."""
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )
    table6 = config["experiments"]["table_6"]

    assert table6["kind"] == "table_6"
    assert table6["datasets"] == ["mnist", "imagenet"]
    assert table6["split_order"] == ["train", "validation"]
    assert table6["entropy_thresholds"] == {"low": 4.0, "medium": 5.0}
    assert table6["quantization"]["interval_sizes"] == {
        "low_entropy": 128,
        "medium_entropy": 64,
        "high_entropy": 43,
    }
    assert table6["mnist"]["dataset"] == {
        "name": "mnist",
        "split": "test",
        "slices": [
            {"name": "Training", "start": 0, "end": 4500},
            {"name": "Validation", "start": 4500, "end": 5500},
        ],
    }
    assert table6["mnist"]["attack"]["name"] == "fgsm"
    assert table6["mnist"]["attack"]["epsilon"] == 0.2
    assert table6["mnist"]["model"]["checkpoint_dir"] == (
        "artifacts/models/mnist/m1/clean_baseline/checkpoints"
    )
    assert table6["mnist"]["filter"] == {
        "name": "adaptive_quantization",
        "type": "adaptive_quantization",
    }
    assert table6["imagenet"]["dataset"]["splits"]["train"] == [
        {"name": "goldfish", "label": 1, "path": "data/imagenet/train/goldfish"},
        {"name": "pineapple", "label": 953, "path": "data/imagenet/train/pineapple"},
        {
            "name": "digital_clock",
            "label": 530,
            "path": "data/imagenet/train/digital_clock",
        },
    ]
    assert table6["imagenet"]["dataset"]["splits"]["validation"] == [
        {"name": "jellyfish", "label": 107, "path": "data/imagenet/validation/jellyfish"},
    ]
    assert table6["imagenet"]["attack"]["epsilon_255"] == 1.0
    assert table6["output_dir"] == "results/table_6"


def test_table_10_m2_config_documents_experiment_parameters() -> None:
    """Table 10 M2 should record saved CW adversarial evaluation parameters."""
    config_path = PROJECT_ROOT / "configs" / "article_reproduction" / "mnist_table_10_m2.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["dataset"] == {
        "name": "mnist",
        "split": "test",
        "start": 5500,
        "samples": 1000,
        "image_shape": [28, 28, 1],
        "value_range": [0.0, 1.0],
    }
    assert config["model"]["name"] == "M2"
    assert config["model"]["checkpoint_dir"] == (
        "artifacts/models/mnist/m2/clean_baseline/checkpoints"
    )
    assert config["detection"]["filter"] == "final"
    assert config["evaluation"]["use_saved_adversarial_examples"] is True
    assert config["evaluation"]["train_model"] is False
    assert config["evaluation"]["generate_attacks"] is False

    attacks = config["attacks"]
    assert attacks[0]["name"] == "CW L2 / M2"
    assert attacks[0]["norm"] == "L2"
    assert attacks[0]["kappas"] == [0.0, 0.5, 1.0, 2.0, 4.0]
    assert attacks[0]["batch_size"] == 1
    assert attacks[0]["max_iterations"] == 2000
    assert attacks[0]["learning_rate"] == 0.1
    assert attacks[0]["binary_search_steps"] == 5
    assert attacks[0]["initial_const"] == 1.0
    assert attacks[0]["adversarial_template"] == (
        "artifacts/adversarial_examples/mnist/m2/cw_l2/"
        "kappa_{kappa}/adversarial_examples.npy"
    )
    assert attacks[1]["name"] == "CW Linf / M2"
    assert attacks[1]["norm"] == "Linf"
    assert attacks[1]["max_iterations"] == 1000
    assert attacks[1]["learning_rate"] == 0.005
    assert attacks[1]["initial_const"] == 0.00001
    assert attacks[1]["largest_const"] == 20.0
    assert attacks[1]["decrease_factor"] == 0.9
    assert attacks[1]["const_factor"] == 2.0
    assert attacks[1]["targeted"] is False
    assert attacks[1]["adversarial_path"] == (
        "artifacts/adversarial_examples/mnist/m2/cw_linf/adversarial_examples.npy"
    )

    assert config["metrics"]["columns"] == [
        "Attack/Model",
        "Dataset",
        "#F",
        "TP",
        "FN",
        "FP",
        "RTP",
        "RTP%",
        "Recall",
        "Precision",
        "F1",
    ]
    assert config["output"]["results_dir"] == "results/table_10/M2_cw"


def test_table_10_m2_consolidated_config_uses_nn_robust_attack_names() -> None:
    """The generic Table 10 M2 config should not point at local CW backends."""
    config_path = PROJECT_ROOT / "configs" / "experiments.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = config["experiments"]["table_10_m2"]["rows"]

    assert [row["attack"]["name"] for row in rows] == [
        "cw_l2_nn_robust",
        "cw_l2_nn_robust",
        "cw_l2_nn_robust",
        "cw_l2_nn_robust",
        "cw_l2_nn_robust",
        "cw_linf_nn_robust",
    ]


def test_legacy_table_10_mnist_scripts_default_to_model_output_dirs() -> None:
    """Legacy MNIST Table 10 scripts should not write into the article root."""
    assert Path(table_10.build_parser().parse_args([]).output_dir) == (
        PROJECT_ROOT / "results" / "table_10" / "m1"
    )
    assert table_10_m2.DEFAULT_OUTPUT_DIR == (
        PROJECT_ROOT / "results" / "table_10" / "M2_cw"
    )


def test_table_10_m2_generation_refuses_existing_adversarial_without_overwrite(
    tmp_path,
) -> None:
    """M2 CW generation should not silently reuse or overwrite old attack arrays."""
    adversarial_path = tmp_path / "adversarial_examples.npy"
    np.save(str(adversarial_path), np.zeros((1, 28, 28, 1), dtype=np.float32))

    with pytest.raises(IOError, match="overwrite"):
        table_10_m2.generate_adversarial_path(
            graph={"sess": object(), "model": object(), "x": object()},
            clean_images=np.zeros((1, 28, 28, 1), dtype=np.float32),
            labels=np.asarray([0]),
            attack_row={
                "norm": "L2",
                "kappa": 0.0,
                "adversarial_path": adversarial_path,
            },
            attack_config={},
            dataset_config={},
            evaluation_config={},
            overwrite=False,
        )


def test_table_10_m2_can_filter_rows_by_kappa() -> None:
    """Targeted M2 runs should evaluate only the requested CW-L2 kappa."""
    config = {
        "attacks": [
            {
                "name": "CW L2 / M2",
                "attack": "CW",
                "norm": "L2",
                "kappas": [0.0, 0.5],
                "adversarial_template": (
                    "artifacts/adversarial_examples/mnist/m2/cw_l2/"
                    "kappa_{kappa}/adversarial_examples.npy"
                ),
            },
            {
                "name": "CW Linf / M2",
                "attack": "CW",
                "norm": "Linf",
                "kappas": [None],
                "adversarial_path": (
                    "artifacts/adversarial_examples/mnist/m2/cw_linf/"
                    "adversarial_examples.npy"
                ),
            },
        ]
    }

    rows = list(
        table_10_m2.filter_attack_rows(
            table_10_m2.configured_attack_rows(config),
            only_kappa=0.0,
        )
    )

    assert len(rows) == 1
    assert rows[0]["norm"] == "L2"
    assert rows[0]["kappa"] == 0.0
    assert str(rows[0]["adversarial_path"]).endswith(
        "cw_l2/kappa_0p0/adversarial_examples.npy"
    )


def test_table_10_m2_can_filter_rows_by_norm() -> None:
    """Targeted M2 runs should evaluate only the requested CW norm."""
    config = {
        "attacks": [
            {
                "name": "CW L2 / M2",
                "attack": "CW",
                "norm": "L2",
                "kappas": [0.0, 0.5],
                "adversarial_template": (
                    "artifacts/adversarial_examples/mnist/m2/cw_l2/"
                    "kappa_{kappa}/adversarial_examples.npy"
                ),
            },
            {
                "name": "CW Linf / M2",
                "attack": "CW",
                "norm": "Linf",
                "kappas": [None],
                "adversarial_path": (
                    "artifacts/adversarial_examples/mnist/m2/cw_linf/"
                    "adversarial_examples.npy"
                ),
            },
        ]
    }

    rows = list(
        table_10_m2.filter_attack_rows(
            table_10_m2.configured_attack_rows(config),
            only_kappa=None,
            only_norm="Linf",
        )
    )

    assert len(rows) == 1
    assert rows[0]["norm"] == "Linf"
    assert str(rows[0]["adversarial_path"]).endswith(
        "cw_linf/adversarial_examples.npy"
    )


def test_table_10_m2_targeted_runs_use_norm_specific_output_dirs(tmp_path) -> None:
    """Targeted M2 CW reports should not overwrite reports from another norm."""
    assert table_10_m2.output_dir_for_targeted_run(
        tmp_path,
        only_kappa=None,
        only_norm=None,
    ) == tmp_path
    assert table_10_m2.output_dir_for_targeted_run(
        tmp_path,
        only_kappa=None,
        only_norm="Linf",
    ) == tmp_path.parent / "M2_cw_Linf"
    assert table_10_m2.output_dir_for_targeted_run(
        tmp_path,
        only_kappa=None,
        only_norm="L2",
    ) == tmp_path.parent / "M2_cw_l2"
    assert table_10_m2.output_dir_for_targeted_run(
        tmp_path,
        only_kappa=0.5,
        only_norm=None,
    ) == tmp_path.parent / "M2_cw_l2"


def test_table_10_m2_l2_generation_requires_nn_robust_attacks_root(
    tmp_path,
) -> None:
    """M2 CW-L2 generation should never fall back to the local CW-L2 backend."""
    adversarial_path = tmp_path / "cw_l2" / "kappa_0p5" / "adversarial_examples.npy"
    images = np.zeros((2, 28, 28, 1), dtype=np.float32)
    labels = np.asarray([1, 2])

    with pytest.raises(ValueError, match="nn-robust-attacks-root"):
        table_10_m2.generate_adversarial_path(
            graph={"sess": "session", "model": "model", "x": "x"},
            clean_images=images,
            labels=labels,
            attack_row={
                "attack": "CW",
                "norm": "L2",
                "kappa": 0.5,
                "adversarial_path": adversarial_path,
            },
            attack_config={"batch_size": 2, "max_iterations": 3},
            dataset_config={},
            evaluation_config={},
            overwrite=True,
        )


def test_table_10_m2_sets_keras_inference_phase(monkeypatch) -> None:
    """M2 generation should force dropout into inference mode."""
    calls = []
    keras_module = types.ModuleType("keras")
    backend_module = types.ModuleType("keras.backend")
    backend_module.set_learning_phase = calls.append
    keras_module.backend = backend_module

    monkeypatch.setitem(sys.modules, "keras", keras_module)
    monkeypatch.setitem(sys.modules, "keras.backend", backend_module)

    table_10_m2._set_keras_inference_phase()

    assert calls == [0]


def test_create_restored_m2_graph_disables_eager_before_placeholder(monkeypatch) -> None:
    """M2 graph creation should enter TF1 graph mode before placeholders."""
    calls = []

    def fake_disable_eager_execution():
        calls.append("disable_eager")

    def fake_placeholder(dtype, shape, name):
        calls.append("placeholder")
        return (dtype, shape, name)

    fake_tf = types.SimpleNamespace(
        float32="float32",
        compat=types.SimpleNamespace(
            v1=types.SimpleNamespace(
                disable_eager_execution=fake_disable_eager_execution,
                placeholder=fake_placeholder,
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setattr(table_10_m2, "create_tf_session", lambda: "session")
    monkeypatch.setattr(table_10_m2, "_set_keras_inference_phase", lambda: None)
    monkeypatch.setattr(
        table_10_m2,
        "build_mnist_m2_model",
        lambda x_placeholder: ("model", ("predictions", x_placeholder)),
    )
    monkeypatch.setattr(
        table_10_m2,
        "load_mnist_m2_model",
        lambda sess, train_dir: "checkpoint",
    )

    graph = table_10_m2.create_restored_m2_graph("train-dir")

    assert calls == ["disable_eager", "placeholder"]
    assert graph["checkpoint"] == "checkpoint"


def test_m2_nn_robust_loader_patches_tf1_symbols_for_tf2(monkeypatch, tmp_path) -> None:
    """nn_robust_attacks loaders should expose TF1 symbols before module import."""
    calls = []

    class FakeTrain:
        pass

    compat_v1 = types.SimpleNamespace(
        disable_eager_execution=lambda: calls.append("disable_eager"),
        placeholder="placeholder",
        global_variables="global_variables",
        variables_initializer="variables_initializer",
        assign="assign",
        assign_add="assign_add",
        gradients="gradients",
        Session="Session",
        GraphKeys="GraphKeys",
        ConfigProto="ConfigProto",
        train=types.SimpleNamespace(AdamOptimizer="AdamOptimizer"),
    )
    fake_tf = types.SimpleNamespace(
        compat=types.SimpleNamespace(v1=compat_v1),
        train=FakeTrain(),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    attack_root = tmp_path / "nn_robust_attacks"
    attack_root.mkdir()
    (attack_root / "l2_attack.py").write_text(
        "\n".join(
            [
                "import tensorflow as tf",
                "assert tf.placeholder == 'placeholder'",
                "assert tf.global_variables == 'global_variables'",
                "assert tf.variables_initializer == 'variables_initializer'",
                "assert tf.train.AdamOptimizer == 'AdamOptimizer'",
                "class CarliniL2:",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )

    attack_class = table_10_m2._load_nn_robust_carlini_l2(str(attack_root))

    assert attack_class.__name__ == "CarliniL2"
    assert calls == ["disable_eager"]


def test_m2_model_build_accepts_modern_keras_conv2d_arguments(monkeypatch) -> None:
    """M2 construction should fall back to TF2/Keras Conv2D argument names."""
    conv_calls = []

    class FakeConvolution2D:
        def __init__(self, filters, *args, **kwargs):
            if "border_mode" in kwargs:
                raise TypeError("border_mode is not supported")
            conv_calls.append((filters, args, kwargs))

    class FakeSequential:
        def __init__(self):
            self.layers = []

        def add(self, layer):
            self.layers.append(layer)

        def __call__(self, x_placeholder):
            return ("predictions", x_placeholder, len(self.layers))

    layers_module = types.ModuleType("keras.layers")
    layers_module.Convolution2D = FakeConvolution2D
    layers_module.MaxPooling2D = lambda **kwargs: ("maxpool", kwargs)
    layers_module.Activation = lambda name: ("activation", name)
    layers_module.Dense = lambda units: ("dense", units)
    layers_module.Dropout = lambda rate: ("dropout", rate)
    layers_module.Flatten = lambda: "flatten"
    models_module = types.ModuleType("keras.models")
    models_module.Sequential = FakeSequential
    keras_module = types.ModuleType("keras")
    keras_module.layers = layers_module
    keras_module.models = models_module

    monkeypatch.setitem(sys.modules, "keras", keras_module)
    monkeypatch.setitem(sys.modules, "keras.layers", layers_module)
    monkeypatch.setitem(sys.modules, "keras.models", models_module)

    model, predictions = mnist_m2.build_mnist_m2_model("x")

    assert predictions == ("predictions", "x", len(model.layers))
    assert conv_calls[0] == (
        32,
        ((3, 3),),
        {"padding": "valid", "input_shape": (28, 28, 1)},
    )
    assert all(call[2]["padding"] == "valid" for call in conv_calls)


def test_m2_nn_robust_adapter_converts_centered_inputs_to_unit_scale() -> None:
    """The nn_robust_attacks adapter should feed M2 with [0, 1] tensors."""
    captured = []

    class FakeModel:
        def __call__(self, data):
            captured.append(data)
            return data

    centered = np.asarray([[[[-0.5], [0.0], [0.5]]]], dtype=np.float32)
    output = table_10_m2.M2NnRobustAdapter(FakeModel()).predict(centered)

    np.testing.assert_allclose(captured[0], centered + 0.5)
    np.testing.assert_allclose(output, centered + 0.5)


def test_table_10_m2_generation_writes_nn_robust_l2_adversarial_array(
    tmp_path,
) -> None:
    """M2 CW-L2 generation should support the original nn_robust_attacks backend."""
    attack_root = tmp_path / "nn_robust_attacks"
    attack_root.mkdir()
    (attack_root / "l2_attack.py").write_text(
        "\n".join(
            [
                "import numpy as np",
                "class CarliniL2:",
                "    def __init__(self, sess, model, batch_size=1, confidence=0, targeted=False, learning_rate=0.1, binary_search_steps=5, max_iterations=2000, abort_early=True, initial_const=1.0, boxmin=-0.5, boxmax=0.5):",
                "        assert batch_size == 1",
                "        assert confidence == 0.5",
                "        assert targeted is False",
                "        assert learning_rate == 0.1",
                "        assert binary_search_steps == 5",
                "        assert max_iterations == 2000",
                "        assert initial_const == 1.0",
                "        assert boxmin == -0.5",
                "        assert boxmax == 0.5",
                "        self.model = model",
                "    def attack(self, imgs, targets):",
                "        assert imgs.min() == -0.5",
                "        assert targets.shape == (1, 10)",
                "        self.model.predict(imgs)",
                "        return imgs + 0.25",
            ]
        ),
        encoding="utf-8",
    )

    adversarial_path = tmp_path / "cw_l2" / "kappa_0p5" / "adversarial_examples.npy"
    images = np.zeros((1, 28, 28, 1), dtype=np.float32)
    labels = np.zeros((1, 10), dtype=np.float32)
    labels[0, 3] = 1.0

    table_10_m2.generate_adversarial_path(
        graph={"sess": "session", "model": lambda data: data, "x": "x"},
        clean_images=images,
        labels=labels,
        attack_row={
            "norm": "L2",
            "kappa": 0.5,
            "adversarial_path": adversarial_path,
        },
        attack_config={
            "batch_size": 1,
            "max_iterations": 2000,
            "learning_rate": 0.1,
            "binary_search_steps": 5,
            "initial_const": 1.0,
            "targeted": False,
        },
        dataset_config={"name": "mnist", "split": "test", "start": 5500},
        evaluation_config={},
        overwrite=True,
        nn_robust_attacks_root=str(attack_root),
    )

    np.testing.assert_array_equal(np.load(str(adversarial_path)), images + 0.25)
    manifest = yaml.safe_load((adversarial_path.parent / "manifest.json").read_text())
    assert manifest["backend"] == "nn_robust_attacks.CarliniL2"
    assert manifest["model"] == "M2"
    assert manifest["norm"] == "L2"
    assert manifest["kappa"] == 0.5
    assert manifest["dataset_start"] == 5500
    assert manifest["samples"] == 1


def test_table_10_m2_generation_writes_nn_robust_linf_adversarial_array(
    tmp_path,
    capsys,
) -> None:
    """M2 CW-Linf generation should use the original nn_robust_attacks backend."""
    attack_root = tmp_path / "nn_robust_attacks"
    attack_root.mkdir()
    (attack_root / "li_attack.py").write_text(
        "\n".join(
            [
                "import numpy as np",
                "class CarliniLi:",
                "    def __init__(self, sess, model, targeted=False, learning_rate=0.005, max_iterations=1000, abort_early=True, initial_const=1e-5, largest_const=20.0, reduce_const=False, decrease_factor=0.9, const_factor=2.0):",
                "        assert targeted is False",
                "        assert learning_rate == 0.005",
                "        assert max_iterations == 1000",
                "        assert abort_early is True",
                "        assert initial_const == 1e-5",
                "        assert largest_const == 20.0",
                "        assert reduce_const is False",
                "        assert decrease_factor == 0.9",
                "        assert const_factor == 2.0",
                "        self.model = model",
                "    def attack(self, imgs, targets):",
                "        assert imgs.min() == -0.5",
                "        assert targets.shape == (1, 10)",
                "        self.model.predict(imgs)",
                "        return imgs + 0.125",
            ]
        ),
        encoding="utf-8",
    )

    adversarial_path = tmp_path / "cw_linf" / "adversarial_examples.npy"
    images = np.zeros((2, 28, 28, 1), dtype=np.float32)
    labels = np.zeros((2, 10), dtype=np.float32)
    labels[0, 3] = 1.0
    labels[1, 4] = 1.0

    table_10_m2.generate_adversarial_path(
        graph={"sess": "session", "model": lambda data: data, "x": "x"},
        clean_images=images,
        labels=labels,
        attack_row={
            "attack": "CW",
            "norm": "Linf",
            "kappa": None,
            "adversarial_path": adversarial_path,
        },
        attack_config={
            "max_iterations": 1000,
            "learning_rate": 0.005,
            "initial_const": 1e-5,
            "largest_const": 20.0,
            "decrease_factor": 0.9,
            "const_factor": 2.0,
            "abort_early": True,
            "reduce_const": False,
            "targeted": False,
        },
        dataset_config={"name": "mnist", "split": "test", "start": 5500},
        evaluation_config={},
        overwrite=True,
        nn_robust_attacks_root=str(attack_root),
    )

    np.testing.assert_array_equal(np.load(str(adversarial_path)), images + 0.125)
    captured = capsys.readouterr()
    assert "cw_linf_sample=1/2" in captured.out
    assert "cw_linf_sample=2/2" in captured.out
    assert "cw_linf_sample_done=1/2 elapsed_seconds=" in captured.out
    assert "cw_linf_sample_done=2/2 elapsed_seconds=" in captured.out
    manifest = yaml.safe_load((adversarial_path.parent / "manifest.json").read_text())
    assert manifest["backend"] == "nn_robust_attacks.CarliniLi"
    assert manifest["model"] == "M2"
    assert manifest["norm"] == "Linf"
    assert manifest["kappa"] is None
