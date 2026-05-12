from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_removed_reference_paths_are_not_present() -> None:
    assert not (PROJECT_ROOT / "src" / "mnist" / "reference").exists()
    assert not (PROJECT_ROOT / "scripts" / "mnist" / "run_reference.py").exists()
    assert not (PROJECT_ROOT / "scripts" / "run_mnist_legacy.py").exists()
    assert not (PROJECT_ROOT / "scripts" / "train_mnist_m2_carlini.py").exists()
    assert not (PROJECT_ROOT / "scripts" / "check_mnist_legacy_environment.py").exists()


def test_current_mnist_scripts_point_to_project_modules() -> None:
    script_imports = {
        "check_environment.py": "from src.mnist.environment import main",
        "check_m2.py": "from src.mnist.m2 import check_m2",
        "generate_adversarial.py": "from src.mnist.adversarial_examples import generate_cw_examples, generate_fgsm_examples",
        "run_m1.py": "from src.mnist.m1 import run_m1",
        "train_m2.py": "from src.mnist.m2 import main",
        "validate_fgsm_filter.py": "from src.mnist.fgsm_filter_validation import validate_fgsm_filter",
    }

    for script_name, expected_import in script_imports.items():
        script = PROJECT_ROOT / "scripts" / "mnist" / script_name
        assert script.exists()
        assert expected_import in script.read_text(encoding="utf-8")
