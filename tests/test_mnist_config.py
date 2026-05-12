from src.mnist.config import MnistExperimentConfig


def test_mnist_defaults_match_reference_values() -> None:
    config = MnistExperimentConfig()

    assert config.m1.epochs == 6
    assert config.m1.batch_size == 128
    assert config.m1.learning_rate == 0.001
    assert config.attacks.fgsm_train_eps == 0.2
    assert config.attacks.fgsm_test_eps == 0.3
    assert config.attacks.cw_l2_max_iterations == 2000
    assert config.attacks.cw_l2_binary_search_steps == 5
    assert config.attacks.cw_l2_initial_const == 1.0
    assert config.attacks.cw_l2_learning_rate == 1e-1
    assert config.attacks.cw_linf_max_iterations == 1000
    assert config.detection.low_entropy_interval == 128
    assert config.detection.mid_entropy_interval == 64
    assert config.detection.high_entropy_interval == 43

