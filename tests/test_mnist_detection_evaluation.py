from src.mnist.detection import DetectionDecision, is_detected
from src.mnist.evaluation import DetectorCounts, ExperimentResult


def test_detection_rule_changes_label() -> None:
    assert is_detected(1, 2)
    assert not is_detected(1, 1)


def test_detector_counts_for_detected_restored_sample() -> None:
    counts = DetectorCounts()
    decision = DetectionDecision(
        original_label=7,
        adversarial_label=3,
        filtered_original_label=7,
        filtered_adversarial_label=7,
        true_label=7,
    )

    counts.update_detection(decision)

    assert counts.test_number == 1
    assert counts.tp == 1
    assert counts.ttp == 1
    assert counts.fn == 0
    assert counts.fp == 0
    assert counts.recall == 1.0
    assert counts.precision == 1.0


def test_detector_counts_for_false_positive_and_false_negative() -> None:
    counts = DetectorCounts()
    decision = DetectionDecision(
        original_label=4,
        adversarial_label=9,
        filtered_original_label=1,
        filtered_adversarial_label=9,
        true_label=4,
    )

    counts.update_detection(decision)

    assert counts.fn == 1
    assert counts.fp == 1
    assert counts.recall == 0.0
    assert counts.precision == 0.0


def test_experiment_result_is_serializable() -> None:
    result = ExperimentResult(attack_name="fgsm", counts=DetectorCounts(tp=1))

    summary = result.as_dict()

    assert summary["attack_name"] == "fgsm"
    assert summary["metrics"]["tp"] == 1

