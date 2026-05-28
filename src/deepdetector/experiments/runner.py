"""Single experiment runner for configured Table 6-9 workflows."""

from __future__ import annotations

from typing import Any, Dict

from deepdetector.experiments.fgsm_split_runner import run_fgsm_split_experiment
from deepdetector.experiments.filter_candidate_runner import run_filter_candidate_experiment


FILTER_ALIASES: Dict[str, Dict[str, Any]] = {
    "adaptive_quantization": {
        "name": "adaptive_quantization",
        "type": "adaptive_quantization",
    },
    "proposed_filter": {
        "name": "proposed_detection_filter",
        "type": "proposed_detection_filter",
    },
    "proposed_detection_filter": {
        "name": "proposed_detection_filter",
        "type": "proposed_detection_filter",
    },
}


def _mean_filter_config(name: str) -> Dict[str, Any]:
    """Return a configured smoothing filter from a compact name."""
    parts = name.split("_")
    if len(parts) != 2:
        raise ValueError("Unknown filter alias: {0}".format(name))

    mask_type, size_text = parts
    size_parts = size_text.split("x")
    if len(size_parts) != 2 or size_parts[0] != size_parts[1]:
        raise ValueError("Unknown filter alias: {0}".format(name))
    mask_size = int(size_parts[0])
    if mask_size <= 0 or mask_size % 2 == 0:
        raise ValueError("Filter alias must use a positive odd mask size: {0}".format(name))

    if mask_type == "cross":
        return {"name": name, "type": "cross_mean", "radius": mask_size // 2}
    if mask_type == "diamond":
        return {"name": name, "type": "diamond_mean", "radius": mask_size // 2}
    if mask_type == "box":
        return {"name": name, "type": "box_mean", "kernel_size": mask_size}
    raise ValueError("Unknown filter alias: {0}".format(name))


def filter_config_from_alias(alias: Any) -> Dict[str, Any]:
    """Normalize one compact filter alias into factory config."""
    if isinstance(alias, dict):
        return dict(alias)

    name = str(alias)
    if name in FILTER_ALIASES:
        return dict(FILTER_ALIASES[name])
    return _mean_filter_config(name)


def build_experiment_config(
    name: str,
    consolidated_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the runner-specific config for one experiment name."""
    experiments = consolidated_config.get("experiments", {})
    if name not in experiments:
        raise ValueError("Unknown experiment: {0}".format(name))

    defaults = consolidated_config.get("defaults", {})
    experiment = dict(experiments[name])
    kind = str(experiment.get("kind", "")).strip()
    if not kind:
        raise ValueError("Experiment must define kind: {0}".format(name))

    base_config: Dict[str, Any] = {
        "experiment_id": name,
        "kind": kind,
        "dataset": {
            "name": defaults.get("dataset", "mnist"),
            "split": experiment.get("split", "test"),
        },
        "model": {
            "name": defaults.get("model", "mnist_m1"),
            "checkpoint_dir": defaults.get("checkpoint_dir"),
        },
        "attack": {
            "name": defaults.get("attack", "fgsm"),
            "epsilon": defaults.get("epsilon", 0.2),
            "clip_min": defaults.get("clip_min", 0.0),
            "clip_max": defaults.get("clip_max", 1.0),
        },
        "evaluation": {
            "exclude_invalid_pairs": defaults.get("exclude_invalid_pairs", False),
            "batch_size": defaults.get("batch_size", 256),
        },
        "output": {
            "dir": experiment.get("output_dir", "results/experiments/{0}".format(name)),
            "csv": "metrics.csv",
            "json": "metrics.json",
        },
    }

    if kind == "split_eval":
        base_config["dataset"]["slices"] = list(experiment.get("slices", []))
        base_config["filter"] = filter_config_from_alias(experiment.get("filter"))
        return base_config

    if kind == "filter_grid":
        slice_config = dict(experiment.get("slice", {}))
        base_config["dataset"].update(
            {
                "start": slice_config.get("start"),
                "end": slice_config.get("end"),
                "slice_name": slice_config.get("name"),
                "high_entropy_only": bool(experiment.get("high_entropy_only", False)),
                "entropy_threshold": {"min": experiment.get("entropy_min", 5.0)},
            }
        )
        base_config["filters"] = [
            filter_config_from_alias(filter_name)
            for filter_name in experiment.get("filters", [])
        ]
        base_config["selection_stage"] = str(slice_config.get("name", name)).lower()
        return base_config

    raise ValueError("Unknown experiment kind: {0}".format(kind))


def run_experiment(name: str, consolidated_config: Dict[str, Any]):
    """Run one configured experiment by name."""
    config = build_experiment_config(name, consolidated_config)
    kind = config["kind"]

    if kind == "split_eval":
        return run_fgsm_split_experiment(config)
    if kind == "filter_grid":
        return run_filter_candidate_experiment(config)
    raise ValueError("Unknown experiment kind: {0}".format(kind))
