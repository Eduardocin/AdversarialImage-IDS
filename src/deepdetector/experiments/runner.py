"""Single experiment runner for configured Table 3-9 workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from deepdetector.experiments.fgsm_split_runner import run_fgsm_split_experiment
from deepdetector.experiments.filter_candidate_runner import run_filter_candidate_experiment
from deepdetector.experiments.table6_runner import run_table6_experiment
from deepdetector.experiments.table7_imagenet_runner import run_table7_imagenet_experiment
from deepdetector.experiments.table8_imagenet_runner import run_table8_imagenet_experiment
from deepdetector.experiments.table9_runner import run_table9_experiment
from deepdetector.experiments.table4_imagenet_runner import run_table4_imagenet_experiment
from deepdetector.experiments.table10_runner import run_table10_group_experiment
from deepdetector.io.paths import resolve_project_path
from deepdetector.io.result_writers import write_metrics_json


FILTER_ALIASES: Dict[str, Dict[str, Any]] = {
    "scalar_quantization_2": {
        "name": "scalar_quantization_2",
        "type": "scalar_quantization",
        "intervals": 2,
    },
    "nonuniform_quantization": {
        "name": "nonuniform_quantization",
        "type": "nonuniform_quantization",
    },
    "nonuniform_quantization_legacy": {
        "name": "nonuniform_quantization_legacy",
        "type": "nonuniform_quantization_legacy",
    },
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

    experiment = dict(experiments[name])
    kind = str(experiment.get("kind", "")).strip()
    if not kind:
        raise ValueError("Experiment must define kind: {0}".format(name))

    output_defaults = dict(consolidated_config.get("defaults", {}).get("output", {}))

    if kind == "composite":
        components = list(experiment.get("components", []))
        if not components:
            raise ValueError("Composite experiment must define components: {0}".format(name))
        return {
            "experiment_id": name,
            "kind": kind,
            "components": components,
            "output": {
                "dir": experiment.get("output_dir", "results/experiments/{0}".format(name)),
                "json": "manifest.json",
            },
        }

    base_config: Dict[str, Any] = {
        "experiment_id": name,
        "kind": kind,
        "dataset": dict(experiment.get("dataset", {})),
        "model": dict(experiment.get("model", {})),
        "attack": dict(experiment.get("attack", {})),
        "evaluation": dict(experiment.get("evaluation", {})),
        "output": {
            "dir": experiment.get("output_dir", "results/experiments/{0}".format(name)),
            "csv": output_defaults.get("csv", "metrics.csv"),
            "json": output_defaults.get("json", "metrics.json"),
        },
    }
    base_config["output"].update(dict(experiment.get("output", {})))

    if kind == "split_eval":
        if "slices" in experiment:
            base_config["dataset"]["slices"] = list(experiment.get("slices", []))
        base_config["filter"] = filter_config_from_alias(experiment.get("filter"))
        return base_config

    if kind == "table_6":
        base_config["mnist"] = dict(experiment.get("mnist", {}))
        base_config["imagenet"] = dict(experiment.get("imagenet", {}))
        base_config["split_order"] = list(experiment.get("split_order", ["train", "validation"]))
        base_config["entropy_thresholds"] = dict(experiment.get("entropy_thresholds", {}))
        base_config["quantization"] = dict(experiment.get("quantization", {}))
        return base_config

    if kind == "table_9":
        base_config["mnist"] = dict(experiment.get("mnist", {}))
        base_config["imagenet"] = dict(experiment.get("imagenet", {}))
        base_config["split_order"] = list(experiment.get("split_order", ["train", "validation"]))
        return base_config

    if kind == "imagenet_table_4":
        base_config["quantization"] = dict(experiment.get("quantization", {}))
        return base_config

    if kind == "imagenet_table_7":
        base_config["filter"] = dict(experiment.get("filter", {}))
        return base_config

    if kind == "imagenet_table_8":
        base_config["filter"] = dict(experiment.get("filter", {}))
        return base_config

    if kind == "table_10_group":
        base_config["model_group"] = str(experiment.get("model_group", ""))
        base_config["dataset_label"] = str(experiment.get("dataset_label", ""))
        base_config["rows"] = [dict(row) for row in experiment.get("rows", [])]
        base_config["filter"] = dict(experiment.get("filter", {}))
        return base_config

    if kind == "filter_grid":
        dataset_override = dict(experiment.get("dataset", {}))
        slice_config = dict(experiment.get("slice", {}))
        if dataset_override:
            slice_config = dataset_override
        base_config["dataset"].update(
            {
                "name": slice_config.get("name"),
                "split": slice_config.get("split"),
                "start": slice_config.get("start"),
                "end": slice_config.get("end"),
                "slice_name": slice_config.get("slice_name") or slice_config.get("label"),
                "high_entropy_only": bool(
                    slice_config.get(
                        "high_entropy_only",
                        experiment.get("high_entropy_only", False),
                    )
                ),
                "entropy_threshold": slice_config.get(
                    "entropy_threshold",
                    {"min": experiment.get("entropy_min", 5.0)},
                ),
            }
        )
        base_config["filters"] = [
            filter_config_from_alias(filter_name)
            for filter_name in experiment.get("filters", [])
        ]
        if not base_config["filters"]:
            raise ValueError("Experiment must define at least one filter: {0}".format(name))
        selection_stage = (
            base_config["dataset"].get("slice_name")
            or slice_config.get("label")
            or name
        )
        base_config["selection_stage"] = str(selection_stage).lower()
        return base_config

    raise ValueError("Unknown experiment kind: {0}".format(kind))


def _component_manifest_entry(
    component_name: str,
    component_config: Dict[str, Any],
    result: Any,
) -> Dict[str, Any]:
    """Return a compact manifest entry for one composite component."""
    entry: Dict[str, Any] = {
        "experiment_id": component_name,
        "kind": component_config["kind"],
        "output_dir": component_config["output"]["dir"],
    }
    if isinstance(result, dict):
        entry.update(result)
        return entry
    if isinstance(result, list):
        entry["status"] = "completed"
        entry["n_rows"] = len(result)
        return entry
    entry["status"] = "completed"
    return entry


def _run_composite_experiment(
    config: Dict[str, Any],
    consolidated_config: Dict[str, Any],
) -> list[Dict[str, Any]]:
    """Run component experiments in declared order and write a manifest."""
    experiment_id = config["experiment_id"]
    component_entries: list[Dict[str, Any]] = []
    for component_name in config["components"]:
        if component_name == experiment_id:
            raise ValueError("Composite experiment cannot include itself: {0}".format(experiment_id))
        component_config = build_experiment_config(component_name, consolidated_config)
        result = run_experiment(component_name, consolidated_config)
        component_entries.append(
            _component_manifest_entry(component_name, component_config, result)
        )

    output_dir = resolve_project_path(config["output"]["dir"]) or Path(config["output"]["dir"])
    write_metrics_json(
        output_dir / config["output"].get("json", "manifest.json"),
        {
            "experiment_id": experiment_id,
            "kind": config["kind"],
            "components": component_entries,
        },
    )
    return component_entries


def run_experiment(name: str, consolidated_config: Dict[str, Any]):
    """Run one configured experiment by name."""
    config = build_experiment_config(name, consolidated_config)
    kind = config["kind"]

    if kind == "composite":
        return _run_composite_experiment(config, consolidated_config)
    if kind == "split_eval":
        return run_fgsm_split_experiment(config)
    if kind == "table_6":
        return run_table6_experiment(config)
    if kind == "table_9":
        return run_table9_experiment(config)
    if kind == "filter_grid":
        return run_filter_candidate_experiment(config)
    if kind == "imagenet_table_4":
        return run_table4_imagenet_experiment(config)
    if kind == "imagenet_table_7":
        return run_table7_imagenet_experiment(config)
    if kind == "imagenet_table_8":
        return run_table8_imagenet_experiment(config)
    if kind == "table_10_group":
        return run_table10_group_experiment(config)
    raise ValueError("Unknown experiment kind: {0}".format(kind))
