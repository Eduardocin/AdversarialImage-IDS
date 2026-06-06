"""Experiment runner wrapper for defense-aware adaptive CW-L2."""

from __future__ import annotations

from typing import Any, Dict

from deepdetector.evaluation.defense_aware import run_defense_aware_experiment


def run_defense_aware(config: Dict[str, Any]):
    """Run the configured defense-aware experiment."""
    return run_defense_aware_experiment(config)
