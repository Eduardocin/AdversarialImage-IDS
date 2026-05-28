"""Helpers for standard experiment metadata payloads."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def build_experiment_payload(
    experiment_id: str,
    config: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the standard JSON payload for experiment outputs."""
    payload = {
        "experiment_id": experiment_id,
        "config": dict(config),
        "metrics": [dict(row) for row in rows],
        "extra": dict(extra or {}),
    }
    if "kind" in config:
        payload["kind"] = config["kind"]
    return payload
