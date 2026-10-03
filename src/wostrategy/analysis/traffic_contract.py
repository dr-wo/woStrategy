"""Provenance required before enriched traffic may support fresh FP analysis."""
from __future__ import annotations

import pandas as pd

from .traffic import COMBINATION_MODE_COMPATIBILITY_MIN, TRAFFIC_EVALUATOR_VERSION


def current_traffic_contract() -> dict[str, str]:
    return {
        "traffic_evaluator_version": TRAFFIC_EVALUATOR_VERSION,
        "traffic_combination_mode": COMBINATION_MODE_COMPATIBILITY_MIN,
    }


def require_current_traffic_contract(frame: pd.DataFrame, *, context: str) -> None:
    """Validate provenance, never manufacture it. Empty frames contain no evidence."""
    if frame.empty:
        return
    expected = current_traffic_contract()
    actual = {key: frame.attrs.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"Incompatible FP traffic {context}: expected {expected}, got {actual}. "
            "Recompute enriched traffic through the current derivation path first; "
            "loading materialized data does not certify its provenance."
        )
    for column, key in (
        ("TrafficEvaluatorVersion", "traffic_evaluator_version"),
        ("TrafficCombinationMode", "traffic_combination_mode"),
    ):
        if column in frame and frame[column].dropna().ne(expected[key]).any():
            raise ValueError(
                f"Incompatible FP traffic {context}: {column} rows contradict "
                f"the declared {expected[key]} provenance."
            )
