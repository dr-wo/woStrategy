from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wostrategy.model.observation_weighting import get_weight_policy


def observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026] * 5,
            "round": [1, 1, 1, 2, 2],
            "support_clean_lap_count": [1, 4, 9, 1, 9],
        }
    )


def test_uniform_weighting():
    assert get_weight_policy("uniform").weights(observations()).tolist() == [1] * 5


def test_event_normalised_gives_events_equal_total_weight():
    weights = get_weight_policy("event_normalised").weights(observations())
    assert weights[:3].sum() == pytest.approx(1.0)
    assert weights[3:].sum() == pytest.approx(1.0)
    assert weights.tolist() == pytest.approx([1 / 3] * 3 + [1 / 2] * 2)


def test_evidence_weighting_uses_sqrt_then_event_normalises():
    weights = get_weight_policy("evidence_event_normalised").weights(observations())
    assert weights[:3].tolist() == pytest.approx(np.array([1, 2, 3]) / 6)
    assert weights[3:].tolist() == pytest.approx([0.25, 0.75])
    assert weights[:3].sum() == pytest.approx(weights[3:].sum())
