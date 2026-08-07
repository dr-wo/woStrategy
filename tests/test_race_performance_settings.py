from __future__ import annotations

import json

from wodata import get_race_performance_review_event_dir, get_race_performance_review_root
from wostrategy.core.race_performance_settings import (
    resolve_race_performance_settings,
    write_resolved_race_performance_settings,
)


def test_event_cache_precedes_latest_resolved_invocation(tmp_path):
    root = get_race_performance_review_root(tmp_path)
    paths = write_resolved_race_performance_settings(
        {"sample_count": 80_000, "fuel_rate_bounds": [0.0, 0.1]},
        output_root=root,
        year=2026,
        races=[11],
        session="R",
    )
    assert all(path.is_file() for path in paths)

    latest = resolve_race_performance_settings(
        year=2026, round_number=11, session="R", data_root=tmp_path
    )
    assert latest is not None
    assert latest.source == "event_latest_invocation"
    assert latest.settings["sample_count"] == 80_000

    event_dir = get_race_performance_review_event_dir(
        year=2026, round_number=11, session="R", data_root=tmp_path
    )
    metadata = event_dir / "race_performance_2026_11_R_metadata.json"
    metadata.write_text(json.dumps({"sample_count": 40_000}), encoding="utf-8")

    cached = resolve_race_performance_settings(
        year=2026, round_number=11, session="R", data_root=tmp_path
    )
    assert cached is not None
    assert cached.source == "event_cached_result"
    assert cached.settings["sample_count"] == 40_000
