from wodata.artifacts import write_weekend_model_config
from wodata.artifacts import weekend_model_root
import json
import pytest
from wostrategy.analysis.pre_race_model_config import ensure_pre_race_model_config


def _write_fp_evidence(root, contributing=("FP1",)):
    root.mkdir(parents=True, exist_ok=True)
    aggregate = root / "aggregate"
    aggregate.mkdir(exist_ok=True)
    (aggregate / "manifest.json").write_text(json.dumps({"contributing_sessions": contributing}))
    (aggregate / "latest_parameters.csv").write_text("parameter,median\nfuel_rate,0.05\n")
    (root / "fp_tyre_evidence_report.json").write_text("{}")
    for session in contributing:
        directory = root / "sessions" / session
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text("{}")
        (directory / "latest_parameters.csv").write_text("parameter,median\n")


def test_race_prerequisite_repairs_missing_fp_evidence_even_with_valid_config(tmp_path):
    write_weekend_model_config(
        _valid_config(sessions=("FP1",)), year=2026, round_number=7, data_root=tmp_path,
    )
    calls = []

    def runner(values):
        calls.append(values)
        _write_fp_evidence(weekend_model_root(2026, 7, tmp_path))
        return 0

    first = ensure_pre_race_model_config(
        season=2026, round_number=7, data_root=tmp_path,
        session_names=("Practice 1",), runner=runner, require_fp_evidence=True,
    )
    second = ensure_pre_race_model_config(
        season=2026, round_number=7, data_root=tmp_path,
        session_names=("Practice 1",), runner=runner, require_fp_evidence=True,
    )
    assert first.status == "generated" and second.status == "reused"
    assert len(calls) == 1


@pytest.mark.parametrize("contributing", [(), ("R",)])
def test_race_prerequisite_rejects_model_without_pre_race_fp_evidence(tmp_path, contributing):
    def runner(values):
        write_weekend_model_config(
            _valid_config(sessions=("FP1",)), year=2026, round_number=7, data_root=tmp_path,
        )
        _write_fp_evidence(weekend_model_root(2026, 7, tmp_path), contributing)
        return 0

    with pytest.raises(RuntimeError, match="no usable FP evidence or contains Race data"):
        ensure_pre_race_model_config(
            season=2026, round_number=7, data_root=tmp_path,
            session_names=("Practice 1",), runner=runner, require_fp_evidence=True,
        )


def _valid_config(year=2026, round_number=7, sessions=("FP1", "FP2", "FP3")):
    return {
        "artifact_schema_version": 2,
        "season": year,
        "round_number": round_number,
        "source_sessions": list(sessions),
        "sample_count": 100,
        "random_seed": 42,
        "fuel_rate_bounds": [0.0, .1],
        "track_rate_bounds": [-.05, .05],
        "default_degradation_bounds": [0.0, .2],
        "default_compound_delta_bounds": [-1.0, 1.0],
        "reference_compound": "MEDIUM",
        "clean_lap_noise_sigma": .35,
    }


def test_missing_model_config_runs_canonical_producer(tmp_path):
    calls = []

    def runner(values):
        calls.append(values)
        write_weekend_model_config(
            _valid_config(sessions=("FP1",)),
            year=2026,
            round_number=7,
            data_root=tmp_path,
        )
        return 0

    result = ensure_pre_race_model_config(
        season=2026,
        round_number=7,
        data_root=tmp_path,
        session_names=("Practice 1", "Sprint", "Race"),
        runner=runner,
    )

    assert result.status == "generated"
    assert result.reason == "missing"
    assert result.sessions == ("FP1",)
    assert calls[0]["sessions"] == ("FP1",)
    assert calls[0]["plot_output"] is None


def test_existing_valid_model_config_is_reused_without_running_producer(tmp_path):
    write_weekend_model_config(
        _valid_config(), year=2026, round_number=7, data_root=tmp_path
    )

    def unexpected(_values):
        raise AssertionError("valid config must be reused")

    result = ensure_pre_race_model_config(
        season=2026,
        round_number=7,
        data_root=tmp_path,
        session_names=("Practice 1", "Practice 2", "Practice 3", "Race"),
        runner=unexpected,
    )

    assert result.status == "reused"
    assert result.reason == "valid and fresh"


def test_config_identity_cannot_silently_cross_rounds(tmp_path):
    write_weekend_model_config(
        _valid_config(round_number=6), year=2026, round_number=7, data_root=tmp_path
    )
    calls = []

    def runner(values):
        calls.append(values)
        write_weekend_model_config(
            _valid_config(), year=2026, round_number=7, data_root=tmp_path
        )
        return 0

    result = ensure_pre_race_model_config(
        season=2026, round_number=7, data_root=tmp_path, runner=runner
    )
    assert result.status == "generated"
    assert "round 6" in result.reason
    assert calls
