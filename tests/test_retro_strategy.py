from wostrategy.analysis.retro_strategy import calculate_retro_green_optimum


def test_retro_green_optimum_reuses_rules_compliant_optimizer() -> None:
    result = calculate_retro_green_optimum(
        retro_tyre_estimate={"compounds": {
            "SOFT": {"performance_delta_to_medium": -0.4, "degradation_seconds_per_lap": 0.2},
            "MEDIUM": {"performance_delta_to_medium": 0.0, "degradation_seconds_per_lap": 0.1},
            "HARD": {"performance_delta_to_medium": 0.3, "degradation_seconds_per_lap": 0.05},
        }},
        total_laps=12,
        green_pit_loss=2.0,
    )
    assert result["internal_id"] == "retro_green_optimum"
    assert result["strategies"]
    assert all(len(set(row["compounds"])) >= 2 for row in result["strategies"])
