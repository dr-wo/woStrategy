# woStrategy

Formula 1 strategy analysis tools built on FastF1.

This repository contains the strategy-analysis code that was previously living
inside a Fast-F1 fork: session wrappers, long-stint preparation, pre-season test
analysis, and plotting utilities.

## Install

```bash
python -m pip install -e .
```

The package depends on `fastf1`. Your installed FastF1 version determines which
events and telemetry formats are supported.

## Architecture

The package is split into four layers:

```text
model -> analysis -> plots -> script
```

- `wostrategy.model`: mathematical models such as track evolution, fuel correction, and tyre degradation experiments.
- `wostrategy.analysis`: dataframe preparation and domain aggregation.
- `wostrategy.plots`: matplotlib figure rendering.
- `wostrategy.script`: CLI entry points and workflow orchestration.

Track evolution supports both `linear` and `exponential` fits. The exponential
fit uses `y = A * e^(-kx) + B`. Selecting `--track-evolution-fit exponential`
also writes linear comparison plots for comparison.

Telemetry-backed clean-lap filtering can use both the gap to the car ahead and,
when configured by analysis code, the gap to the car behind. The behind-car gap
is derived from telemetry rows where another driver reports the current driver
as `DriverAhead`, then aggregated to the same per-lap min/mean summary shape as
the existing ahead-car metrics.

## Usage

```python
from wostrategy import Session, run_two_day_benchmark_race_sim

session = Session(2026, 2, 3, test=True)

result = run_two_day_benchmark_race_sim(
    year=2026,
    round_number=2,
    benchmark_session=2,
    comparison_session=3,
    output_prefix="temp/r2s2_r2s3",
    min_laps=30,
    reference_laps=57,
    test=True,
)
```

You can also run the pre-season analysis example with:

```bash
python -m wostrategy.script.pre_season_analysis
```

Push-lap track development:

```bash
python -m wostrategy.script.push_lap_track_development \
  --year 2026 \
  --race 4 \
  --section Q \
  --new-tyre-only \
  --track-evolution-fit exponential
```

Qualifying performance tracking:

```bash
python -m wostrategy.script.quali_performance_tracker \
  --year 2026 \
  --race-range "[1, 5]" \
  --target-team Mercedes \
  --new-tyre-only \
  --last-quali-part-only \
  --allow-lap-time-only \
  --track-evolution-fit exponential
```

`--last-quali-part-only` changes only the final driver/team performance
selection. Track evolution is still fitted from all eligible quali push laps
across Q1/Q2/Q3, but each driver's presented result is selected only from the
last qualifying part they reached. For example, a driver eliminated in Q2 uses
only Q2 corrected laps for the final result even if a corrected Q1 lap is
faster.

`--allow-lap-time-only` keeps telemetry as the preferred clean-lap source. If
telemetry loading fails, or the requested per-lap telemetry gap column is
missing/empty, the tracker falls back to lap-time-only push-lap selection for
that race. The final console output includes a brief line such as
`Lap-time-only races: R6` or `Lap-time-only races: none`.

Example local outputs from `temp/`:

<p>
  <img src="../temp/quali_performance_tracker_2026_1-5_Mercedes_linear_fastest.png" alt="Qualifying performance fastest example" width="32%">
  <img src="../temp/quali_performance_tracker_2026_1-5_Mercedes_linear_average.png" alt="Qualifying performance average example" width="32%">
  <img src="../temp/quali_performance_tracker_2026_1-5_Mercedes_linear_best_sectors.png" alt="Qualifying performance best sectors example" width="32%">
</p>
