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
the existing ahead-car metrics. Stale telemetry cache files that predate the
derived `TimeDeltaToDriverAhead` column are rebuilt automatically.

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
  --race 7 \
  --section Q \
  --new-tyre-only \
  --allow-lap-time-only \
  --track-evolution-fit exponential
```

The script prefers telemetry gap summaries for clean-lap selection. If telemetry
is unavailable or does not contain the requested gap column, `--allow-lap-time-only`
falls back to lap-time-only push-lap selection and prints
`Lap-time-only mode: True`.

Tyre strategy summary for one feature race:

```bash
python -m wostrategy.script.tyre_strategy_summary \
  --year 2026 \
  --race 6
```

This prints Chinese and English two-column strategy tables ordered by finishing
position, and writes CSVs to `temp/` by default:

```text
temp/tyre_strategy_summary_2026_6_R_chinese.csv
temp/tyre_strategy_summary_2026_6_R_english.csv
```

Pure lap-time trace comparison:

```bash
python -m wostrategy.script.pure_lap_time_trace \
  --year 2026 \
  --race 7 \
  --session R \
  --traces-json '{"RUS": {"lap": ["37-61"], "off-set": 0.12}, "HAM": {"lap": ["41-61"], "off-set": 0}}' \
  --delta-traces-json '{"RUS vs HAM": {"trace_a": "RUS", "trace_b": "HAM", "lap": ["7-21"]}}' \
  --y-range 80 83
```

The script first collects laps from `traces`, plots each trace against collected
lap number, then optionally computes accumulated deltas from that collected
plot data. Delta trace `lap` values are collected lap numbers, not real race
lap numbers. In the example above, collected lap 7 for `RUS` is compared with
collected lap 7 for `HAM`, even if those are different real race laps. The
accumulated delta is drawn as a black line on the right-hand axis. Positive
delta means `trace_a` lost time to `trace_b`.

Qualifying performance tracking:

```bash
python -m wostrategy.script.quali_performance_tracker \
  --year 2026 \
  --race-range "[1, 7]" \
  --target-team Mercedes \
  --new-tyre-only \
  --last-quali-part-only \
  --allow-lap-time-only \
  --track-evolution-fit exponential
```

`--last-quali-part-only` changes only the final driver/team performance
selection. Track evolution is still fitted from all eligible quali push laps
across Q1/Q2/Q3, but each driver's presented result is selected only from the
last qualifying part they entered. For example, a driver eliminated in Q2 uses
only Q2 corrected laps for the final result even if a corrected Q1 lap is
faster. A driver who enters Q3 but has no valid Q3 push lap is not allowed to
fall back to a faster Q2 lap.

`--allow-lap-time-only` keeps telemetry as the preferred clean-lap source. If
telemetry loading fails, or the requested per-lap telemetry gap column is
missing/empty, the tracker falls back to lap-time-only push-lap selection for
that race. The final console output includes a brief line such as
`Lap-time-only races: R6` or `Lap-time-only races: none`.

For every plot set, the tracker also writes a `<output-stem>_usage.csv` file.
The CSV includes the plotted result rows and a `SourceLaps` column showing
exactly which driver, qualifying part, lap, or sector contributed to each
`fastest`, `average`, and `best_sectors` point.

Example local outputs from `temp/`:

<p>
  <img src="../temp/quali_performance_tracker_2026_1-5_Mercedes_linear_fastest.png" alt="Qualifying performance fastest example" width="32%">
  <img src="../temp/quali_performance_tracker_2026_1-5_Mercedes_linear_average.png" alt="Qualifying performance average example" width="32%">
  <img src="../temp/quali_performance_tracker_2026_1-5_Mercedes_linear_best_sectors.png" alt="Qualifying performance best sectors example" width="32%">
</p>

Race long-run performance:

```bash
python -m wostrategy.script.long_run_performance \
  --year 2026 \
  --race-range 4 7 \
  --section R \
  --reference-team Mercedes \
  --track-evolution-rate-source quali \
  --clean-mean-time-delta-seconds 3 \
  --clean-mean-time-delta-behind-seconds 1
```

This workflow filters consecutive clean-air race runs, fits driver stint
performance against tyre age, removes obvious stint-estimate outliers, corrects
compound estimates to a shared tyre-life-zero reference lap, and aggregates to
team performance by compound usage. With `--track-evolution-rate-source quali`,
it runs or reuses the linear qualifying track-evolution rate for the same race
and applies that rate to race stint estimates.

Outputs are written to `--output-dir` (`temp/` by default), including filtered
lap CSVs, fitted stint summaries, driver-estimate sanity diagnostics,
track-evolution correction stats, team/compound reference estimates, final team
performance CSVs, driver fit plots, and an aggregate long-run performance trend
plot. Race tick labels include event names when FastF1 event metadata is
available.
