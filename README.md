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

The package is split into five layers:

```text
model -> algorithm -> analysis -> plots -> script
```

- `wostrategy.model`: mathematical models such as track evolution, fuel correction, and tyre degradation experiments.
- `wostrategy.algorithm`: reusable sampling and optimization algorithms that should not know about loading, plotting, or persistence.
- `wostrategy.analysis`: dataframe preparation and domain aggregation.
- `wostrategy.plots`: matplotlib figure rendering.
- `wostrategy.script`: CLI entry points and workflow orchestration.

Track evolution supports both `linear` and `exponential` fits. The exponential
fit uses `y = A * e^(-kx) + B`. Selecting `--track-evolution-fit exponential`
also writes linear comparison plots for comparison.

Telemetry-backed clean-lap filtering can use both the gap to the car ahead and,
when configured by analysis code, the gap to the car behind. The behind-car gap
is preferably derived from synchronized session-time track positions so lapped
traffic can still be detected physically; the older `DriverAhead` inversion is
kept as a fallback. Stale telemetry cache files that predate the derived
`TimeDeltaToDriverAhead` column are rebuilt automatically.

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

Monte Carlo race performance review:

```bash
python -m wostrategy.script.race_performance_review \
  --year 2026 \
  --race 7 \
  --session R \
  --sample-count 50000 \
  --sampling-strategy latin-hypercube \
  --fuel-rate-bounds 0 0.10 \
  --track-rate-bounds -0.05 0.05 \
  --default-compound-degradation-bounds 0 0.50 \
  --tyre-delta-bounds -1.0 1.0 \
  --compound-delta-reference HARD \
  --team-variation-fraction 0.5 \
  --team-variation-absolute-min 0.005 \
  --clean-lap-noise-sigma 0.5 \
  --team-baseline-mode average-drivers
```

This workflow loads race laps with telemetry gap summaries, requires telemetry
clean-air data, skips races only when the median driver wet/intermediate lap
proportion exceeds the configured threshold, and runs a weighted Monte Carlo
correction model. Each sample draws global fuel and track-evolution rates,
compound degradation rates, compound lap-time deltas relative to a reference
compound, and bounded team-compound degradation variation. Corrected clean laps
are fitted to driver or team baselines, scored by global RMSE, and converted to
Gaussian-like weights.
Use `--limit-negative-track-correction` to clamp sampled track-evolution rates
to non-negative values, preventing the correction from making later-race laps
longer as the track ages.

Weighting is configurable with `--weight-strategy`. The default `gaussian`
keeps the original unnormalized weighting:

```python
weight = exp(-(rmse**2) / (2 * sigma**2))
```

`best-rmse-relative` uses the best sampled RMSE as the likelihood reference and
normalizes weights to sum to one:

```python
best_rmse = min(rmse)
weight = exp(-N_eff * (rmse**2 - best_rmse**2) / (2 * sigma**2))
weight = weight / sum(weight)
```

Use `--weight-effective-sample-count` to provide `N_eff`; by default it uses
the clean lap count.

Clean-air filtering uses telemetry-derived gap summaries. When physical track
position samples are available, both ahead and behind gaps are derived from
same-session-time car distances so lapping scenarios are handled consistently;
otherwise the workflow falls back to the existing FastF1 driver-ahead stream.
`--min-clean-air-laps` is interpreted as an inclusive minimum, so a four-lap
block is accepted when the value is `4`.

By default, clean laps are selected as consecutive clean-air chunks within a
driver stint. Use `--treat-stint-as-whole` to instead group all clean laps from
the same driver/stint into one run when the stint has at least
`--min-clean-air-laps` clean laps in total.

Tyre-age correction defaults to `--tyre-age-mode stint`, so the first lap of a
stint is treated as tyre age 0 even if the fitted tyre set was already used.
Use `--tyre-age-mode overall` to instead use the session `TyreLife` value.

Team corrected baseline pace can be reported by averaging driver baselines,
taking the best corrected driver baseline, or fitting directly at team level.
CSV outputs are written to `cache/race_performance_review/` by default,
including clean laps, sampled parameters, degradation samples, baseline samples,
team baseline samples, and weighted P10/median/P90 summaries. Compound deltas
are saved separately from degradation slopes so tyre grip offsets and tyre-age
degradation can be inspected independently.

The workflow also saves `*_sample_diagnostics.csv` with quantitative fit-health
metrics including best RMSE, weighted RMSE, RMSE P10/median/P90, total weight,
effective sample size, effective sample fraction, and top-1% weight share.

Use `--use-cached-monte-carlo` to reuse existing per-race Monte Carlo CSVs when
they are present and calculate only missing races. Disable it with
`--no-use-cached-monte-carlo` to rerun all requested races from scratch.
When cached `*_baseline_pace.csv` and `*_sample_parameters.csv` are available,
the requested `--team-baseline-mode` is rebuilt from cached driver/team baseline
samples, so switching between `average-drivers` and `best-driver` does not
require rerunning Monte Carlo.

When track temperature is above `--degradation-order-track-temperature`
(`20` Celsius by default), the base compound degradation sampler enforces
`SOFT >= MEDIUM >= HARD`. The same temperature gate also constrains compound
lap-time deltas so the softer compound is quicker at the tyre-age reference:
`SOFT <= MEDIUM <= HARD` because lower delta means faster. Use
`--track-temperature` to provide the actual track temperature when the loaded
lap dataframe does not contain a track temperature column.

The same script also saves a race performance tracker plot to `temp/` by
default. It plots each team's weighted median corrected race baseline as a
percentage of `--reference-team`, using F1 team colors:

```bash
python -m wostrategy.script.race_performance_review \
  --year 2026 \
  --race "[1, 7]" \
  --reference-team Mercedes \
  --plot-output temp/race_performance_tracker_2026_1-7_mercedes.png
```

The P10/P90 uncertainty band is not drawn by default. Add
`--plot-uncertainty-band` when you want it on the tracker plot.

Add `--plot-rmse-background` to shade each GP by the Monte Carlo
`WeightedRMSESeconds` diagnostic: below 0.5s is green, 0.75s is orange, and
1.0s or higher is red. A colorbar is added to the plot. If fewer than five
teams have data for a GP, that GP is marked with a black striped background
instead.
