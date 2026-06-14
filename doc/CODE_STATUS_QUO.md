# woStrategy Code Status Quo

Last reviewed: 2026-06-14

This document is a compact map of the current `woStrategy` package so future coding agents can orient quickly without rereading every module. It describes the repository as observed in the working tree, including uncommitted and untracked files.

## Package Purpose

`woStrategy` is a Python package for Formula 1 strategy analysis on top of FastF1. It wraps FastF1 session loading, adds project-specific lap enrichment, prepares long-stint, pre-season testing, push-lap, and qualifying datasets, and renders matplotlib plots for race simulations, cumulative laps, single-lap comparisons, telemetry gap maps, push-lap track development, and qualifying performance tracking.

The package is configured in `pyproject.toml`:

- Build backend: `hatchling`
- Package source: `src/wostrategy`
- Python requirement: `>=3.9`
- Runtime dependencies: `fastf1`, `matplotlib`, `numpy`, `pandas`
- Tests configured under `tests`

## Repository Layout

```text
woStrategy/
  README.md
  pyproject.toml
  cache/telemetry/              # Local pickle cache for per-session telemetry
  doc/                          # Agent/project docs
  src/wostrategy/
    __init__.py                 # Public package exports
    core/                       # Session wrapper, lap loaders, telemetry cache/loaders
    model/                      # Mathematical model implementations
    analysis/                   # Domain analysis and dataframe aggregation
    tools/                      # Data preparation and end-to-end analysis workflows
    plots/                      # Matplotlib plot renderers and style maps
    script/                     # CLI-style runnable scripts
  tests/
    test_clean_lap_track_development.py
    test_quali_performance_tracker.py
    test_telemetry_loader.py
```

Current intended dependency direction:

```text
model -> analysis -> plots -> script
```

- `model` contains reusable math/model fitting code such as track evolution, fuel correction, and tyre degradation experiments.
- `analysis` prepares and aggregates domain dataframes, calls model objects where needed, and does not render figures.
- `plots` renders matplotlib figures from already prepared data.
- `script` owns CLI defaults, argument parsing, session loading, output paths, printing, and figure closing.

## Public API Surface

`wostrategy.__init__` re-exports the main APIs:

- Core session and telemetry: `Session`, `TelemetryDataLoader`, `DistanceInterpolationTimeDeltaEstimator`, `TimeDeltaEstimator`, `load_session_telemetry`, `load_or_cache_session_telemetry`, `get_session_telemetry_cache_path`, `summarize_lap_gap_metrics`
- Lap loaders: `load_session_laps`, `load_session_laps_with_telemetry_gap_summary`, `load_all_session_laps`, `load_all_session_laps_with_telemetry_gap_summary`
- Analysis helpers: `add_half_day_label`, `add_session_value_column`, `forenoon_afternoon_delta`, `export_long_effective_stints`, `run_two_day_benchmark_race_sim`

Plot functions are exposed through `wostrategy.plots`, not through the root package.

`wostrategy.analysis` re-exports selected model objects for convenience, but new model implementation imports should use `wostrategy.model`.

## Model Layer

`src/wostrategy/model/track_evolution.py`

- Defines the track-evolution fitting interface and implementations:
  - `TrackEvolutionModel`
  - `LinearTrackEvolutionModel`
  - `ExponentialTrackEvolutionModel`
  - `TrackEvolutionFit`
- Supported fit names:
  - `linear`: `lap_time = slope * x + intercept`
  - `exponential`: `lap_time = A * e^(-kx) + B`
- Exponential fitting uses NumPy only: it scans candidate `k` values and solves `A`/`B` via least squares for each candidate. No SciPy dependency is required.
- Shared helpers include:
  - `get_track_evolution_model`
  - `get_track_evolution_term_config`
  - `fit_compound_track_evolution`
  - `add_track_evolution_correction`
  - `dominant_compound`
- Correction columns include:
  - `track_evolution_fit_model`
  - `track_evolution_seconds_per_lap`
  - `track_evo_correction_seconds`
  - `track_evo_corrected_lap_time`
  - `track_evo_corrected_lap_time_seconds`

`src/wostrategy/model/fuel_consumption.py`

- Contains `FuelCorrection` and `FixedRateFuelCorrection`.
- `FixedRateFuelCorrection` estimates initial fuel by session type and can add `RemainingFuel` and `FuelCorrection`.
- Also defines a linear fuel-consumption model interface and term-config helper for long-run model presets.
- Still experimental and not yet covered by the focused tests.

`src/wostrategy/model/tyre_degragation.py`

- Contains exploratory tyre degradation code and a linear tyre-degradation model interface used by long-run model presets.
- The previous import-time execution was moved behind `main()` so importing the module is safe.
- File name still has a spelling issue: `tyre_degragation.py`.

`src/wostrategy/model/long_run_performance.py`

- Defines configurable long-run lap-time model presets and fitting helpers.
- Presets include `linear_components` and `linear_components_exponential_track`.
- Supports custom model configs composed of linear and exponential terms.
- Provides `LongRunLapTimeFit`, `get_long_run_model_config`, `fit_long_run_lap_time_model`, `normalize_long_run_model_config`, and `reference_values_from_config`.

Compatibility note:

- `src/wostrategy/analysis/track_evolution.py`
- `src/wostrategy/analysis/fuel_consumption.py`
- `src/wostrategy/analysis/tyre_degragation.py`

These are thin re-export shims to keep older imports working while model implementations live in `wostrategy.model`.

## Core Session Model

`src/wostrategy/core/session.py`

`Session` is a chainable wrapper around `fastf1.core.Session`.

Constructor behavior:

- Accepts `year`, `round`, `session_name`, and `test`.
- Uses `fastf1.get_session(...)` for normal events.
- Uses `fastf1.get_testing_event(...).get_session(...)` when `test=True`.
- Calls `data.load()` immediately.
- Stores original laps in `_original_laps`, exposes mutable current laps as `self.laps`.
- Computes `lap_distance` from `get_circuit_info().marshal_sectors["Distance"].max()`.
- Computes `race_lap_number` as `max(ceil(300000 / lap_distance), 78)`.
- Adds `StintLapNumber` via groupby on `Driver` and `Stint`.

Chainable filters:

- `drivers(drivers)`
- `quicklaps(threshold=1.07)`
- `compounds(compounds)`
- `track_status(status)`
- `stint(stint)`
- `clean_laps(car_ahead)`
- `effective_stint()`
- `reset()`

Important behavior:

- `_fill_missing_outlap_laptimes` fills missing out-lap `LapTime` when `PitOutTime` and the next `LapStartTime` can produce a positive duration.
- Missing or invalid circuit distance falls back to `race_lap_number = 78`, so sessions with incomplete circuit metadata can still load laps.
- `clean_laps` computes a time gap from FastF1 car data using `DistanceToDriverAhead / speed`.
- `effective_stint` adds `EffectiveStint` and `EffectiveStintLapNumber`; it heuristically marks short or slow-lap-heavy stints as quali-sim and combines eligible stints with small pit gaps.

Current caveat:

- `effective_stint` prints a lot of debug output directly and mutates `self.laps` in place.

## Lap Loading

`src/wostrategy/core/session_loader.py`

`load_session_laps` is the generic batch loader. It loops over `rounds` and `session_names`, builds a session using an injected `session_factory`, optionally runs `enrich_session(session)`, copies `session.laps`, and appends `Year`, `Round`, `SessionName`, available FastF1 event metadata, and available session result rank.

`load_session_laps_with_telemetry_gap_summary` does the same, then loads or reuses full telemetry for each session and merges per-lap gap summary columns:

- `MinTimeDeltaToDriverAhead`
- `MeanTimeDeltaToDriverAhead`
- `MinDistanceToDriverAhead`
- `MeanDistanceToDriverAhead`
- `MinTimeDeltaToDriverBehind`
- `MeanTimeDeltaToDriverBehind`
- `MinDistanceToDriverBehind`
- `MeanDistanceToDriverBehind`
- `SessionResultRank`, when FastF1 `session.results` exposes driver result positions

Both loaders catch exceptions per session, print skip messages, and continue.

Session event metadata behavior:

- `_add_session_event_metadata` reads `session.data.event` when using the project `Session` wrapper, or `session.event` for FastF1-like test doubles.
- It adds any available `EventName`, `EventFormat`, `EventCountry`, `EventLocation`, and `OfficialEventName`.
- Plot helpers use this metadata to show event-aware race tick labels.

Session result rank behavior:

- `_add_session_result_rank` reads `session.data.results` when using the project `Session` wrapper, or `session.results` for FastF1-like test doubles.
- It maps `results["Position"]` by `results["Abbreviation"]` to lap rows as `SessionResultRank`.
- If results are unavailable or do not expose expected columns, loaders proceed without this column.

## Telemetry Loading And Cache

`src/wostrategy/core/telemetry_loader.py`

Default cache path:

```python
DEFAULT_TELEMETRY_CACHE_DIR = <repo>/cache/telemetry
```

`get_session_telemetry_cache_path` names cache files as:

```text
<year>_<round>_<session>
```

There is no file extension by default. Unsafe cache-name characters are replaced with `-`.

`DistanceInterpolationTimeDeltaEstimator`:

- Requires `Distance`, `DistanceToDriverAhead`, and `Time`.
- Converts `Time` to seconds.
- Builds a monotonic distance to time axis for a lap.
- Estimates `TimeDeltaToDriverAhead` by interpolating the time at `Distance + DistanceToDriverAhead`.
- Handles wraparound past the end of the lap by applying lap offsets.
- Returns `NaN` for invalid or impossible samples.

`TelemetryDataLoader`:

- Iterates `session.laps.iterlaps()`.
- Calls each lap's `get_telemetry()` by default.
- Adds `TimeDeltaToDriverAhead`.
- Copies metadata columns from the lap, including driver, team, lap number, stint, compound, tyre life, lap time, lap start, and pit markers.
- Skips lap-level errors by default.

Top-level telemetry helpers:

- `load_session_telemetry`: batch loads full telemetry across sessions.
- `load_or_cache_session_telemetry`: reads a pickle cache unless `force_refresh=True`; otherwise loads and writes a pickle. Existing cache files that are missing the current estimator output column, for example `TimeDeltaToDriverAhead`, are treated as stale and rebuilt.
- `summarize_lap_gap_metrics`: aggregates full telemetry to per-lap min and mean time/distance gaps.
- Behind-car gaps are derived by matching telemetry rows where another car's `DriverAhead` equals the current driver's `DriverNumber`; when this cannot be derived, the behind-gap columns are still present with missing values.

## Tools

`src/wostrategy/tools/load_sessions.py`

- `load_all_session_laps`: user-facing wrapper around `load_session_laps`.
- `load_all_session_laps_with_telemetry_gap_summary`: same, with telemetry cache and gap-summary merge.

`src/wostrategy/tools/forenoon_afternoon.py`

- `add_half_day_label` adds `LapStartSeconds` and `HalfDay` using a 4.5 hour cutoff. Values before cutoff are `forenoon`; everything else, including missing values, becomes `afternoon`.
- `forenoon_afternoon_delta` loads effective-stint laps, filters obvious quali-sim/very slow laps, writes a CSV summary, writes a scatter PNG beside the CSV, and returns a map `{(Round, SessionName): afternoon_minus_forenoon_delta}`.

`src/wostrategy/tools/long_effective_stints.py`

- `export_long_effective_stints` loads sessions with `effective_stint`, keeps effective stints whose max `EffectiveStintLapNumber` is greater than `min_laps`, writes the selected laps to CSV, prints a summary, and returns the result dataframe.

`src/wostrategy/tools/session_values.py`

- `add_session_value_column` maps per-session scalar values into a dataframe using `(Round, SessionName)` keys.

`src/wostrategy/tools/two_day_benchmark_race_sim.py`

`run_two_day_benchmark_race_sim` orchestrates the two-day benchmark race-sim flow:

1. Compute AM/PM correction via `forenoon_afternoon_delta`.
2. Export long effective stints.
3. Build an AM-to-AM session offset between benchmark and comparison sessions.
4. Prepare race-sim data with corrections and offsets.
5. Render uncorrected and corrected race-sim plots.

Return keys:

- `correction_map`
- `session_offset_map`
- `long_laps`
- `plots`
- `morning_session_summary`

## Analysis Layer

`src/wostrategy/analysis/push_laps.py`

- Contains push-lap dataframe preparation and selection logic.
- `PushLapSelector` wraps the common configuration:
  - quick-lap threshold
  - min/mean clean gap filter to the car ahead
  - optional min/mean clean gap filter to the car behind
  - dry compounds
  - new-tyre-only filtering
- `add_push_lap_flags` marks:
  - `IsQuickLap`
  - `IsCleanLap`
  - `IsPushLap`
  - `LapPatternRole`
  - `LapStartSeconds`
  - `LapStartMinutes`
  - `SessionLapOrder`
- Push laps are recognized inside out-lap / push-lap / in-lap style patterns.
- Quick laps are defined against each driver's fastest uncorrected `LapTime` in the loaded session.
- Clean-gap filtering requires exactly one of:
  - `clean_min_time_delta_seconds`
  - `clean_mean_time_delta_seconds`
- Behind-car clean-gap filtering is optional. When configured, exactly one of the corresponding behind thresholds should be used:
  - `clean_min_time_delta_behind_seconds`
  - `clean_mean_time_delta_behind_seconds`
- A clean threshold of `0` is accepted and means no clean-gap cut beyond the quick-lap requirement.
- `select_dry_push_laps` filters to dry compounds and, by default in scripts, new tyres only via `FreshTyre`.
- `select_top_drivers` prefers `SessionResultRank`; it falls back to fastest uncorrected session lap if result ranks are unavailable.

`src/wostrategy/analysis/quali_performance.py`

- Contains qualifying performance analysis and team aggregation.
- `QualiPerformanceAnalyzer` wraps repeated calculation configuration.
- `calculate_quali_performance`:
  - returns `"Wet"` when wet/intermediate compounds are present
  - flags clean dry push laps
  - optionally filters to new tyres only
  - can run in lap-time-only mode, where clean laps are quick non-in/out laps without telemetry gap filtering
  - fits track evolution on the dominant dry compound using `wostrategy.model.track_evolution`
  - optionally filters only the fit sample to top drivers
  - applies the selected fit to eligible laps
  - optionally restricts final performance laps to each driver's latest entered qualifying part via `last_quali_part_only`; this latest part is determined from all prepared session laps, not only valid push laps
  - adds corrected sector times when sector data is present
  - produces corrected quickest driver and team summaries
- `relative_team_pace_rows` builds long-form summary rows for relative team pace plots and includes event metadata when available.
- Team aggregation helpers include:
  - `team_fastest_and_average_rows`
  - `team_best_sector_rows`
  - `quickest_driver_laps`
  - `quickest_team_laps`
- Average mode uses one driver only if a team has one result, or if teammate delta exceeds the configured threshold.
- Best-sector mode preserves each lap's original S1/S2/S3 ratio during correction, then builds a team lap from best corrected sector values across eligible team laps/drivers.

## Pre-Season Data Preparation

`src/wostrategy/tools/pre_season_test/prepare_cumulative_laps_by_day.py`

- Requires `Round`, `SessionName`, `Driver`, `Team`.
- Produces day order labels, driver cumulative lap counts, team cumulative lap counts, and style maps.

`src/wostrategy/tools/pre_season_test/prepare_single_lap_comparison.py`

- Can load single-lap comparison laps directly.
- Drops missing lap times, drivers, teams, and lap starts.
- Removes in-laps and out-laps.
- Applies AM/PM correction to afternoon laps when a correction map is supplied.
- Selects each driver's best adjusted lap and computes delta to quickest.

`src/wostrategy/tools/pre_season_test/prepare_race_sim.py`

- Requires driver/team/lap-time/effective-stint/session metadata.
- Keeps effective stints longer than `min_laps`.
- Excludes effective stint lap 1 from representative windows.
- Applies AM/PM correction and optional session offsets.
- Builds corrected and uncorrected reference contexts.
- Raises if no full representative window from lap 2 through `reference_laps` exists.

## Plotting

`src/wostrategy/plots/style_maps.py`

- Defines hardcoded F1 team colors for common current team names.
- Falls back to matplotlib `tab20`.
- Assigns line styles and bar hatches by driver rank within each team, based on lap counts.

`src/wostrategy/plots/pre_season_test/cumulative_laps_by_day.py`

- Renders driver and team cumulative lap count plots.
- Saves two files when `output_path` is given: `<base>_drivers.<ext>` and `<base>_teams.<ext>`.

`src/wostrategy/plots/pre_season_test/race_sim.py`

- Renders uncorrected and corrected cumulative delta plots against a selected reference pace.
- Saves `<base>_uncorrected.<ext>` and `<base>_corrected.<ext>` when `output_path` has an extension.

`src/wostrategy/plots/pre_season_test/single_lap_comparison.py`

- Renders driver bar chart of delta to quickest adjusted lap.

`src/wostrategy/plots/telemetry.py`

- `plot_front_car_delta_circuit_map` renders two circuit maps for a single telemetry dataframe:
  - Time delta to car ahead
  - Distance to car ahead
- Uses `LineCollection` colored by telemetry columns.

`src/wostrategy/plots/track_development.py`

- Contains push-lap track-development figure rendering.
- `TrackDevelopmentPlotter` is a small OO wrapper around one `TrackEvolutionModel`.
- Functional helpers:
  - `plot_compound_lap_time_fits`
  - `plot_top_driver_summary`
- Compound plots render a 2x2 panel:
  - all dry-compound push laps, no fit
  - SOFT with fit
  - MEDIUM with fit
  - HARD with fit
- Top-driver summary plots render:
  - all push laps, no fit
  - dominant compound, no fit
  - top X drivers, with fit
  - top X drivers on dominant compound, with fit
- Scatter points are colored by team using `F1_TEAM_COLORS`; teammates are distinguished with filled versus hollow circle markers.

`src/wostrategy/plots/quali_performance.py`

- Contains qualifying relative pace plotting.
- `QualiPerformancePlotter` renders all available result-type figures for a target team.
- Functional helpers:
  - `plot_relative_team_pace`
  - `save_relative_team_pace_figures`
  - `result_output_path`
  - `sync_y_limits`
- Race-axis tick labels include short event names when `EventName`/race metadata is available.
- Result types currently include:
  - `fastest`
  - `average`
  - `best_sectors`

## Scripts

Scripts are importable modules under `src/wostrategy/script`.

`pre_season_analysis.py`

- Example end-to-end script.
- Loads 2026 testing sessions, plots cumulative laps, runs two-day race sim, plots single-lap comparison, then shows the figures.

`push_lap_track_development.py`

- CLI for one-session push-lap analysis.
- Loads laps with telemetry gap summaries, with optional lap-time-only fallback when telemetry gaps are unavailable.
- Delegates push-lap flagging/filtering to `analysis.push_laps`.
- Delegates figure rendering to `plots.track_development`.
- Supports `--new-tyre-only` / `--no-new-tyre-only`; default is new tyres only.
- Supports `--track-evolution-fit {linear,exponential}`.
- Default fit is currently `exponential`.
- When `exponential` is selected, the script writes both linear and exponential plot files for comparison.
- Writes paired plots for session time and total lap order, plus `_summary` and `_summary_total_lap` plots when `top_driver_count` is set.

`telemetry_gap_map.py`

- CLI for one lap's front-car time/distance gap map.
- Loads one FastF1 session and driver lap, builds telemetry with `TelemetryDataLoader(skip_lap_errors=False)`, and plots the two telemetry maps.

`tyre_strategy_summary.py`

- CLI for one feature race tyre strategy summary.
- Loads one race session with `load_all_session_laps`, derives tyre segments per driver, and prints Chinese and English two-column tables.
- Orders rows by `SessionResultRank` finishing position when available, falling back to driver name.
- Detects tyre changes from compound changes or `TyreLife` decreasing; this handles fitted old tyres without requiring the new stint age to start at zero.
- Ignores pit-lane pass-throughs where pit markers exist but tyre age continues.
- Writes separate two-column CSV tables to `temp/` by default, with `_chinese.csv` and `_english.csv` suffixes.

`export_telemetry_cache_csv.py`

- Exports one cached telemetry pickle to CSV.
- Requires the cache file to already exist.

`quali_performance_tracker.py`

- CLI for dry formal qualifying (`Q`) performance tracking across a race range.
- Loads lap data plus cached telemetry gap summaries; wet or intermediate tyre usage returns/skips `Wet`.
- Supports `--allow-lap-time-only`, which tries telemetry first and falls back to plain lap-time push-lap selection only when telemetry loading fails or the requested gap summary column is unavailable.
- Prints a final one-line fallback summary, for example `Lap-time-only races: R6` or `Lap-time-only races: none`.
- Delegates qualifying correction and aggregation to `analysis.quali_performance`.
- Delegates relative pace figure rendering/saving to `plots.quali_performance`.
- Filters to clean dry push laps, optionally requiring `FreshTyre` new-tyre laps only; default is new tyres only.
- Uses `SessionLapOrder` and a dominant dry compound (>50% of selected laps) to fit track evolution.
- Optional `top_driver_count` filters only the evolution-fit sample; the fitted model is then applied to all eligible laps, including drivers not used in the fit.
- Supports `--last-quali-part-only` / `--no-last-quali-part-only`; the current script config enables it by default.
- When `last_quali_part_only` is enabled, track evolution is still fitted from all eligible quali push laps across Q1/Q2/Q3, but final fastest/average/best-sector presentation uses only each driver's latest entered qualifying part. If a driver entered Q3 but has no valid Q3 push lap, their Q2 laps are not used as a fallback.
- Supports `--track-evolution-fit {linear,exponential}`.
- The current script config uses `exponential`; selecting it runs and saves both linear and exponential result plots for comparison.
- Corrects lap times relative to the last eligible push lap of the quali and adds track-evolution corrected lap-time columns.
- Tags each selected lap as `Q1`, `Q2`, or `Q3` using FastF1 qualifying splits when available, with a time-order fallback for plain DataFrames.
- Produces separate relative pace plots for:
  - team fastest driver result
  - team average result from both drivers' corrected fastest laps
  - optional best-sector result when `calculate_best_sectors` is enabled
- Writes a `<output-stem>_usage.csv` beside saved plots. It includes plotted rows plus a `SourceLaps` column showing the driver/lap/Q-part or sector composition used for each plotted point.
- Printed summaries include evolution-fit drivers, team driver/lap/Q-part usage, average-mode fallback notes, and best-sector composition.

`long_run_performance.py`

- CLI for race long-run performance analysis over a numeric race range.
- Loads race laps with telemetry gap summaries and filters to consecutive clean-air dry runs.
- Fits driver stint estimates against tyre age, removes obvious stint-estimate outliers, corrects estimates to a shared tyre-life-zero reference lap, and aggregates team performance by compound coverage/usage.
- Can use `--track-evolution-rate-source quali` to run or reuse a linear qualifying track-evolution rate for race stint correction; `race` disables the external quali rate.
- Reference team defaults to the WCC leader after the requested end race when `--reference-team` is omitted.
- Writes per-race CSVs for filtered laps, driver stint fits, driver estimate sanity diagnostics, track-evolution correction stats, team/compound reference estimates, and final team performance.
- Writes driver fit plots and an aggregate long-run performance trend plot.
- Prints diagnostic summaries for fitted driver estimates, excluded estimates, compound coverage, and saved output paths.

## Tests

Current tests:

- `tests/test_telemetry_loader.py`
  - Distance interpolation estimator
  - Telemetry loader metadata handling
  - Multi-session telemetry loading
  - Pickle cache write/reuse
  - Stale telemetry cache refresh when `TimeDeltaToDriverAhead` is missing
  - Per-lap gap summaries
  - Derived car-behind per-lap gap summaries
  - Merge of lap summaries into session laps
  - Merge of FastF1-style session result rank into lap rows
- `tests/test_clean_lap_track_development.py`
  - Push-lap pattern recognition between out-laps and in-laps
  - Slow laps, consecutive push laps, and dirty lap rejection
  - Driver fastest lap threshold behavior
  - Min vs mean clean-gap filters
  - Optional car-behind clean-gap filtering
  - Session lap order by `LapStartTime`
  - Top-driver selection using FastF1 result rank before fastest-lap fallback
  - Top-driver count greater than available drivers
  - Dominant compound detection
- `tests/test_quali_performance_tracker.py`
  - Wet tyre early return
  - Track-evolution lap-time correction against the last push-lap reference
  - Team two-driver quickest corrected lap reporting
  - Dominant compound validation
  - New-tyre filtering
  - Top-driver filtered evolution applied to all drivers
  - Lap-time-only qualifying calculation without telemetry gap columns
  - Requested telemetry gap column availability for fallback
  - Exponential track-evolution fit path
  - Q1/Q2/Q3 qualifying-part propagation
  - Last-qualifying-part-only final performance selection using the driver's latest entered qualifying part
  - Team fastest/average aggregation and teammate-delta fallback
  - Corrected sector ratio preservation
  - Best-sector team result assembly
- `tests/test_session.py`
  - Session race-lap fallback when circuit distance metadata is missing
  - Valid circuit distance handling
- `tests/test_tyre_strategy_summary.py`
  - Drive-through pit-lane pass ignored when tyre age continues
  - Same-compound tyre change detected when tyre age decreases
  - Chinese and English compound strategy formatting
  - Table ordering by finishing rank
  - Separate two-column CSV exports

Observed verification status in this workspace:

- `../Fast-F1/.venv/bin/python -m pytest tests/test_tyre_strategy_summary.py`: passed with `5 passed, 1 warning`.
- `../Fast-F1/.venv/bin/python -m pytest tests/test_clean_lap_track_development.py tests/test_quali_performance_tracker.py tests/test_session.py`: passed with `33 passed, 1 warning`.
- `python3 -m compileall src tests`: passed.
- `python3 -m pytest -q`: failed because Python 3.9 has no pytest installed.
- A neighboring Fast-F1 virtualenv does have the needed test stack.
- `../Fast-F1/.venv/bin/python -m pytest tests/test_clean_lap_track_development.py tests/test_quali_performance_tracker.py`: passed with `27 passed, 1 warning`.
- `python3 -m py_compile` on touched model/analysis/plot/script modules: passed.

## Data And Cache Conventions

- Runtime outputs in examples/scripts generally write to `temp/` relative to the current working directory.
- Telemetry pickle cache defaults to `woStrategy/cache/telemetry`.
- Current cache files in the working tree include sessions like `2026_4_FP1` and `2026_5_SQ`.
- Cache files are pandas pickle files without an extension.
- CSV export is available through `export_telemetry_cache_csv.py`.

## Current Worktree Notes

At review time, `woStrategy` is a git repository on branch `main` with existing modified and untracked files before this document was added. Notable changed/untracked areas include:

- Modified public exports and loaders under `src/wostrategy`
- Untracked telemetry loader and telemetry plot/script modules
- Untracked tests
- Modified `.gitignore`

Future agents should inspect `git status --short` before editing and avoid reverting these existing changes unless explicitly asked.

## High-Risk Areas For Future Changes

- FastF1 dependency coupling: many methods assume FastF1 `Laps` behavior such as `.iterlaps()`, `.pick_drivers()`, `.pick_quicklaps()`, and telemetry column names.
- In-place mutation: `Session.laps`, `add_half_day_label`, `add_session_value_column`, `effective_stint`, and several preparation functions mutate dataframes or session state.
- Print-based logging: loaders and analysis workflows print progress and skip messages instead of using structured logging.
- Error handling: batch loaders swallow session-level exceptions and continue; useful for long runs, but can hide broken assumptions.
- Telemetry cache invalidation: cache filenames include only year, round, and session, not FastF1 version, loader settings, telemetry kwargs, or estimator implementation.
- Result rank availability: top-driver summaries use FastF1 `session.results` rank when loaded into `SessionResultRank`; if result data is absent or stale, the script falls back to fastest uncorrected session lap.
- Time-gap estimator: interpolates using the current car's lap distance/time trace and `DistanceToDriverAhead`; this is an approximation, not a true synced front-car telemetry comparison.
- `model/tyre_degragation.py`: exploratory code and spelling issue remain; import-time side effects have been removed.
- `model/fuel_consumption.py`: placeholder/experimental code should not be treated as a stable API.

## Common Extension Points

Use these existing extension points instead of adding parallel logic:

- Inject `session_factory` into `load_session_laps` or `load_session_laps_with_telemetry_gap_summary` for tests or alternate session sources.
- Pass `enrich_session` to batch loaders when lap-level derived columns are needed before concatenation.
- Add new per-lap telemetry metrics by extending `TelemetryDataLoader` or adding aggregation beside `summarize_lap_gap_metrics`.
- Add new modelling workflows by following the current split:
  - `model/*.py` contains mathematical model classes and pure fitting/correction routines.
  - `analysis/*.py` prepares dataframes, calls model classes, and returns tabular results.
  - `plots/*.py` renders matplotlib figures from analysis results.
  - `script/*.py` handles CLI defaults, argument parsing, session loading, paths, printing, and display/close behavior.

## Quick Start For Future Agents

1. Read `pyproject.toml` and this file.
2. Check `git status --short` inside `woStrategy`.
3. For core data behavior, read `core/session.py`, `core/session_loader.py`, and `core/telemetry_loader.py`.
4. For pre-season workflows, read `tools/two_day_benchmark_race_sim.py` and `tools/pre_season_test/*`.
5. For model changes, start with `model/track_evolution.py`, `model/fuel_consumption.py`, or `model/tyre_degragation.py`.
6. For push-lap work, start with `analysis/push_laps.py`, `plots/track_development.py`, `script/push_lap_track_development.py`, and `tests/test_clean_lap_track_development.py`.
7. For quali performance work, start with `analysis/quali_performance.py`, `plots/quali_performance.py`, `script/quali_performance_tracker.py`, and `tests/test_quali_performance_tracker.py`.
8. For telemetry gap work, start with `core/telemetry_loader.py`, `plots/telemetry.py`, and `script/telemetry_gap_map.py`.
9. Install test tooling before relying on local test execution in this environment, or use the neighboring `../Fast-F1/.venv/bin/python` runner if it is still available.
