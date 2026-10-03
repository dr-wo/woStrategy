# Descriptive circuit traffic history

This report-only feature has three owners: woStrategy calculates timing-corroborated interactions and retrospective references; woData owns durable source/result paths; woWeekend explicitly prepares history and assembles the pre-race context/figures. It supplies no simulator probabilities or strategy parameters.

## Prepare, then render

From an environment with the sibling packages installed:

```sh
woweekend-traffic-history --circuit Monza --target-season 2026 --data-root /path/to/woData
```

The circuit argument is an exact historical calendar **Location**, rather than a country/event-name wildcard. Main Race sessions only. It considers the eight calendar years immediately preceding the target season. Missing calendar/session/analysis data stays in coverage with its reason; the window is never extended. The first preparation loads only that circuit through the standard laps loader. Prepared sessions and a dedicated FastF1 cache live under woData's `wostrategy/circuit_traffic_sources/schema_v1/` namespace. Existing unrelated source caches are not changed.

A valid saved calculation is reused. `--refresh` explicitly recalculates saved session exports and retries years without prepared exports; it is not an upstream force-download flag. If an upstream correction is required, prepare a separately identified source dataset rather than silently treating an old export as refreshed. `--seed-history` can bootstrap from the completed research summary's hash-validated inputs, without changing them.

`--review-dir /path/to/new-review` renders just the historical pre-race section and the existing report bundle for review. Use a new review directory because report bundles follow woWeekend's existing immutable bundle convention. Normal race preparation and report regeneration consume saved values only. Missing/stale/corrupt history is reported as unavailable, never fitted by report rendering. Preparing history is optional and does not alter pre-race operational gates.

## Records and groups

The two disjoint subsets are calendar years 2022–2025 **inside** the current eight-year window, and all other years inside it. ALL contains the complete window and is plotted in a separate panel, never stacked on its own subsets. Each group and each calendar year retains the same five distributions, usable N, signed interaction-weighted mean, exclusions, BLUE support, extrapolation count, and common histogram edges/counts. Null outer edges mean negative/positive infinity. Empty distributions have N=0, zero frequency counts, and null statistics.

Estimated advantage is the passed-car reference minus passing-car reference at actual tyre states, positive when the passing car is estimated faster. The **observed successful-pass minimum and median advantages** come directly from eligible timing-corroborated successful-pass records. P50 uses the arithmetic mean of the middle two records for even N. All minimum-tie interaction IDs remain traceable. These are neither a minimum physical requirement nor a 50% passing-success threshold.

## Preserved calculation

`_circuit_traffic_core.py` extracts the completed `circuit_traffic_summary` timing/reference helpers into package code, without runtime imports from research. The original calculation and its prior gates/artifacts are unchanged. It retains timing/position/sector/next-boundary corroboration, green/dry eligibility, pit/restart/assigned-incident handling, the whole-lap local window, and the exclusion of compound interactions from both residuals and advantage. Lapping matches session time across different lap counters. Assigned BLUE messages evidence the lapped car's flag context; they do not uniquely identify its lapper.

The reference retains its team-pooled ridge coefficients/penalties and four-boundary timing screen. It excludes the union of the target laps plus two maximum target-lap durations on either side. Fitting/prediction age is retained-record `StintLapNumber-1`; `TyreLife-1` is retained separately. Nonzero offsets remain unsupported, including used sets and retained-count discrepancies. No older compound names are remapped into SOFT/MEDIUM/HARD to increase historical support. Age extrapolation stays labelled, missing references stay null, and no new quality gate is introduced. References may use later event evidence, are team-level estimates, and are not measured true free-air pace or causal traffic losses.

## Storage and cache identity

woData stores content-addressed `result.json` artifacts under:

```text
wostrategy/circuit_traffic_history/schema_v1/circuit=<normalized-key>/target_season=<year>/artifacts/<content-hash>/result.json
```

An atomic `latest.json` pointer selects the current artifact. The result contains every pairing (including exclusions and missing values), source screens, grouped/yearly summaries, histogram edges/counts, coverage, minimum-source IDs, calculation/reference rules, code hashes, input-file hashes and library versions. Reading verifies content, schema, circuit, window and the current calculation identity. Explicit preparation also checks changed locally present input files before reusing analysis. Rendering uses the durable artifact and does not require refitting or raw telemetry.

`woweekend.reports.historical_traffic` uses this same persisted result for text, numeric context and five three-panel figures. Optional history follows the existing report-only context convention: the component loads successfully even when its nested data status is partial/unavailable. Strategy prerequisites and existing cutoff gates are unchanged.

## Acceptance evidence

The executable Monza target-2026 acceptance evidence is saved under `woStrategy/research/traffic_history_integration/`. All eight years 2018–2025 loaded. The 811 overlapping 2021–2025 pairing rows match the frozen result across 143 fields; reference predictions are identical and maximum residual serialization differences are below 7e-16 seconds. Focused tests cover partitioning, target-year exclusion, unavailable years, empty groups, negative/tied minima, exact medians, pooled record statistics, storage integrity, cache-only report rendering, and the existing race-preparation workflow.
