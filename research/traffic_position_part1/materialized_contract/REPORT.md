# Materialized FP traffic contract — final remediation

The common FP tyre-evidence boundary now rejects incompatible materialized
frames before preparation, selection or fitting. The cache wrapper no longer
certifies returned frames. Current identity originates at actual traffic
derivation, and the same validator is used by affected enriched-cache readers.
Ready for independent final re-review; no independent approval is claimed.

## Contract and provenance flow

`analysis/traffic_contract.py` defines the shared contract and validator:
`unified-traffic-v4` plus `compatibility_min`. Nonempty inputs require both
DataFrame attributes. If row-level `TrafficEvaluatorVersion` or
`TrafficCombinationMode` is present, its non-null values must agree. Null rows
can represent laps without a gap observation; they do not override existing
traffic availability/clean-selection rules. A conflicting session or row rejects
the input with its context. Validation never mutates a frame or invents metadata.

| Transition | Origin, validation and preservation |
|---|---|
| Raw/session source → telemetry | Existing `TelemetryDataLoader` and version-aware telemetry cache supply observations; these are not themselves certification of enriched FP laps. |
| Telemetry → enriched lap frame | `load_session_laps_with_telemetry_gap_summary` runs the current `summarize_lap_gap_metrics`. Only after successful aggregation/merge does it assign the current contract. Existing source identity is cleared first, and overlapping old gap metrics are replaced. A failed or mixed failed/successful derivation cannot certify the combined frame. |
| Default session loader → enriched cache | `_load_fastf1_session` and the tools wrapper preserve the derivation result. `load_cached_weekend_sessions` validates cached frames and loader-returned frames using the shared validator. Incompatible cached inputs may refresh through derivation; incompatible returned inputs are explicitly rejected before being written. |
| Injected `session_loader` → cache | Supported callbacks must return genuinely derived compatible frames. Valid metadata/content are preserved. Stale, missing or conflicting provenance is never overwritten with current constants. |
| Cache/materialized frame → `calculate_fp_tyre_evidence` | Loader validation occurs before cache writes. The calculator then passes accepted frames to the common analyser. Existing evidence is only replaced after successful analysis. |
| Any materialized frame → `analyse_fp_tyre_evidence_sessions` | Every nonempty frame is checked before normalization or quantitative work, including all sessions in a mixed mapping. No caller can bypass this check merely by avoiding a file reader. |
| Analysis → persisted evidence | The payload and input fingerprint retain validated per-session traffic contracts taken from input attributes. Numerical calculation is unchanged. Saved historical evidence is read as a saved artefact, not recalculated or retroactively certified. |

Empty frames and `None` sessions remain ignored by the analyser. An all-empty
mapping still raises its existing “At least one non-empty” error. Compatible
inputs with no strict clean evidence retain their existing no-evidence error.
Nonempty unversioned frames are explicitly unsupported, even if a caller expects
them eventually to be excluded. No synthetic current provenance is inferred.

## Callers audited

- `calculate_fp_tyre_evidence` → analyser: cache/default/injected loaders all
  encounter the shared validation; the analyser independently rechecks them.
- `run_historical_fp_race_diagnostic` → analyser: its earlier direct-read guard
  now uses the common validator. `refresh_fp_degradation_calibration` delegates
  to it, and its nonempty history-reuse guard uses the common contract constants.
- `historical_fp_backfill.inspect_fp_session_artifact` now uses the same
  validator when inspecting enriched inputs, including contradictory row tags.
- Planner GUI → `OfflineAnalysisService` → `TyreInputService` → calculator:
  no alternate frame path was found. Its injected calculator seam is a separate
  service test seam, not a bypass inside the production analyser.
- Public `wostrategy` and `wostrategy.analysis` exports refer to the same analyser.
  External callers and research/tests supplying materialized frames must meet
  its documented contract. Previously accepted stale R11 fixtures are no longer
  legitimate fresh quantitative inputs.
- `pre_race_analysis` uses the normal cache loader for its standard source path.
  Its explicit expert-CSV legacy model workflow does not call this FP evidence
  analyser or certify CSV data as v4; that distinct model workflow is unchanged,
  as required by the no-tyre-model-change scope. No additional analyser caller
  was found in woWeekend or woData.

The only provenance-free supported source in this affected evidence flow is
the raw/session source *before* traffic derivation. Arbitrary materialized
callback output is not that source boundary. Tests that model successful
derivation now explicitly construct current-contract fixtures; historical data
are not relabelled to make tests pass.

## Before/after evidence

[reproduction.json](reproduction.json) records source hashes and exact outcomes.
The script reads the real local 2026 R11 FP1/FP2/FP3 pickles. Their stored v2
metadata is preserved. V3 and unversioned variants are in-memory copies only.
Baseline analyser/loader source is frozen from commit
`bf5e12cabf61135c3ce438659b3c7af879d793a1`; those two modules were unchanged by
the intervening remediations. Before-behavior writes are confined to temporary
directories. All original source-file hashes remain identical after the run.

| Case | Before | After |
|---|---|---|
| Direct materialized R11 v2 | 8 runs, 55 selected laps, 3 contrasts | Rejected at FP1 contract check before quantitative preparation |
| Same observations with v3 or no attributes | Same quantitative result | Explicitly rejected |
| Injected stale v2/v3 loader | Three new pickles labelled v4; their quantitative contents equal the stale originals; fresh evidence written | No pickle or fresh evidence written; source frames preserved |
| Previously falsely v4-labelled R11 pickle | Accepted by the old wrapper | Rejected because its row-level evaluator remains v2 and contradicts its v4 attributes |
| Current representative fixture | 6 runs, 60 clean laps, 3 contrasts | Every non-identity output field equals baseline exactly, including inventory, selected laps, contrasts and fit/quality output |
| Genuine derivation fixture | Stale source rear metric 999 s | Actual current aggregation produces target-owned 0.5 s, replaces the old metric and writes compatible metadata |
| Failed derivation fixture | Raw/stale source laps available, telemetry unavailable | No current identity and no enriched pickle accepted/written |

The valid-current comparison uses a constructed representative fixture, not
relabelling of R11 observations. Identity fields intentionally change to include
validated traffic provenance. The separate derivation regression runs the real
gap aggregation, stubbing only raw telemetry acquisition; it verifies persisted
metadata **and** quantitative content.

## Tests, preservation and remaining boundary

- Strategy: **134 passed, 3 skipped**, including 15 new materialized/loader
  regressions and all 15 preceding direct-cache/calibration regressions.
- Planner: **99 passed**, including retained-model quarantine, explicit sector,
  real analysis boundary, worker and tyre-input service tests.
- Total: **233 passed, 3 skipped**; the 15 new tests are included, not additional.
  The skips are the existing optional local R11 quantitative fixtures whose
  worktree-relative data path is unavailable. Their helper now also refuses
  stale local caches if provided. Actual R11 rejection is separately reproduced
  against the available data root by the script, not hidden behind these skips.
- Saved parameter/manifest files, failed-calculation evidence, and frozen
  historical calibration version bytes are checked unchanged. No user's source
  caches, saved models or calibration versions were rewritten by this work.
- Diff checks pass. All edits in this pass are confined to the Strategy feature
  worktree. woData and woPlanner source, including A and C, are unchanged.

No known supported path in the **affected FP tyre-evidence and historical
diagnostic flow** can now obtain fresh quantitative output from declared stale,
missing or contradictory traffic provenance. Existing historical artefact reads
remain reads, not fresh fitting or v4 certification. Validation checks declared
provenance and contradictory row tags; it does not recompute supplied compatible
v4 frames. No wider model/cache redesign is claimed.

No remaining blocker was found for this finding's independent final re-review.
The separate, pre-existing production gap-accuracy blocker remains; position
gaps stay diagnostic-only. Traffic thresholds, rear policy, inference code and
default quarantine behavior are unchanged. No merge, push, deployment or Part 2.

## Changed files and reproduction

Production: `analysis/traffic_contract.py` (new), `analysis/fp_tyre_evidence.py`,
`analysis/fp_race_diagnostic.py`, `analysis/historical_fp_backfill.py`,
`core/pre_race_session_data.py`, `core/session_loader.py`.
Tests: `test_fp_materialized_contract.py` (new), `test_fp_traffic_cache_contract.py`,
`test_fp_tyre_evidence.py`, `test_pre_race_session_data.py`.
This folder adds the report, script, JSON, test logs and
[code/test diff stat](diff-stat.txt), measured against the previously staged
remediation. Earlier reports/logs remain historical records.

```sh
ROOT=/Users/zhxutong/dr-wo
STRATEGY="$ROOT/woStrategy/.worktrees/traffic-position-part1"
DATA="$ROOT/woData/.worktrees/traffic-position-part1"
PLANNER="$ROOT/woPlanner/.worktrees/traffic-position-part1"
PYTHON="$ROOT/.venv/bin/python"
export PYTHONPATH="$STRATEGY/src:$DATA/src:$PLANNER/src:$ROOT/woWeekend/src"
export MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen
export MPLCONFIGDIR=/private/tmp/traffic-position-part1/mpl
export WODATA_ROOT="$ROOT/woStrategy/.worktrees/woData"
cd "$STRATEGY"
"$PYTHON" -B research/traffic_position_part1/materialized_contract/reproduce.py \
  --data-root "$ROOT/woData" --output /private/tmp/fp-contract-reproduction.json
"$PYTHON" -B -m pytest -q tests/test_fp_materialized_contract.py \
  tests/test_fp_tyre_evidence.py tests/test_fp_traffic_cache_contract.py \
  tests/test_fp_race_diagnostic.py tests/test_fp_pre_race_report.py \
  tests/test_historical_fp_backfill.py tests/test_pre_race_session_data.py \
  tests/test_telemetry_loader.py tests/test_position_traffic.py \
  tests/test_traffic_evaluator.py tests/test_traffic_review_remediation.py \
  tests/test_r14_clean_lap_traffic.py tests/test_pre_race_analysis.py
"$PYTHON" -B -m pytest -q "$PLANNER/tests/test_analysis_service.py" \
  "$PLANNER/tests/test_analysis_worker.py" "$PLANNER/tests/test_live_analysis_boundary.py" \
  "$PLANNER/tests/test_telemetry_worker.py" "$PLANNER/tests/test_tyre_inputs.py"
```

The worktree-layout override is explained in the preceding remediation README.
The prior unrelated exact floating-point assertion is outside this focused run;
its prerequisite reproduction and assertion remain untouched.
