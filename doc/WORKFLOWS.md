# Analysis workflows and visual provenance

[woStrategy overview](../README.md) · [dr-wo project home](https://github.com/dr-wo/dr-wo)

Run the commands below from the `woStrategy` checkout after installing `woData`
and `woStrategy`, with that Python environment active. The examples distinguish saved-input strategy calculations,
FP modelling, Race Retro and weekend orchestration; these are different stages.

## Offline strategy example

This small example uses the saved R16 regression scenario from an actual
race-preparation run. It needs no network access or full timing cache. The input
file records the source run, compound deltas, degradation rates, distance and
pit-loss assumption. It is a conditional historical scenario, not a new forecast.

```bash
python - <<'PY'
import json
from pathlib import Path
from wodata.contracts import StrategySearchState
from wostrategy.algorithm.exact_strategy_search import (
    StrategyModel, StrategyRules, search_best_compound_sequences,
)

saved = json.loads(Path("tests/fixtures/degradation_cutoff_r16.json").read_text())
model = StrategyModel(saved["compound_delta"], saved["degradation"])
state = StrategySearchState(0, saved["total_laps"], "MEDIUM", 0, None, 2)
results = search_best_compound_sequences(
    state, model, saved["pit_loss"], k=5, allow_any_start=True,
    strategy_rules=StrategyRules(minimum_distinct_dry_compounds=2),
)
for result in results:
    print(result.rank, result.formatted_strategy,
          f"+{result.delta_to_best:.3f} s relative to best")
PY
```

This example caps the search at two stops and returns the best pit schedule per
compound sequence. Costs contain tyre terms and pit loss; common race-time terms
cancel between candidates. A ranked result is not a finishing-position forecast.

## Single-event analysis

### Race performance and Retro tyre estimates

```bash
python -m wostrategy.script.race_performance_review \
  --year 2026 --race '[13, 13]' --session R \
  --sample-count 5000 --sampling-strategy latin-hypercube \
  --reference-team Mercedes --use-cached-monte-carlo \
  --plot-output outputs/race_r13.png
```

The small sample count is suitable for a smoke run, not a claim of sampler
convergence. The script reuses a cache only when its metadata matches the requested
settings; otherwise it computes new results. Expect timing/telemetry acquisition
if required data is missing. Cached and new outputs must not be compared as if
all modelling settings were identical.

Event outputs normally live under
`WODATA_ROOT/wostrategy/race_performance_review/schema_v1/year=YYYY/round=R/session=R/`.
They include clean laps, candidate parameters, diagnostics, team baselines and
`tyre_information.csv`, the compact planner-facing compound summary. Multi-event
trackers use a separate range partition. `--output-dir` selects a different
analysis output root; `--plot-output` only changes the plot destination.

Race inference corrects observed lap time for fuel proxy, race-lap track effect,
tyre age and compound delta, then fits a baseline. P10/P90 are weighted parameter
quantiles, not proof that fuel and track effects have been separately identified.
See [the architecture guide](RACE_MC_AND_FP_ARCHITECTURE.md).

### Joint practice model

```bash
python -m wostrategy.script.pre_race_analysis \
  --year 2026 --round 13 --sessions FP1 FP2 FP3 \
  --sample-count 5000 --reference-compound HARD
```

Practice sessions are filtered independently before contributing to a joint fit
with a shared fuel term, session-specific track/tyre terms and per-run intercepts.
The FP model and its HARD reference are distinct from the operational pre-race
prediction adapter, which validates complete compound inputs and normalises
performance to MEDIUM = 0. FP diagnostic estimates are not automatically promoted
to production strategy inputs.

The separate cross-event tyre pipeline consumes historical Race Retro summaries
and processed Pirelli descriptors. Its model families, chronological validation,
readiness rules and explicit preparation commands are documented in
[cross-event tyre prediction](CROSS_EVENT_TYRE_PREDICTION_V1.md).

### Qualifying and targeted diagnostics

The qualifying tracker selects eligible push laps and corrects track evolution;
it does not explicitly estimate fuel or tyre degradation. Its published series
uses the faster driver per team. See its command options:

```bash
python -m wostrategy.script.quali_performance_tracker --help
```

Additional tools cover [local pairwise race pace](LOCAL_PAIRWISE_RACE_PERFORMANCE.md),
[traffic evaluation](traffic_evaluation.md),
[circuit traffic history](circuit_traffic_history.md),
[push–cool evidence](push_cool_sequence_evidence/REPORT.md) and
[telemetry-intensity diagnostics](telemetry_intensity_validation/REPORT.md).
The latter two are bounded research studies, not additional production estimators.

## Desktop and weekend workflows

`woplanner` loads completed sessions and supports strategy what-ifs.
`woplanner-live` opens live/recorded race timing and `woplanner-quali` opens the
qualifying monitor. Select a recording using the application's replay picker;
recording formats and operating details are in the
[Planner guide](https://github.com/dr-wo/woPlanner).

[woWeekend](https://github.com/dr-wo/woWeekend) requires its companion packages,
including [woStanding](https://github.com/dr-wo/woStanding). Once installed, the
race command can generate and validate an operator configuration:

```bash
woweekend-race --generate-config race.example.json
woweekend-race --validate-config race.example.json
```

Review that configuration for the intended event, data location and operational
assumptions before executing a weekend workflow. Race preparation needs a processed,
validated Pirelli preview for the target event; its orchestration prepares/reuses
FP inputs and the model configuration. Post-race processing can replay model
history and assemble a portable report bundle. Numerical artifacts and report
assembly remain separate from authoring narrative conclusions.

Saved runs retain resolved inputs, manifests and output references. Report-only
rendering uses the selected run's artifacts. Historical interruption/traffic
context must be prepared explicitly; report rendering does not fetch it.
Empirical pit-loss observations never overwrite operational pit-loss settings.

The published [R13 report example](https://github.com/dr-wo/woWeekend/tree/main/examples/2026-R13)
retains its original status and limitations. In particular, its green-race
comparison is not an event-aware SC/VSC hindsight optimum.

## README visual sources

All README images have white plotting backgrounds. Historical captures are kept
unchanged; they demonstrate the named historical workflow, not fresh validation
of every current application path.

| README image | Source and interpretation |
| --- | --- |
| `assets/showcase/race_replay_r13.png` | Byte-for-byte copy of woPlanner's `doc/assets/race_replay_cumulative_delta_2026_r13.png`. Historical R13 actual-pace replay. Its combined SC/VSC control predates the current separate controls. |
| `assets/showcase/track_map_r13.png` | Byte-for-byte copy of woPlanner's `doc/assets/track_map_main_r13_popup_large.png`. Historical position view with approximate time-offset mapping. |
| `assets/showcase/strategy_cutoff_r16.png` | Recomputed using `calculate_degradation_cutoffs` and `save_degradation_cutoff_plot` from `tests/fixtures/degradation_cutoff_r16.json`. Two-dry-compound rule, saved FP coverage anchors, 56 laps, 23.7 s pit loss. |
| `assets/showcase/modelled_undercut_r16.png` | The same saved `StrategyModel`, passed to `modelled_undercut_curve` and woWeekend's existing `save_figure` report renderer. Nine ordered compound pairs at model ages 0–56; no observed out-lap interpretation. |
| `assets/tyre_degradation_evolution_2026_r13.png` | Existing historical causal R13 model-update figure, retained unchanged. |
| `assets/race_performance_tracker_2026_r1-r12.png` | Existing season tracker through R12, retained unchanged. It does not claim coverage of later rounds. |

Reproduce the two strategy figures after installing woWeekend and its dependencies:

```bash
python - <<'PY'
import json
from pathlib import Path
from wodata.contracts import StrategySearchState
from wostrategy.algorithm.exact_strategy_search import StrategyModel, StrategyRules
from wostrategy.analysis.degradation_cutoff import calculate_degradation_cutoffs
from wostrategy.plots.degradation_cutoff import save_degradation_cutoff_plot
from wostrategy.analysis.modelled_undercut import modelled_undercut_curve
from woweekend.reports.modelled_undercut import save_figure

saved = json.loads(Path("tests/fixtures/degradation_cutoff_r16.json").read_text())
model = StrategyModel(saved["compound_delta"], saved["degradation"])
state = StrategySearchState(0, saved["total_laps"], "MEDIUM", 0, None, saved["max_stops"])
result = calculate_degradation_cutoffs(
    model=model, state=state, pit_loss=saved["pit_loss"],
    strategy_rules=StrategyRules(minimum_distinct_dry_compounds=2),
    coverage_anchors=saved["coverage_anchors"],
)
out = Path("outputs/readme")
save_degradation_cutoff_plot(
    result, out / "strategy_cutoff_r16.png",
    fp_diagnostic_markers=saved["coverage_anchors"],
)
save_figure(modelled_undercut_curve(model, saved["total_laps"]),
            out / "modelled_undercut_r16.png")
PY
```

The expected positive cutoffs are approximately 0.0935391 and 0.1927441 s/lap.
The scan reaches 0.197 s/lap and reports a supported right boundary. Changing
pit loss, distance, compound ratios or rules changes this conditional result.
