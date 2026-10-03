# Part 1 reproduction

Read [REPORT.md](REPORT.md) first. This is a stopped-at-shadow correctness repair,
not validated production time gaps. Production thresholds and tyre fitting are
unchanged. Historical scripts only read the local data root and write their
explicit output directory. No network, acquisition, model persistence or UI.

Three feature worktrees use branch `codex/traffic-position-part1`. Their prerequisite
bases and required existing changes are recorded in `prerequisites.json`.
The woData and woPlanner prerequisite commits must accompany this change; using
only their original HEADs loses the live timing/boundary contract used in this study.

Set the three worktree paths and the existing runtime/data root:

```sh
ROOT=/Users/zhxutong/dr-wo
STRATEGY="$ROOT/woStrategy/.worktrees/traffic-position-part1"
DATA="$ROOT/woData/.worktrees/traffic-position-part1"
PLANNER="$ROOT/woPlanner/.worktrees/traffic-position-part1"
PYTHON="$ROOT/.venv/bin/python"
export PYTHONPATH="$STRATEGY/src:$DATA/src:$PLANNER/src:$ROOT/woWeekend/src"
export MPLBACKEND=Agg
export MPLCONFIGDIR=/private/tmp/traffic-position-repro-mpl
export QT_QPA_PLATFORM=offscreen
OUT=/private/tmp/traffic-position-reproduction
"$PYTHON" -B "$STRATEGY/research/traffic_position_part1/reproduce.py" \
  --data-root "$ROOT/woData" --output "$OUT"
"$PYTHON" -B "$STRATEGY/research/traffic_position_part1/summarize.py" "$OUT"
```

The reproduction loads frozen pre-task traffic code directly from woStrategy
commit `57495559a860814a96ade41f30dc74df699c9ed6` for the before comparison. It never
checks it out or replaces working files. `--skip-legacy` omits that comparison.
The local R12/R13 Race recordings and previous Q Position archives are required;
missing inputs are errors, not permission to fetch or fabricate evidence.

Outputs include full sample CSVs, target-ownership observations, timestamp-aware
independent timing comparisons, source identities and `results.json`/`casebook.json`.
The saved JSON records this run; runtimes will vary. Sample counts and numeric
results should reproduce. Position-reference availability is explicitly a
historical preload assumption, not a claim that the archive file existed live.

Focused checks:

```sh
"$PYTHON" -B -m pytest -q "$DATA/tests/test_live_reducer.py" \
  "$DATA/tests/test_live_telemetry_buffers.py" "$DATA/tests/test_pending_lap_contract.py" \
  "$DATA/tests/test_r15_live_timing.py"
"$PYTHON" -B -m pytest -q "$PLANNER/tests/test_telemetry_worker.py" \
  "$PLANNER/tests/test_live_analysis_boundary.py" "$PLANNER/tests/test_analysis_service.py"
"$PYTHON" -B -m pytest -q "$STRATEGY/tests/test_position_traffic.py" \
  "$STRATEGY/tests/test_traffic_evaluator.py" "$STRATEGY/tests/test_r14_clean_lap_traffic.py" \
  "$STRATEGY/tests/test_race_performance_review.py"
```

The last command has one independently reproduced pre-existing floating-point
exact-equality failure, documented in the report and `baseline_failure.log`.
Use the headless plotting backend above: the native macOS backend aborts in the
noninteractive test process.

Review each repository against its `review_base` in `prerequisites.json`; use
`git diff <review_base>` and `git diff --stat <review_base>`. Prerequisite changes
can separately be reviewed against `original_head`. Nothing is merged or pushed.
