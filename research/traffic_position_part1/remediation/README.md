# Reproduce the review remediation

Use the existing local runtime and all three feature worktrees. Commands below
only run tests/probes; they do not check out or modify another Git revision.
Test caches and models use pytest temporary directories. No network is needed.

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
"$PYTHON" -B research/traffic_position_part1/remediation/probes.py --before
"$PYTHON" -B research/traffic_position_part1/remediation/probes.py
"$PYTHON" -B research/traffic_position_part1/remediation/grouping_probe.py
```

`--before` deliberately exits nonzero (18 failing probes). It loads production
modules from the local immutable author commits in `baseline_commits.json`
using `git show`, with the same new regression assertions. It does not use a
warmed live store or check out baseline files. A normal run expects 20 passes.
The grouping probe additionally executes pristine prerequisite traffic code.

`WODATA_ROOT` here accommodates two existing path-resolution tests that infer
a sibling data checkout from the tested module's worktree location. The path
is used for assertions; it does not create or populate that directory. Test
fixtures supply separate temporary paths for actual writes. Omitting this
override adds two layout-only assertion failures; no production path logic or
assertion was changed to hide them. Use Agg because native macOS plotting
aborts in this headless test process.

Run the broader targeted suites:

```sh
"$PYTHON" -B -m pytest -q \
  "$STRATEGY/tests/test_position_traffic.py" \
  "$STRATEGY/tests/test_traffic_evaluator.py" \
  "$STRATEGY/tests/test_r14_clean_lap_traffic.py" \
  "$STRATEGY/tests/test_race_performance_review.py" \
  "$STRATEGY/tests/test_telemetry_loader.py" \
  "$STRATEGY/tests/test_pre_race_session_data.py" \
  "$STRATEGY/tests/test_traffic_review_remediation.py"
"$PYTHON" -B -m pytest -q \
  "$PLANNER/tests/test_telemetry_worker.py" \
  "$PLANNER/tests/test_live_analysis_boundary.py" \
  "$PLANNER/tests/test_analysis_service.py" \
  "$PLANNER/tests/test_analysis_worker.py"
"$PYTHON" -B -m pytest -q \
  "$DATA/tests/test_live_reducer.py" \
  "$DATA/tests/test_live_telemetry_buffers.py" \
  "$DATA/tests/test_pending_lap_contract.py" \
  "$DATA/tests/test_r15_live_timing.py"
```

Expected: Strategy 120 passed / one known float failure, Planner 87 passed,
Data 122 passed. Logs in this directory record these runs. The aggregate is
329 passing tests, not 349: the 20 review probes overlap the broader suites.

Reproduce the unrelated float failure from pristine prerequisite source,
without modifying any checkout:

```sh
FLOAT_BASE=$(mktemp -d /private/tmp/traffic-float-base.XXXXXX)
git -C "$STRATEGY" archive 57495559a860814a96ade41f30dc74df699c9ed6 \
  src tests/test_race_performance_review.py | tar -x -C "$FLOAT_BASE"
PYTHONPATH="$FLOAT_BASE/src:$DATA/src:$PLANNER/src:$ROOT/woWeekend/src" \
  "$PYTHON" -B -m pytest -q \
  "$FLOAT_BASE/tests/test_race_performance_review.py::test_relative_team_pace_rows_use_sample_paired_relative_baseline_when_available"
```

Expected failure: `66.66666666666666 == 200.0 / 3.0`, unchanged assertion.
The delivery reproduction is saved in `prerequisite_float_failure.log`.

For the remediation-only review, run `git diff HEAD` and `git diff --stat HEAD`
in each feature worktree. HEAD is the author baseline captured before this
remediation, not the prerequisite base. For complete Part 1 review, compare
against each `review_base` in `../prerequisites.json`. Required pre-existing
changes can separately be compared with `original_head`. Nothing is merged,
pushed or deployed. The earlier R12/R13 experiment remains reproducible with
the parent README; its archived numeric evidence was not regenerated here.
