# Part 1 final narrow remediation

Both remaining findings are addressed. Ready for independent final re-review;
this is an author recommendation, not an independent approval. No activation,
threshold, rear-policy, tyre-inference, merge, push, deployment or Part 2 changes.
Existing isolated worktrees and previously staged remediation are preserved.

## Exact files changed in this pass

- woStrategy `src/wostrategy/analysis/fp_race_diagnostic.py`
- woStrategy `tests/test_fp_traffic_cache_contract.py` (new)
- woPlanner `README.md`, only its live traffic section and earlier live-update description
- woPlanner `doc/FP_TYRE_LIVE_INTEGRATION.md`, only Traffic assumptions
- woStrategy `research/traffic_position_part1/final_narrow/REPORT.md` (this file),
  `before.log`, `strategy.log`, and `planner.log` (new reproduction records)

No woData files were changed in this pass. The earlier remediation report/logs
remain records of that earlier review stage rather than being relabelled.

## Cache boundary repair

`run_historical_fp_race_diagnostic` now checks every enriched lap pickle it
selects before passing it into fresh FP analysis. The required identity is
`unified-traffic-v4` plus `compatibility_min`, using the same constants and
metadata contract as the normal version-aware loader. Missing metadata, v2/v3,
and another combination mode raise a descriptive `ValueError` containing the
path and expected/actual identity. The command does not download, relabel or
rewrite the incompatible input. Use the existing version-aware loader to
refresh it explicitly before retrying. Rejection occurs before diagnostic
outputs are written; saved models are unchanged.

`refresh_fp_degradation_calibration` delegates to that checked path when
recomputing FP history. It can also reuse a prior diagnostic history CSV when
the recomputation prerequisites are absent. New history rows therefore carry
the same version/mode contract, and reused nonempty history must satisfy it.
Unversioned or incompatible history is rejected before calibration writes.
Empty history contains no quantitative evidence and retains its existing empty
calibration behavior. Frozen historical calibration versions are not rewritten.

The diagnostic records enriched-lap file hashes and their contract in its
summary/fingerprint. A separate statement explicitly identifies saved FP
parameters as historical estimates: they are not refitted or certified under
v4 by this diagnostic. Existing model files and previously saved artefacts are
neither invalidated nor retroactively relabelled.

## Before/after reproduction

Tests use a temporary event with persisted FP parameters and a physical lap
pickle. Source acquisition/programme selection is isolated with a deterministic
selected-run fixture; run fitting, model reads, diagnostic output and calibration
reuse are exercised. This is an entry-point regression, not a new tyre study or
a claim to have recomputed the real R11 caches. No event-specific rule was added.

| Probe | Before | After |
|---|---|---|
| v2/v3 enriched cache, both entry points | Accepted; expected rejection did not occur | Explicit rejection before the analysis callback; all files unchanged |
| Missing version/mode or incompatible mode | Accepted | Same rejection contract |
| Current v4 cache | Calculation ran without the new contract/provenance disclosure | Accepted; initial and repeat diagnostic files identical; subsequent normal loader hits the same cache |
| Old/unversioned calibration CSV reused without acquisition prerequisites | Accepted | Explicit rejection, preserving prior outputs and saved artefacts |
| Saved model preservation | Not used as a version-freshness claim | Parameters/manifest bytes unchanged across accepted and rejected calls |

The same 15 new tests were replayed against the pre-fix module from immutable
commit `bf5e12cabf61135c3ce438659b3c7af879d793a1` (that file was unchanged by the
previous remediation): **15 failed before, all 15 passed after**. Ten before
failures demonstrate missing input rejection, three demonstrate history reuse
without the contract, and two demonstrate missing provenance disclosure for
otherwise accepted v4 calculations. See [before.log](before.log).

## Validation and operational documentation

- Strategy: **117 passed**, including all 15 new cases, historical FP diagnostic,
  FP report/calibration, historical backfill, normal enriched/telemetry cache
  loaders, position shadow, traffic evaluator, ownership and R14 traffic tests.
- Planner: **87 passed**, covering telemetry worker, real analysis boundary,
  service/UI quarantine, retained state, explicit sector mode and analysis worker.
- **204 passed total**, with no failures or skips in these runs. The 15 new cases
  are included, not added again. Logs: [strategy.log](strategy.log),
  [planner.log](planner.log). The unrelated race-review float assertion is outside
  this focused run; its earlier prerequisite reproduction remains unchanged.

Both operational documents now state the actual selector/status, repeated
`traffic-unavailable`, no automatic sector fallback, retained model/history,
explicit sector selection and inactive diagnostic position gaps. Planner
runtime code was not edited in this pass. Its existing tests reconfirm these
boundaries. No remaining blocker was found for independent final re-review of
these two fixes. The previously documented gap-accuracy blocker still prohibits
position-gap activation; stale input availability requires an explicit refresh.

## Durable reproduction

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
"$PYTHON" -B -m pytest -q tests/test_fp_traffic_cache_contract.py \
  tests/test_fp_race_diagnostic.py tests/test_fp_pre_race_report.py \
  tests/test_historical_fp_backfill.py tests/test_pre_race_session_data.py \
  tests/test_telemetry_loader.py tests/test_position_traffic.py \
  tests/test_traffic_evaluator.py tests/test_traffic_review_remediation.py \
  tests/test_r14_clean_lap_traffic.py
"$PYTHON" -B -m pytest -q "$PLANNER/tests/test_telemetry_worker.py" \
  "$PLANNER/tests/test_live_analysis_boundary.py" \
  "$PLANNER/tests/test_analysis_service.py" "$PLANNER/tests/test_analysis_worker.py"
```

The layout override is explained in the preceding remediation README. To repeat
the expected failing before run without checking out or modifying baseline files:

```sh
"$PYTHON" -B - <<'PY'
import subprocess
import pytest
from wostrategy.analysis import fp_race_diagnostic as module
source = subprocess.check_output([
    'git', 'show',
    'bf5e12cabf61135c3ce438659b3c7af879d793a1:src/wostrategy/analysis/fp_race_diagnostic.py',
], text=True)
exec(compile(source, 'fp_race_diagnostic.py (before final remediation)', 'exec'), module.__dict__)
raise SystemExit(pytest.main(['tests/test_fp_traffic_cache_contract.py', '-q', '--tb=short']))
PY
```
