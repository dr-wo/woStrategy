# R&D-2 Part 1 — integrated and closed

Closed locally on 2026-10-03. The task owner supplied the independent final gate:
**A PASS / B PASS / C PASS**. This integration preserved that decision and resolved
only the newer-main conflicts. Position-gap activation remains a separate
unresolved task. Rear-filter relaxation belongs to Part 2 and was not implemented.

## Commits and prerequisites

All reviewed worktrees remain at `.worktrees/traffic-position-part1` on
`codex/traffic-position-part1`; their staged changes were committed exactly before
integration. Existing author/prerequisite commits were merged directly, without
squashing, rebasing or manually recreating reviewed code.

| Repository | Main before integration | Reviewed author HEAD | Final reviewed feature commit | Local integration merge |
| --- | --- | --- | --- | --- |
| woData | `5a91b13a10aec968be80f67fac50ea0225cd61a0` | `78287530b284f9d8d602603d324aa27339388d2e` | `671eb26f47d6f3110e859e4d43cd7e00cecef4ca` | `869466cdceda28145af0b5c2441350bac07c6223` |
| woStrategy | `5368b9139c83e8c63d7394cf6a9477d7547d2678` | `bf5e12cabf61135c3ce438659b3c7af879d793a1` | `dac01810fcdfb4ce91a5993e90b67848933b2d5b` | `6d15d69f947a9589f63b4b178088bbb1284b1fa6` |
| woPlanner | `d52f722ab933a6ccc7bccc54ee16d3136d02deff` | `b68b0f80c63667e23624f385cb024091d99c8d91` | `909a7cc28cfc1a1b5d1da3ce25067be5d8181cb2` | `ce38331c79d08abc75af5a8085d9afb5c0dea557` |

- woData review base `ce07e9fa7408e4fd0e792b596e65100702c45497`
  captures prerequisites atop `bd40bbb85dc5381c66232abfc41324d216dc701f`.
  The same prerequisite files were already on main through `5836381`.
- woStrategy review base is `57495559a860814a96ade41f30dc74df699c9ed6`;
  newer main's archive documentation was preserved.
- woPlanner review base `b08ba5e9d0de15c2daa7146538690ba7ffba0e3e`
  captures prerequisites atop `73f42aa9e82a3cd23c87b2f657b929ebcecc8040`.
  The same prerequisite files were already represented by `901c7af`; newer
  compound-offset and track-map changes were preserved.

A maps to the woData history/status merge plus the existing Strategy shadow and
Planner causal-position changes. B maps to `dac0181` and its Strategy merge:
rear ownership and complete v4 enforcement land together on main. C maps to
`909a7cc` and its Planner merge: default quarantine, prepared-row guard, retained
state and UI disclosure land together. Earlier author commits span boundaries;
they remain explicit historical ancestors, not separate partially deployed states.

Post-merge closure commits are woData `8c2ba265ca429562dc1737fb11449d3e8112bef7`
(documentation) and woPlanner `b303a11eb9b14bd26db72549e3bea70d7723668f`
(documentation and a new test-fixture correction). The woStrategy commit adding
this file completes the canonical documentation and evidence index.

## Conflict resolution

`woPlanner/src/woplanner/live/telemetry.py`: keep one canonical woData retention
implementation for track-map and causal replay stores: **at most 6,000 samples
per driver, and only the latest 1,200 embedded sample-time seconds**. The newer
map needs a recent complete non-pit lap, not unlimited samples within the window.
Time-only retention does not impose a count bound for dense ingress, so removing
the reviewed cap was unnecessary. For 22 identities, each store retains at most
132,000 position samples. At 5 Hz the count cap spans about 20 minutes; at 10 Hz
about 10 minutes. This is a per-driver/store bound, not a global bound on replay
message history or raw archives, and does not assume a fixed total identity count.

The new cap-exhaustion test feeds 7,001 samples at 10 Hz, verifies exactly 6,000
remain, and extracts a recent complete 100-second/1,001-point track reference.
It then verifies delayed/duplicate packets cannot rewind the latest position or
grow the buffer, and tests inclusive 1,200-second eviction. The existing real
R13 replay also passes: 22 official drivers, no ghost markers, and the RUS lap-2
reference of approximately 86.651 seconds with at least 300 points.

Causal analysis retains positions only after replay through the immutable ingress
cut and filtering embedded sample times. A new backward-cut test warms and evicts
history, then verifies that rebuilding the earlier cut excludes both a pre-cut
future sample and a late-arriving old sample. Existing worker cutoff and delayed
batch tests remain. Later materialized snapshots still cannot claim arrival
causality. Raw position samples remain absent from MC inputs; map time-offset
visuals do not activate position-traffic gap decisions.

`README.md`: preserve main's committed/provisional canonical timing distinction
and all track-map documentation, adding the reviewed quarantine/retention contract.
`tests/test_live_analysis_boundary.py`: preserve the canonical/display assertions,
explicitly select sector mode for the clean-lap scenario, and retain the reviewed
materialized-snapshot causal-position assertion. No assertion was dropped.

## Merged-main validation

Tests imported the actual main checkout sources, not feature-worktree modules:
[source paths](integration/source_paths.json). Exact test file lists are in
[test_suites.json](integration/test_suites.json).

| Suite | Result | Evidence |
| --- | --- | --- |
| woData retention, reducer, pending/canonical timing, replay | 125 passed | [data.log](integration/data.log) |
| woStrategy ownership/conflicts/grouping, shadow, v4 caches/materialized frames, FP evidence/diagnostics/calibration/backfill consumers, race review | 189 passed, 3 skipped, 1 known failure | [strategy.log](integration/strategy.log) |
| woPlanner causal retention, worker, service/UI quarantine, sector selection, track map and real replay, compound-offset regression | 221 passed | [planner.log](integration/planner.log) |

**535 passed, 3 skipped, one unchanged pre-existing failure.** The new retention
test initially had an incomplete synthetic lap fixture; it was corrected without
changing production code or dropping assertions, then the entire Planner suite
passed. The three local R11 FP tests skip because the existing enriched cache
requires genuine traffic recomputation; it was not refreshed or relabelled.
See [skip reasons](integration/fp_cache_skips.log). Deterministic v2/v3/unversioned
and contradictory-contract rejection tests passed independently of that cache.

The exact float assertion `66.66666666666666 == 200.0 / 3.0` also fails on pristine
Strategy base `57495559a860814a96ade41f30dc74df699c9ed6`, extracted to a temporary
directory without changing a checkout. [Base reproduction](integration/prerequisite_float.log).
The assertion and production arithmetic were left unchanged.

Reproduce the merged suites using the existing adjacent repositories and runtime:

```sh
ROOT=/Users/zhxutong/dr-wo
export PYTHONPATH="$ROOT/woStrategy/src:$ROOT/woData/src:$ROOT/woPlanner/src:$ROOT/woWeekend/src"
export WODATA_ROOT="$ROOT/woData"
export MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen
export MPLCONFIGDIR=/private/tmp/traffic-part1-merged-mpl
"$ROOT/.venv/bin/python" -B - <<'PYTEST'
import json, subprocess
from pathlib import Path
root = Path('/Users/zhxutong/dr-wo')
suites = json.loads((root / 'woStrategy/research/traffic_position_part1/integration/test_suites.json').read_text())
for repo, tests in suites.items():
    result = subprocess.run(
        [str(root / '.venv/bin/python'), '-B', '-m', 'pytest', '-q', '-rs', '--tb=short',
         *[f'tests/{name}.py' for name in tests]], cwd=root / repo,
    )
    print(repo, 'exit status:', result.returncode)
PYTEST
```

Source comparisons confirm that Strategy runtime code and the woData reducer
match the final reviewed feature content. Planner retains newer-main track-map
and compound-offset functionality alongside the reviewed quarantine. Production
traffic thresholds and rear-filter policy are unchanged; position-derived gaps
cannot enter production clean-lap selection. Fresh FP evidence rejects stale
traffic contracts; saved models/calibrations remain historical and unchanged.

## Documentation and preservation

Canonical documentation updated:

- woData: `README.md`.
- woPlanner: `README.md`, `doc/FP_TYRE_LIVE_INTEGRATION.md`.
- woStrategy: `README.md`, `doc/traffic_evaluation.md`,
  `doc/RACE_MC_AND_FP_ARCHITECTURE.md`, `doc/CODE_STATUS_QUO.md`.
- Part 1 research: `README.md`, the current-status preface of `REPORT.md`, and
  this closure. Earlier reports/results remain historical records; they were
  not regenerated or retrospectively relabelled.

Current docs consistently describe disabled zero-origin live reconstruction,
`traffic-unavailable`, retained Race model/history, explicit sector selection,
diagnostic-only position traffic, fresh v4 FP evidence, and historical saved
models. No canonical doc still prescribes automatic sector fallback.

[Preservation audit](integration/preservation.json) verified 3,689 pre-existing
woData files and 1,157 woStrategy files: hashes are unchanged except for the
intentional traffic-only README update; its unrelated BSF patch remains identical
hunk-for-hunk. No unrelated change was staged, stashed, cleaned, reverted or
committed. woData remains dirty only with its original preview/fetch/untracked
work; woStrategy retains its original BSF README/untracked research/code work;
woPlanner has no tracked changes and retains its pre-existing untracked worktrees.
Reviewed feature worktrees are clean. Diff hygiene checks pass.

Only the requested local integration and closure occurred. No push, remote merge,
deployment, cache refresh, position-gap activation or Part 2 work was performed.
