# R&D-2 Part 1 — position traffic correctness repair

**Historical author report, superseded for merge readiness:** the independent
review returned FAIL / NOT READY. Read [the remediation report](remediation/REPORT.md)
for corrected ownership, cache identity, unknown-status handling and explicit
live quarantine. Evidence and hashes below describe the original Part 1 run;
they are not a claim that the original patch passed independent review.

**Decision: retain and expose the position shadow, repair target ownership, and
quarantine the defective default live gap reconstruction. Do not activate
position-derived gap decisions.** This is the minimum repair supported by the
experiment. Part 2/rear-filter research has not begun.

## Repaired and validated

- The actual worker analysis store retains up to **6,000 timestamped positions
  per car and 1,200 seconds**, rather than one. The ingress-bounded replay and
  sample-time cutoff both apply. Delayed batches are ordered by embedded time;
  old packets cannot replace a newer latest position. Duplicate conflicting
  positions retain an unknown-status marker. Missing, malformed or timezone-free
  embedded timestamps cannot acquire an invented packet-time position.
- `PositionReference` requires a directed closed loop, frame, scale, source hash,
  observation cutoff and availability declaration. No Race geometry builder or
  event-specific reference is installed in production. The projection reuses
  the nearest-segment mathematics of woData's `PitLaneGeometry`, vectorised for
  circuit history; it does not substitute pit-lane geometry for a circuit.
- The shadow uses exact common timestamps, circular along-track coordinates and
  physical candidates regardless of classification lap. It exposes missing or
  invalid field members, route ambiguity, off-route/unknown-status positions,
  coincident placeholders and missing passage history. A partial field gives
  **candidates among available cars**, not a certified nearest car in the field.
- Shared direct-rear observations now retain `DirectRearTargetDriver`,
  `DirectRearSampleTime` and `DirectRearTargetLapNumber`. Exact target telemetry
  labels win; otherwise both bounded bracketing labels must agree. Ambiguous
  boundaries are unassigned. The follower's lap is never borrowed.
- The default live path returns explicit `unknown` traffic. It cannot fall back
  to the demonstrated invalid zero origins or automatically become sector-based
  clean evidence. The pre-existing, explicitly selected sector mode is unchanged.
  Prepared rows still use the existing worker contract. Numeric thresholds,
  compatibility selection, pace/run requirements and tyre-model code are unchanged.

The helper that slices an already materialized later snapshot deliberately drops
positions: it lacks receipt lineage and cannot prove an old sample was available
at an earlier cutoff. Positions are preserved by the real worker's ingress replay.
This distinction is covered by tests.

## Historical observations

Inputs and code identities are in `results.json`, `source_identity.json` and
`prerequisites.json`. The R12 Race hash remains
`a46c01d2bf6ae87549abedefed1da04b7c003457941cd3b416f25c64472a1a23`.
The frozen legacy traffic code is executed for a fresh before comparison.
R12 reaches the real leader-lap-18 boundary at canonical record **28,404**,
**2026-08-23 13:57:42.431 UTC**, with 99,638 retained positions.

References are NOR's earlier Q traces: R12 publication window 1176.146–1249.189 s,
271 closed-loop vertices, 4212.859 m; R13 window 1101.571–1184.541 s,
316 vertices, 5759.566 m. Their phase origins are arbitrary shared origins, not
calibrated finish lines. Both are **explicit historical-preload assumptions**.
In particular the R12 archive was materialized after the Race; its previous-day
source observations do not prove that a production reference asset existed live.
Without an explicitly available reference the shadow returns unavailable.

### R12 PIA 9–12

| Target lap | Old selected rear mean (s) | Ownership-only direct mean (s) | Position shadow rear mean / min (s) | Supported passage / position samples |
|---|---:|---:|---:|---:|
| 9 | 0.547 | 0.559 | 1.170 / 0.936 | 289 / 296 |
| 10 | 0.800 | 0.793 | 1.192 / 0.711 | 290 / 293 |
| 11 | 0.698 | 0.905 | 1.261 / 0.722 | 277 / 288 |
| 12 | 0.805 | 0.805 | 1.388 / 1.060 | 282 / 290 |

These columns are **different measurements**, not an accuracy comparison with the
old numbers. The old method used integrated speed origins and distance/speed or
lap-trace time interpolation. The shadow uses past spatial passage headways.

The legacy lap-10 rear contributors were LEC 276, RUS 21, HAM 15. Ownership alone
changes that to LEC 268, RUS 24, HAM 15: it fixes bookkeeping, **not geometry**.
Observations moved out/in of PIA laps 9–12 are respectively 3/5, 8/3, 4/8, 5/4.

Position reconstruction selects **RUS in all 1,167 observations** of the target
publication windows, with no LEC rear attribution. Three target coordinate wraps
and 17 target/rear zero-straddling pairs retain that candidate. However BEA and
VER have zero-coordinate placeholders marked `OnTrack` throughout this window.
Every row therefore reports `partial_field`; it does not silently exclude them
and claim complete field coverage. Of 1,167 rear headways, 29 fail the passage
bracket quality checks. No lap is promoted to clean. The sub-second intervals in
laps 10/11 remain visible; the lap-9 diagnostic minimum also warns against treating
these headways as exact timing observations.

### Cold start

Empty position stores consume only packets arriving after the joins below, plus
explicit static number/code identities and the frozen Q reference. No warmed lap,
timing, speed or position state is copied. Comparisons use common timestamps at
that same leader-lap-18 cutoff, including the later portion of the retained history.

| Join UTC | Common frames | Rear-candidate differences | Initially missing headways vs full prefix | Max difference when both available |
|---|---:|---:|---:|---:|
| 13:45:00 | 2,899 | 0 | 4 | 0 s |
| 13:46:14 | 2,614 | 0 | 4 | 0 s |
| 13:48:00 | 2,220 | 0 | 0 | 0 s |

This validates absence of artificial join-dependent phase offsets. It is a
recorded packet cold start, not a newly captured live subscription handshake or
an end-to-end latency measurement. R12 lacks actual receipt-clock history; the
monotonic source-publication frontier is the stated packet-availability proxy.

### Independent timing and non-R12 check

Timing is used only for checking, not aligning geometry. Each comparison uses a
published interval at/before the position timestamp and already present by that
packet's recorded ingress; maximum interval age is 3 seconds. Recorded ranks must
be adjacent, counters equal, and both cars recorded out of the pits. Repeated
position samples can compare to one timing update: these are correlated sample
comparisons, not independent timing measurements or lap-decision error rates.

| Case / side | Comparisons | Absolute disagreement median / p95 / max (s) | Disagreements at existing side threshold |
|---|---:|---:|---:|
| R12 PIA rear | 904 | 0.055 / 0.172 / 0.463 | 71 / 904 at 1 s (37 passage-only, 34 timing-only) |
| R12 PIA ahead | 842 | 0.059 / 0.208 / 0.336 | 0 / 842 at 2.5 s |
| R13 PIA/NOR/RUS rear | 2,879 | 0.085 / 0.456 / 1.591 | 203 / 2,879 at 1 s |
| R13 PIA/NOR/RUS ahead | 1,911 | 0.102 / 0.491 / 2.290 | 97 / 1,911 at 2.5 s |

These are empirical disagreements, **not uncertainty bounds**. A past passage at
the follower's current coordinate and a published timing interval have different
temporal support. Accurate production decisions are not established by small
median disagreement or correct neighbour identity.

The R13 control uses its actual leader-lap-8 boundary, record 15,965,
2026-09-06 13:50:03.277 UTC. It retains 99,697 positions. A genuine recorded order
change is reproduced: at 13:45:43.128, NOR is 8.637 m behind PIA, ranks 6/5; at
13:45:49.008, NOR is 37.736 m ahead, ranks 5/6. Both counters remain 5 and recorded
pit states false. This validates that the geometry can change relative order;
it does not certify every nearest-car attribution in that tightly packed interval.
Full details are in `casebook.json` and the reproducible sample CSVs.

## Measurement semantics and deliberately unavailable cases

A shadow headway is elapsed time since the spatially ahead car last passed the
follower's current projected coordinate. Both interpolation endpoints are past
observations. Reverse/jump brackets, spans over 1 s, segment speeds over 120 m/s,
or passage ages over 10 s are unavailable. These are explicit **shadow quality
controls**, not tuned production traffic thresholds or uncertainty estimates.

Per-lap shadow means are unweighted means of supported samples in the **target's
own (previous completion publication, completion publication] interval**. First
or nonconsecutive boundaries are unassigned. Those publication boundaries are
not established physical finish-crossing times. The direct path separately uses
target telemetry labels; it does not silently claim physical crossing accuracy.

`Position.z` on-track status alone does not establish main-track versus pit route.
The generic shadow accepts explicit per-sample `InPit` context; the production
snapshot currently lacks that timestamped history and reports
`PitContext=position_status_only`. R13 includes pit-car proximity candidates,
which are excluded from the independent timing comparison. Lapped cars remain
physical candidates; timing-interval comparisons deliberately require equal
counters because classification intervals are otherwise not comparable.

No reference, future/unavailable reference, wrong frame, incomplete field,
placeholder, invalid/stale position or insufficient passage history can become
`measured`, `race_informed`, or clean evidence here. The shadow always carries
`TrafficStatus=unknown`. Stale cars are not carried forward to invent a
common-time field. No new uncertainty distribution is synthesized.

**Concrete activation blockers:** independently establish passage/timing support
near the 1 s and 2.5 s boundaries; validate physical target-lap boundaries; supply
causal pit/retirement/route context and a provenance-controlled available reference
asset. The current evidence does not justify connecting these diagnostic gaps to
production selection. There is no proposed rear-filter change.

## Consumers, availability, retention and validation

The affected production consumers are the live telemetry worker and live Race
clean-lap preparation; their existing unknown-traffic gate now withholds these
laps. R12 has 306 completed-lap rows marked unknown, R13 127. Default live Race
updates can therefore remain `waiting-for-clean-laps`, retaining existing model
state. Explicit sector mode, pre-race/manual Strategy Evaluation, saved models,
Strategy Prediction UI and tyre inference are not redesigned or recomputed.
The shared direct ownership fix also affects offline Race/long-run adapters using
that evaluator; their existing numerical estimator and selection policy remain.

At 22 cars the normalized position buffer is capped at 132,000 samples per store
and 20 minutes per car. The worker and causal replay builder can each hold a store.
The map still reads only each deque's last sample, and full histories do not cross
MC IPC. Existing raw ingress history and speed-buffer retention are unchanged;
this is not a bound on total process memory. On this machine, full R12 prefix
reduction took 43.2 s, R13 3.7 s; the three-target shadow took about 7–8 s per case.
The shadow is opt-in diagnostic work and adds no production gap-computation cost.
Source timestamps beyond an analysis cut are excluded; unsupported/corrupt history
can reduce coverage rather than being reconstructed from speed.

Validation: **120 woData tests and 73 woPlanner tests pass**. The broader affected
woStrategy selection has **83 passes and one pre-existing failure**:
`test_relative_team_pace_rows_use_sample_paired_relative_baseline_when_available`
compares 66.66666666666666 to 200/3 by exact equality. The same failure is reproduced
against the original checkout's source; it is unrelated to traffic and left intact.
All new position/ownership tests pass. Tests cover real worker ingress boundaries,
late old packets, pre-cut future samples, delayed/batched retention, timestamp
validity/conflicts, stale/missing/placeholder/off-route positions, wrap, lapped
candidates, explicit pit context, cold start, history insufficiency and target
boundary ownership. Native macOS plotting aborts headlessly; the documented Agg
backend avoids that environment failure.

Changed source/test files (all relative to their respective feature worktrees):

| Repository | Files |
|---|---|
| woData | `src/wodata/live/reducer.py`; `tests/test_live_telemetry_buffers.py` |
| woStrategy | `src/wostrategy/analysis/traffic.py`; new `src/wostrategy/analysis/position_traffic.py`; new `tests/test_position_traffic.py` |
| woPlanner | `src/woplanner/analysis/service.py`; `src/woplanner/live/telemetry.py`; `tests/test_analysis_service.py`; `tests/test_live_analysis_boundary.py`; `tests/test_telemetry_worker.py` |

The woStrategy `research/traffic_position_part1/` directory contains this report,
reproduction scripts, numeric casebook, prerequisite/source identities and test logs.

Changes are isolated in three `codex/traffic-position-part1` worktrees. Required
pre-existing live timing changes are captured separately as woData prerequisite
`ce07e9f` and woPlanner prerequisite `b08ba5e`; original working files were not
modified by this task. Source identities distinguish historical-run bytes from
review bytes: later edits only clarified the shadow docstring and disabled unsafe
position retention in the non-ingress snapshot helper, with its regression test.
See README for reproduction, exact prerequisite files/hashes and diff commands.
No merge, push or deployment. Stop after Part 1.
