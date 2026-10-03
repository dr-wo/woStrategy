# Part 1 independent-review remediation

Author recommendation: **ready for another independent review**, with production
position-derived gap decisions still disabled. This does not supersede the
reviewer's FAIL with an independent PASS. No Part 2, threshold tuning, merge,
push or deployment was performed.

## Findings and before/after evidence

The same regression probes were replayed against immutable original Part 1
author commits and the remediated source: **18 failed / 2 passed before;
20 passed after**. The two prior passes were v2-cache refresh checks; v3-cache
checks failed before. These are reproductions of the supplied review findings,
not a claim to possess the independent reviewer's unpublished test harness.
See [before.log](before.log), [after.log](after.log), and [reproduction instructions](README.md).

| Finding | Original Part 1 result | Remediated result |
|---|---|---|
| 1. Legacy rear ownership | Target A lap 10 receives no rear gap from B lap 9; legacy status is `ok`. Missing time/identity also reports `ok`. Shared-unavailable fallback loses the same observation. | A lap 10 receives the recorded 0.5 s rear gap and fails the unchanged 1 s rear clean threshold. Legacy fallback retains it. Missing sample time/identity gives `unknown`, never a clean pass. |
| 2. Cache identity | `unified-traffic-v3` accepts old v3 telemetry sentinel values (999 m), and enriched lap cache avoids recomputation. | `unified-traffic-v4` rejects v2/v3 derived caches; telemetry recomputes to 50 m in the fixture. Enriched laps recompute. A subsequent cache hit equals fresh output and does not reload inputs. |
| 3. Conflicting exact labels | Duplicate A labels 10/11 resolve by input order (helper picks 10; prepared path picks 11 in the reported fixtures). | Conflicting exact ownership is unassigned, including after progress preparation. Reversing input yields equal output. Identical duplicate lap labels remain usable. |
| 4. Grouping | Extra `Year` with requested `(Driver, LapNumber)` raises `KeyError: Year`; prerequisite succeeds. | Same two output keys as prerequisite, no leaked Year requirement. Separate internal sessions remain separate during ownership association. [grouping.json](grouping.json) records all three versions. |
| 5. Malformed position status | `garbage` becomes `False`; shadow excludes the car and reports `FieldComplete=True`. | Malformed/unknown status stays `None`; car is disclosed as unavailable, `FieldComplete=False`, traffic remains `unknown`. Recognised OnTrack/OffTrack and 0/1 retain known booleans. |
| 6. Default quarantine | First update says `waiting-for-clean-laps`, selector promises automatic sector fallback, prepared automatic `ok` rows bypass quarantine. | First and repeated calls say `traffic-unavailable`, explain no new Race update, preserve state, and reject prepared automatic rows. UI states availability and retention explicitly; explicit sector selection remains available. |

## Ownership and availability contract

All reachable direct-rear aggregations in the four repositories were traced:

- Shared `evaluate_direct_fastf1_style` attaches target driver, sample timestamp
  and target lap before `_summarize_direct_samples` groups rear observations.
- `summarize_lap_gap_metrics` reaches legacy
  `_summarize_lap_gaps_from_driver_ahead` → `_summarize_lap_gap_behind` →
  `_summarize_lap_gap_behind_from_driver_ahead`. This now resolves driver-number
  identity within the internal session and uses the same timestamp ownership
  helper. Driver lookup never joins on the follower's lap.
- `_shared_direct_with_legacy_fallback`, the legacy-only return and legacy
  comparison columns consume that corrected summary. Supported legacy callers
  remain present. Physical paths aggregate their original target rows rather
  than reassigning follower observations.

Raw target labels survive until conflict detection. Exact nonconflicting labels
or bounded, same-label bracketing observations support ownership; there is no
extrapolation across lap boundaries. Conflicting number-to-driver identities
also remain unavailable. Session identity is retained internally even when
the caller deliberately requests a coarser output grouping.

Individual unsupported observations are unassigned. A lap with no supported
rear summary is explicitly unavailable, unless the existing combination policy
has a physical rear metric. This changes availability, not the numerical
threshold or ahead/rear selection policy. Older legacy inputs without target
timestamps will consequently lose clean eligibility. Valid observations may
still aggregate alongside discarded ambiguous observations; no new claim of
complete temporal coverage is introduced. Telemetry labels remain labels, not
independently verified finish crossings.

## Cache, consumers and retained state

The version bump uses existing checks in telemetry loading, enriched pre-race
lap loading, historical FP backfill, and race-review derived cache metadata.
No cache files in the user's data store were deleted or proactively rewritten.
Old derived entries are rejected or refreshed when those consumers next run.
Canonical saved model schema and stored models are not invalidated. Existing
models remain historical artefacts; this repair does not retroactively certify
their inputs or relabel them as v4-derived.

Default live recalculation exits before fingerprint/count mutation, settings
writes, fitting or history publication. Tests preserve saved files byte for
byte, loaded model equality, prior evidence count/source/fingerprint, and
in-memory run/parameter history across repeated calls. The UI test prevents
model reload or temporary history publication on unavailable results. The log
distinguishes new eligible laps from retained evidence instead of showing a
misleading decline to zero. No-model calls likewise remain explicit and write
no files. The default gap adapter returns unknown even if it receives prepared
automatic rows. Explicit sector rows must identify the existing sector method;
the pre-existing sector computation, thresholds and fitting path are unchanged.

No position-derived gap is connected to clean-lap selection. Position history
retention (6,000 samples/car, 1,200 s) and shadow reference geometry are unchanged.
The ownership fix groups raw timestamp labels once per call; it adds no persistent
history. Default quarantine skips fitting work. This remediation does not claim
a new historical runtime benchmark.

## Validation

| Suite | Result |
|---|---|
| Recreated review probes, original author sources | 18 failed, 2 passed (expected failures) |
| Same probes, remediated sources | 20 passed |
| woStrategy Part 1 plus cache/ownership suites | 120 passed, 1 pre-existing failure |
| woPlanner telemetry/boundary/service/worker suites | 87 passed |
| woData reducer/history/pending-lap/R15 suites | 122 passed |
| Prerequisite versus author versus repair grouping probe | Expected success / KeyError / success |

There are **329 passing tests and one known failure** across the three broader
runs. Review probes overlap those tests and are not added to that total.
The sole failure is the unchanged exact floating-point comparison in
`test_relative_team_pace_rows_use_sample_paired_relative_baseline_when_available`:
`66.66666666666666 == 200.0 / 3.0`. It was reproduced again using pristine
woStrategy prerequisite source `57495559a860814a96ade41f30dc74df699c9ed6` with the
same runtime. See [prerequisite_float_failure.log](prerequisite_float_failure.log).
Its assertion was not edited. Headless plotting and a documented worktree-layout
environment override are required for this local test setup.

The original historical R12/R13 numeric experiment was not rerun for this focused
remediation. Its archived results remain historical evidence, not current v4
cache results. The targeted suites recheck cold/prefix isolation, boundaries,
delay/batching, position validity and disabled-shadow behavior. No new accuracy
claim is made for passage headways.

## Separate merge recommendations and blockers

| Component | Recommendation | Boundary / remaining blocker |
|---|---|---|
| A: position history + disabled shadow | Ready for re-review, including the malformed-status repair. | Merge only as retained history and diagnostic shadow. Historical neighbour correctness does not validate headway thresholds; partial field and temporal support remain explicit. |
| B: ownership | Ready for re-review together with v4 cache invalidation, grouping and conflict fixes. | Deploying only the ownership logic without the version change is not acceptable. Inputs without target timestamp support remain unavailable. No independent gap-accuracy claim. |
| C: default quarantine | Ready for re-review with service, UI and prepared-row guard together. | Automatic Race updates remain unavailable. This operational limitation is intentional until production gap decisions are validated. Explicit sector mode is a user-selected existing alternative, never automatic fallback. |

The unresolved blocker to **activation**, not to review of a disabled shadow, is
timestamp-aware production headway validation through the real analysis
boundary. The original R12 shadow had 71 threshold disagreements among 904
rear comparisons at 1 s and partial-field limitations. Correct neighbours
alone do not resolve that. The separate baseline floating-point test failure
also prevents describing the broader run as completely green. Independent
merge-gate review must be rerun; no merge approval is claimed here.

## Scope and source identities

All edits are confined to the existing three `codex/traffic-position-part1`
worktrees. Original Part 1 staged changes were recorded in local author-baseline
commits for durable before/after replay; hashes are in
[baseline_commits.json](baseline_commits.json). The earlier required woData and
woPlanner prerequisites remain identified in [../prerequisites.json](../prerequisites.json)
and must accompany the complete Part 1 change. They have not been silently
dropped. No unrelated original-checkout changes are included.

Changed production files: woStrategy `analysis/traffic.py`; woData
`live/reducer.py`; woPlanner `analysis/service.py` and `analysis/gui.py`.
Tests: woStrategy position, traffic evaluator, new review regression suite,
telemetry loader, pre-race data and race-review cache tests; woData telemetry
buffers; woPlanner analysis service and telemetry worker. The original report
and README now mark their historical status. This folder contains reproduction
scripts, source identity, logs and the remediation report.

See [diff-stat.txt](diff-stat.txt) for the exact remediation-only per-repository
`git diff --stat HEAD` at delivery. This is separate from the complete Part 1
diff against each `review_base`.
