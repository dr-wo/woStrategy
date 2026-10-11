# Qualifying traffic phase-validity fix — 9 October 2026

Qualifying now returns unavailable traffic for unsupported circular phase. The
3-second clean-air threshold, field +2% gate, experimental field +7% and own-best
logic, operational classification, canonical fit/aggregation/sign, global push
index, projection/publication/recovery/cutoff contracts are unchanged. No R11/R8
experiment was run. Existing unrelated working changes were preserved.

## Changed path and contract

The root cause crossed `RollingQualiPhysicalTraffic.finalize` in woPlanner and
`_prepare_progress` / `_evaluate_session` in woStrategy. Both inferred full
traversals from integrated distance, without proving the retained origin or
completion. The evaluator could then silently omit an uncertain competitor.

The qualifying adapter now enables `TrafficEvaluationConfig.require_phase_validity`
and identifies evaluated results as `quali-physical-phase-v1`. Existing
non-qualifying adapters keep their previous numerical/version contract; this
change does not certify their legacy reconstruction.

`LapStartTime` belongs to the first row of a traversal, derived causally from the
preceding reducer `lap_counter` publication. `LapEndTime` belongs to its last
row and requires an accepted `normal_timed` completion. The independently
published start/end interval must agree with the completed lap duration within
1 second (the two existing 0.5-second boundary tolerances). These are timing-feed
anchors, not surveyed track positions or a new publication-clock contract.
After the independent-review repair, both anchors come from the original
`PublicationTime` carried by `QualiTrafficEvidence`, never the reducer's advanced
`session_time_s`. Missing provenance, mismatched coordinate kinds, or conflicting
boundary/completion ownership cannot establish phase. Existing completion keys
and operational event timestamps are unchanged.

The shared `validated_complete_lap_span` requires both retained boundary rows,
each within the existing 0.5-second temporal tolerance of its anchor, finite
nonnegative speeds, strictly increasing times, and no integration interval over
the existing 1-second bracketing limit. The whole traversal must use the on-track
route; pit/in-lap or confirmed out-lap rows invalidate it. This does not change
the operational OUT/UNKNOWN classifications or candidate eligibility rules.

Only then can the existing distance-range sanity check and normalization apply.
`_PhaseValid` records the result; unsupported progress is NaN and
`_ProgressQuality` is `phase_unavailable`. A supplied `Distance` does not establish
this contract. No anchored-partial offset reconstruction is introduced.

## Truncation, competitors, and history

- **Right truncation:** no accepted end anchor means no phase. Removing even one
  boundary row from an otherwise complete traversal removes its anchor.
- **Prefix eviction:** the anchor is never copied to the first surviving row.
  Lost-origin spans become unavailable; no zero rebase or inferred offset is used.
- **Complete eviction:** one last-state row per car is retained as an invalid
  span, so a previously on-track car cannot silently vanish. This adds at most
  one stale row per driver beyond the 240-second window.
- **Invalid target:** produces no numeric physical result.
- **Invalid competitor:** remains on the interpolation/state axis. An eligible
  competitor with unsupported phase invalidates traffic at that target sample.
  Explicit off-track evidence retains its existing exclusion semantics.
- **Lap availability:** any such phase uncertainty makes the lap `unknown` and
  clears its numeric physical summaries, even if other matched samples pass the
  ordinary coverage gate. Coverage cannot turn uncertain competition into clean
  evidence. Existing downstream missingness and FP-prior fallback are retained.
- **Circuit history:** only validated complete target traversals enter the
  32-contribution median history, keyed by existing completion identity and still
  subject to its existing 1,000 m minimum. Bootstrap needs no distance estimate;
  subsequent candidates must pass the evaluator's unchanged strict 0.5–1.5 range
  against the **pre-admission** median. Re-finalization cannot duplicate a key.
  Explicit completion invalidation removes its contribution and end anchor;
  the next median uses only the remaining contributions. Invalid competitors do
  not prevent a valid target from contributing length.

Interpolation requires validated endpoints and either the same lap or adjacent
lap identities. Pit/invalid rows remain barriers; interpolation does not bridge
them by deleting them first.

## Saved R16 before/after evidence

The self-contained gzip inputs in `tests/fixtures/quali_traffic_phase` are byte
copies of the completed R&D's field+7 saved inputs. `provenance.json` records
their source paths, SHA-256 hashes, circuit estimates and selected mismatch
rows. Tests reproduce the saved legacy identities and numbers before asserting
the corrected unavailable result. The saved schema has no trustworthy boundary
anchors; none are invented for the replay.

| Target | Elapsed s | Previous ahead | Previous separation | Corrected |
| --- | ---: | --- | --- | --- |
| OCO lap6 | 95.328079 | BOT | 1.772066 m / 0.032884 s | unavailable; no identity/gap |
| OCO lap6 | 37.467831 | LAW | 2.594411 m / 0.040258 s | unavailable; no identity/gap |
| BEA lap6 | 70.342753 | BOR | 1.693484 m / 0.024097 s | unavailable; no identity/gap |
| BEA lap6 | 81.362086 | BOT | 2.789484 m / 0.040656 s | unavailable; no identity/gap |

All four candidate lap4 spans were unfinished, with no evicted prefix, as
established by the saved provenance. Whole-lap means previously 2.443860108 s
(OCO) and 1.640980866 s (BEA) become unavailable. The implementation does not
claim a replacement ahead car or a clean lap.

The actual production R16 recording was also replayed through sequence 6451:
23 pace-admitted samples, zero available traffic samples, zero circuit-history
members, zero supported diagnostic series, FP prior 0 s/push, zero live calls to
the pooled estimator, and unchanged field +2%. The prior four-driver fit at
this checkpoint relied on unsupported traffic. The replay regression now checks
the existing missing-data fallback; the separate canonical tests still verify
supported per-driver fits, weighting, sign and consumption.

Machine-readable results: [saved cases and production checkpoint](assets/quali_traffic_phase_validity_r16.json).

## Complete controls and boundary coverage

The deterministic 1,000 m, three-car, 100 m/s control still selects B at exactly
100 m / 1 s at t=5. All physical sample columns and all lap-summary columns are
equal to the previous evaluator on complete anchored histories. The causal
adapter control retains exactly a 1,000 m circuit estimate and 1 s mean gap.
Removing B's tail/prefix previously selected C at 200 m; the corrected result
is unavailable. Tests also cover single-row boundary loss, missing anchors,
supplied distance without anchors, uncompleted anchored partials, sparse/missing
speed, pit/out/in-lap transitions, full eviction, exact boundary timestamps,
interpolation across lap boundaries, and one uncertain sample despite otherwise
sufficient coverage.

Availability deliberately decreases for these states and for insufficiently
sampled completed laps. Reliable partial-span phase recovery remains outside
this fix. The production R16 checkpoint demonstrates that the decrease can
remove all live diagnostic support; no admission threshold is relaxed to
compensate.

## Initial-fix verification (before independent review)

Commands use `/Users/zhxutong/dr-wo/.venv/bin/python`,
`PYTHONDONTWRITEBYTECODE=1`, `MPLCONFIGDIR=/private/tmp/quali-phase-mpl`, and
`QT_QPA_PLATFORM=offscreen`.

- Focused geometry/adapter regressions plus existing evaluator and qualifying
  physical tests: **45 passed**.
- Existing traffic/reconstruction/cache/pre-quali tests: **64 passed**.
- Qualifying diagnostic, canonical track, service, operational, R16 replay and
  projection, final blockers, historical cutoffs, projection identity/lifecycle/
  version skew, publication and elimination-line tests: **305 passed**.
- R16 checkpoint verification script completed successfully; syntax parsing
  and scoped `git diff --check` passed.

Two old assertions were necessarily revised: the unanchored synthetic adapter
fixture can no longer claim available traffic, and the R16 checkpoint can no
longer require a fit from invalid traffic. No production estimator or cutoff
behavior was changed to satisfy them.

Focused reproduction from woStrategy:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/quali-phase-mpl QT_QPA_PLATFORM=offscreen \
/Users/zhxutong/dr-wo/.venv/bin/python -m pytest -q \
  tests/test_quali_traffic_phase_validity.py tests/test_traffic_evaluator.py \
  ../woPlanner/tests/test_quali_rolling_phase_validity.py \
  ../woPlanner/tests/test_quali_physical_traffic.py
```

The production checkpoint can be reproduced from woPlanner using
`scripts/verify_quali_r16_canonical_track.py --stop-sequence 6451 --output /private/tmp/quali-phase-r16-anchor.json`
with the same Python/environment. The four saved-input cases need only the
self-contained test fixtures, not the full R16 recording or research directories.

## Independent-review blocker repairs

The four review reproductions were recreated through the real
`QualiSessionStore`, then run with the saved pre-repair adapter and the revised
adapter against the same raw evidence. [Machine-readable reproductions and
R13/R14/R16 audit](assets/quali_traffic_phase_review.json) include input/source
hashes and the exact outputs.

| Finding | Pre-repair result | Repaired result |
| --- | --- | --- |
| Original boundary at 0/10 s delivered after the reducer clock advanced to 100/110 s | False validated 100 m / 1 s gap | Anchors retain 0/10 s; retained 100–110 s span is unavailable and cannot seed history |
| Explicit same-boundary completion invalidation | End anchor 110 s and anonymous 1,000 m history member survive | End anchor revoked and keyed contribution removed; later evaluation cannot re-admit the key |
| Pit publication at 107 s, last usable competitor position at 106 s | Stale on-track veto persists after 107 s | Samples before 107 s remain unresolved; at 107 s the known off-track competitor ceases vetoing and valid C/200 m evidence is usable |
| Anchored 2,500 m candidate after a 1,000 m bootstrap | History becomes [1,000, 2,500], median 1,750 m, candidate validates itself | Candidate fails against 1,000 m before mutation; history remains [1,000] and candidate traffic is unavailable |

### Identity, provenance, and state lifecycle

`woData.live.quali_models.QualiTrafficEvidence` and the additive snapshot ledger
expose the existing source `PublicationTime` for observed counter boundaries,
accepted completion ownership, and affirmative timing pit-state publications.
The reducer exports newly explicit invalidation keys from its existing
invalidation decisions. It does not change those decisions, completion keys,
projection ownership, operational timestamps, or playback/recovery conventions.
Best-only clears/changes and unrelated result-register updates do not revoke
physical completions.

`RollingQualiPhysicalTraffic.ingest` consumes this ledger before speed/position
rejection. Completion keys bind retained end anchors and circuit contributions;
source-owner maps reject contradictory ownership, including dependencies on
start anchors. Explicit revocations persist through buffer eviction and cannot
be undone by re-finalization. The 32-member history remains insertion-ordered:
revocation removes a retained member without restoring already evicted members,
and the median is recomputed from the remaining keyed values. A regression with
1,000 m and 1,200 m contributions confirms the new median is 1,200 m after
revoking the former.

Affirmative pit/off-track state is retained separately from integration rows.
The shared evaluator's optional `candidate_states` stream applies eligibility
as a step function at source time, never at arrival time or by interpolating a
state transition. Original Position status/pit-side evidence is also retained
when speed is missing or stale. A current snapshot's structural state is not
backdated onto an older position sample. Repeated later observations are kept
within the retention window so an older delayed pit packet cannot erase later
on-track evidence. Contradictory same-time state evidence remains conservative.
Disappearance without affirmative state continues to veto uncertain traffic.

History admission and circular reconstruction share
`lap_span_within_circuit_range`; bootstrap retains the existing 1,000 m minimum,
and the 1-second continuity and 0.5-second boundary tolerances are unchanged.
Legacy model objects without provenance/event keys remain supported by the
service but cannot establish physical phase.

### Three-event production availability comparison

Both adapters consumed the same full, causal raw replay snapshots with the
production field +2% gate and 3-second clean-air threshold. No field+7 or
own-best experiment was run. Pace sample identities, times and global push
indices were compared explicitly and are identical.

| Event | Messages | Pace-admitted samples | Available traffic before → after | Final diagnostic support |
| --- | ---: | ---: | --- | --- |
| R13 | 19,076 | 87 | 0 → 0 | 0; FP prior |
| R14 | 21,139 | 79 | 0 → 0 | 0; FP prior |
| R16 | 16,496 | 62 | 0 → 0 | 0; FP prior |

All three histories remain empty throughout the audit. There are no per-lap
traffic status/gap changes relative to the prior phase-validity implementation.
The synthetic blocker reproductions intentionally change availability where
the prior lifecycle/provenance handling was wrong. The R14 replay assertion now
checks null aggregate / FP-prior fallback while preserving its operational,
classification, cutoff and unrelated assertions.

The review repair touches the two woData model/reducer files, woPlanner's
`physical_traffic.py`, and woStrategy's shared `traffic.py`. Tests update the
anchored synthetic evidence contract and R14 fallback expectation, and add
`test_quali_traffic_review_blockers.py`. The new read-only reproduction/audit
script is `woPlanner/scripts/verify_quali_traffic_phase_review.py`; from woPlanner:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/quali-phase-mpl QT_QPA_PLATFORM=offscreen \
/Users/zhxutong/dr-wo/.venv/bin/python scripts/verify_quali_traffic_phase_review.py \
  --output /private/tmp/quali-phase-review.json
```

For the saved before/after comparison, add `--baseline-adapter
/private/tmp/quali-phase-blockers-before/physical_traffic.py`. Its hash is recorded
in the audit artifact. This baseline is the uncommitted implementation as it
stood when this repair began, not a Git HEAD or an alternative pace policy.

Residual limits are unchanged: the temporal tolerances do not certify surveyed
physical position, snapshot-held speed remains an approximation, and the
1-second continuity requirement excludes much real qualifying data. Missing or
ambiguous provenance remains unavailable; no partial-phase recovery or threshold
relaxation is introduced. No known instance of the four reported blockers is
left open.

### Final repair verification

- **57 passed:** blocker reproductions, rolling phase validity, qualifying
  physical traffic, legacy-model compatibility, and shared phase-validity tests.
  This includes the final conflicting-publication ownership regression.
- **82 passed:** shared phase validity, existing evaluator/position/circuit
  history/review-remediation traffic tests, R14 clean-lap tests, FP cache
  contracts, and pre-quali tests.
- **125 passed:** woData qualifying tests plus original-publication and recovery
  publication-identity contracts.
- **391 passed:** the full woPlanner qualifying suite collected for the final
  broad run, including R14/R16 replays, diagnostic/canonical fits, operational,
  publication, projection, recovery-related compatibility, historical cutoff,
  elimination-line, planning and GUI tests. The additional ownership edge case
  added during that run passed in the final 57-test focused run above.
- The final three-event audit replayed **56,711 messages** and found zero
  per-lap traffic status/gap changes. All six recorded source hashes match the
  final files; scoped syntax and diff-whitespace checks pass.

All changes remain unstaged and uncommitted. Existing unrelated working-tree
changes were preserved.

## Final lifecycle repairs (2026-10-10)

This pass changes only two production paths: the shared evaluator's
`_apply_candidate_states` / `_interpolate_validated_driver`, and the rolling
adapter's initialization / successful circuit-history admission. Publication
provenance, reducer behavior, phase tolerances, distance sanity, pace admission,
clean-air thresholds, fitting and cutoff policies are unchanged.

### Evaluation-time phase and on-track returns

The candidate-state overlay now checks the interpolation's `valid` mask at each
requested timestamp. That mask already requires temporal support, eligible and
phase-valid endpoints, the existing bracketing/age limits, and compatible lap
identities. The old `phase_known` mask checked only historical endpoint validity
and is removed. An affirmative on-track state cannot create missing progress.

An explicitly off-track car remains excluded from its source timestamp until
the next affirmative on-track state. On return, unsupported current phase
restores uncertainty and suppresses numeric nearest-car output; supported
current progress permits normal numeric participation. No source-time ordering
or state storage code changes. Tests retain the preceding pit interval, delayed
older publications, conservative same-time contradictions and historical
samples before the return.

The real-reducer control's supported overlap still yields 100 m / 1 s. Its
whole-lap assertion now expects unknown: C's telemetry ends at 108 s and B's at
109 s while both remain affirmatively on track. The formerly confident tail
depended on the reported defect. The shared complete anchored control retains
its existing exact numerical comparison.

### Successful admission identity and lifecycle

`_admitted_completion_keys` records the existing stable completion key only
after successful physical contribution. It is independent of insertion-ordered
`_span_history`, whose active contributions remain capped at 32. Eviction no
longer makes a completion new evidence, and repeated finalization neither
reorders nor displaces retained contributions. Never-successfully-admitted
completions remain eligible for a later valid admission.

The existing `_revoked` set remains separate and permanent for the adapter's
lifetime. Revocation removes a retained contribution and its end anchor; an
already evicted contribution remains barred without restoring any other member.
Successful admission memory is retained across qualifying parts, matching the
existing service's circuit-history lifecycle. Completion keys include the part;
a real Q2 completion is admitted normally. `QualiLiveService.reset()` creates a
fresh adapter and clears both admission and revocation memory. Independent
sessions can admit identical key values without sharing identity state.

### Independent reproductions and compatibility

The [lifecycle audit artifact](assets/quali_traffic_lifecycle.json) records
source/input hashes, full ordered history before and after re-finalization,
evaluation-time outputs, disabled-mode comparisons and replay results. Both
production modules were copied before editing; the audit explicitly binds the
saved adapter to the saved shared evaluator, preventing the baseline from
inheriting the repaired interpolation behavior.

| Reproduction | Before repair | After repair |
| --- | --- | --- |
| B traversal ends 104 s; pit 104.25 s; return 105 s without progress | Confident C / 200 m / 2 s at 105 s | Uncertain; numeric geometry suppressed |
| Same sequence during pit at 104.5 s | C / 200 m / 2 s | Identical |
| 34 real reducer completions; 32 active contributions; evicted 1,000 m completion finalized three more times | Median 1,100 → 1,000 m; legitimate contribution displaced | Median stays 1,100 m; history and order unchanged on every repetition |

The phase-validity-disabled comparison checks every sample column and every
lap-summary column against the frozen evaluator. Anchored-return and
partial-return synthetic inputs plus saved R16 OCO/BEA inputs are exactly equal
(21, 21, 151 and 144 target samples respectively).

Reproduce the audit from woPlanner with:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/quali-phase-mpl QT_QPA_PLATFORM=offscreen \
/Users/zhxutong/dr-wo/.venv/bin/python scripts/verify_quali_traffic_lifecycle.py \
  --baseline-directory /private/tmp/quali-phase-lifecycle-before \
  --output ../woStrategy/doc/assets/quali_traffic_lifecycle.json
```

The baseline directory contains this pass's pre-repair `traffic.py` and
`physical_traffic.py`; its hashes are recorded in the artifact. It is a local
verification snapshot, not a committed fixture or a Git HEAD comparison.

### Final lifecycle verification

- **64 passed:** both lifecycle findings, the prior four blockers, rolling
  phase validity, physical traffic, legacy-model compatibility and shared
  phase-validity regressions. Before the repair, the three new tests selected
  for the two independent reproductions all failed on the reported defects.
- **84 passed:** shared traffic/phase validity, position and circuit history,
  prior remediation, R14 clean-lap, FP cache and pre-quali tests.
- **125 passed:** reducer/qualifying, original-publication and recovery
  publication-identity tests.
- **397 passed in 323.78 seconds:** the full woPlanner qualifying suite,
  including R14/R16 replays, operational classification, publication/projection
  compatibility, canonical fitting, historical cutoffs, elimination lines,
  service lifecycle and GUI behavior.
- Four phase-validity-disabled before/after comparisons match exactly.
- Nine scoped Python files pass AST syntax and whitespace checks; scoped Git
  diff checks pass. The final audit's recorded source hashes match the files.

The production field +2% / 3-second clean-air audit compares pace identities,
lap times and global push indices as well as traffic availability:

| Event | Messages | Pace admitted before → after | Usable traffic before → after | Diagnostic source |
| --- | ---: | --- | --- | --- |
| R13 | 19,076 | 87 → 87 | 0 → 0 | FP prior |
| R14 | 21,139 | 79 → 79 | 0 → 0 | FP prior |
| R16 | 16,496 | 62 → 62 | 0 → 0 | FP prior |

There are zero per-lap traffic status/gap changes across all 56,711 messages,
and both versions' circuit histories remain empty throughout these replays.
No continuity or availability recovery is introduced.

Residual limits: phase recovery remains deliberately unavailable for unsupported
telemetry, and held-speed integration retains its existing approximation.
Admission memory uses one key per successful completion for the session rather
than a 32-key cap; session reset releases it. No remaining instance of either
reported lifecycle defect is known from the focused verification.

All lifecycle repairs and verification artifacts remain unstaged and
uncommitted; unrelated working-tree changes were preserved.
