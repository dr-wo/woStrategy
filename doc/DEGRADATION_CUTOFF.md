# Degradation cutoff analysis

Status: implemented; accepted for commit on 2026-10-10. Verification comprises
63 targeted passing tests, all four saved R15/R16 scenarios, visually checked
before/after figures, and an independent review with 24 passing synthetic/plot
tests and no outstanding substantive findings. The saved regression evidence
linked below is retained with the feature.

FP absolute degradation remains structurally confounded. Cutoffs are conditional
on the effective compound-ratio sensitivity model; unavailable classes, explicit
scan limits or computational guards can leave right-side coverage incomplete.

`wostrategy.analysis.degradation_cutoff.calculate_degradation_cutoffs` compares
the best exact fixed-stop strategies through four stops while varying MEDIUM
tyre degradation. It produces the strategy-envelope sweep, refined primary
0→1, 1→2, 2→3 and, when needed, 3→4 crossings, non-monotonic reversal regions, warnings, and the
rules used by the fixed-stop optimiser.

## Function behaviour

The public inputs are the strategy model and state, pit loss, optional
`StrategyRules`, scan controls, `coverage_anchors`, and
`max_upper_extension_blocks` (default 128). The normal automatic scan is
prediction-centred:

1. The initial lower bound is 25% of predicted MEDIUM degradation.
2. The initial upper bound is three times predicted MEDIUM degradation, subject
   to the existing minimum span.
3. Every scan point scales all compound degradations by the same
   `candidate_medium / predicted_medium` factor. This preserves S:M:H ratios.
4. Compound performance deltas are not changed.
5. The existing exact fixed-stop optimiser evaluates the best zero-, one-, two-,
   and three-stop strategy under either unrestricted or supplied sporting rules.
6. Crossings are refined with the existing bisection refinement.

If the initial automatic range has no positive 0→1 or 1→2 bracket, the lower
bound is progressively halved. Each added interval uses the configured coarse
step as a maximum point spacing. The search stops when it brackets the required
positive crossings or evaluates zero. It never evaluates negative degradation.

An explicitly supplied `scan_min` is a hard lower bound. In that case no
automatic extension occurs below the supplied value.

Existing initial-scan 2→3 values are preserved. Lower-extension samples remain
excluded from 2→3 selection and coverage. Anchor-driven upper extension may
resolve a missing 2→3; 3→4 is evaluated lazily only when needed for anchor coverage.

## Result fields and statuses

In addition to the cutoffs and sweep, `DegradationCutoffResult` records:

- `predicted_medium_degradation`: the model value around which the automatic
  range is constructed;
- `initial_scan_min`: the lower bound before any extension;
- `effective_scan_min`: the lowest value actually scanned;
- `primary_0_to_1` and `primary_0_to_1_status`: the unrestricted zero-to-one
  crossing, or `unavailable_rules_compliant` when the two-dry-compound rule
  makes a zero-stop race invalid;
- `primary_1_to_2_status`: one of:
  - `found_initial_scan`;
  - `found_after_lower_extension`;
  - `no_positive_degradation_crossing`;
  - `not_found_above_explicit_scan_min`.

`cutoff_table` serializes permitted transitions as report-ready rows (3→4 only
when sought), with the
supplied value and status. `no_positive_degradation_crossing` means the automatic
search reached zero and found no crossing at positive degradation. Equality
exactly at zero does not count as a positive cutoff.

## Tested coverage and boundary comparisons

The initial scan and its numerical cutoffs are unchanged. In addition
to the existing initial/effective lower bounds, results include
`initial_scan_max` and `effective_scan_max`. Bounds are MEDIUM absolute degradation
rates in seconds per tyre-age lap (labelled `s/lap`); S/M/H rates scale together
at fixed effective-model ratios. They are not compound performance differences.

Each `cutoff_table` row adds `coverage` and a report-ready `display_text`:

- `coverage.scan_min/max` are the coarse samples used to test that transition.
  For 0→1 and 1→2 they include any lower extension. For 2→3 they exclude the
  lower-extension samples, consistent with initial/upper-scan selection; 3→4 also excludes lower-extension samples.
- `comparable_sample_count` counts samples with finite costs for both classes;
  `comparable_interval_count` counts adjacent sample pairs with both classes
  available at both ends. Isolated comparable samples cannot bracket a crossing.
- `coverage.status` distinguishes `both_classes_feasible`,
  `lower_class_unavailable`, `higher_class_unavailable`,
  `both_classes_unavailable`, `insufficient_comparable_samples`, and
  `unavailable_rules_compliant`. Feasibility refers to the upper boundary;
  comparable counts describe coverage within the sampled interval.
- `coverage.upper_boundary` gives the sampled MEDIUM rate, both stop counts,
  feasible flags and costs, `higher_minus_lower_cost` and
  `lower_cost_stop_count`. Positive margin means fewer stops cost less; negative
  means more stops cost less. Zero means equal cost and no unique winner.
  Unavailable costs have null margin/winner. Unknown/non-finite evidence is not
  described as an infeasible or losing strategy.

Existing result statuses and numerical fields are preserved. In particular,
`not_found_in_scan_range` means no sampled pairwise 2→3 bracket was found in the
tested range. It does not guarantee either class was feasible, does not require
either to be globally optimal, and does not establish `cutoff > scan_max`.
Use coverage status and boundary evidence to explain this outcome rather than
collapsing it into `N/A`.

## Evidence-aware upper coverage

FP MEDIUM posterior markers are scalar contextual coverage anchors for the
MEDIUM degradation x-axis. They do not assert equality between the FP compound
posterior vector and the effective-ratio cutoff sensitivity model. FP SOFT/HARD
posteriors are not projected or substituted. This remains conditional strategy
sensitivity, not an FP prediction of the true or probabilistic optimal stop count.

woWeekend retains the existing available-session/measured-support filter with
positive usable laps/runs. Only finite non-negative MEDIUM posterior medians
are passed as anchors. P10/P90 remain diagnostic metadata; they are not additional
coverage points. Unavailable sessions, invalid values and unrelated automatic-model
reference values do not drive extension. The right-most accepted median governs.

1. Keep the initial grid, existing cutoffs and lower-extension behavior. Respect
   `state.max_future_stops`; the cutoff cap is four and does not change normal
   strategy-search configuration.
2. With no valid anchor, or an already-known supported boundary to its right,
   do not extend. An anchor within the original resolved `refine_tolerance` of
   a cutoff is **at the boundary** and also needs no extension. The default
   tolerance remains `min(coarse_step / 8, 0.001)` (0.000625 for R15/R16).
3. Seek the next supported transition in increasing stop-count order. Append
   blocks of five original coarse steps, checking every sample, reusing cached
   costs. Stop sampling a block as soon as a bracket or failure appears.
4. Require a finite lower-to-higher-stop crossing; refine locally at
   `min(refine_tolerance / 16, 1e-7)` and verify a second resolution, flanking
   signs and cost residual continuity. Do not manufacture unstable cutoffs.
5. If the crossing remains left of the anchor, seek the next transition. Activate
   the 4-stop class only when 3→4 is needed, backfilling its costs in accumulated
   initial/upper samples. Compliant R16 therefore never evaluates stop4.
6. Stop at the first right-side boundary, a coincident boundary, an unavailable
   or nonfinite class, insufficient comparable samples, unreliable/unstable
   crossing, event stop limit, or the maximum supported 3→4 boundary.
   **Never search 4→5.** The final appended sample is at most one coarse step
   beyond a found boundary; the plot adds ordinary axis padding.

An explicit `scan_max` is authoritative: neither expansion nor refinement probes
exceed it. Lazy 3→4 can still be found inside an already supplied scan range.
If unresolved, report **coverage incomplete due to explicit scan_max**, the
anchor, last supported boundary, actual bound, and supplied boundary costs/margin.

Finite stop classes do not bound the x-search: parallel costs can fail to cross
forever. `max_upper_extension_blocks` limits total appended blocks (128 by default,
configurable through the calculator API), independently of the anchor or predicted
degradation. Exhaustion is `computational_guard_reached`. This is a computational
budget, not a physical/model-validity ceiling. No fixed 5× rule was adopted.

`anchor_coverage` records accepted anchors/source/session, scalar semantics,
min/max, original equality tolerance, explicit maximum, block budget, extension
history, transitions encountered and refinement costs/margins, transition sought,
left/right/coincident boundaries, stopping reason, final boundary coverage, and
report-ready `display_text`. Initial/final bounds remain in the Phase-1 fields.
New cutoffs found beyond the initial scan use
`found_after_upper_evidence_extension`; previous cutoff statuses stay unchanged.
Stop reasons distinguish `no_valid_anchors`, `right_boundary_found`,
`anchor_at_boundary`, `explicit_scan_max_reached`, `event_max_stops_reached`,
`unsupported_beyond_4_stop`, `fixed_stop_class_unavailable`,
`nonfinite_strategy_cost`, `insufficient_comparable_samples`,
`no_reliable_crossing_in_comparable_samples`, `unstable_crossing`, and
`computational_guard_reached`.

Saved R15 max anchor is 0.0384578775, already below both 1→2 cutoffs: no extension.
Saved R16 max anchor is 0.1916566507. Its initial range remains 0.00975–0.117;
compliant 2→3 near 0.192744 completes coverage, while unrestricted 2→3 near
0.182551 lies left of the anchor and requires 3→4, reproduced at 0.359090874.
Final upper samples are 0.197 compliant and 0.362 unrestricted. Existing
0→1/1→2 values are retained. The [saved replay summary](assets/degradation_cutoff_anchor_acceptance/summary.json)
and adjacent before/after figures record the production verification. The [upper-range study](DEGRADATION_CUTOFF_UPPER_RANGE_STUDY.md)
records the prior numerical investigation and the policy decision.

The same supplied text is shown below the cutoff figure for unresolved
transitions. Curves and real cutoff lines are unchanged; no fake line is drawn.
Unrestricted 0→1 remains plotted and rules-compliant 0→1 remains explicitly
unavailable. Report bundles retain all additive fields. Older saved artifacts
without those fields remain readable; authoring instructions require explicit
`scan coverage unavailable`, not invented bounds or boundary costs.

The cutoff plot labels predicted MEDIUM degradation and each available 0→1,
1→2, 2→3 and, when sought, 3→4 cutoff. Rules-compliant plots explicitly label zero-stop and 0→1 as
unavailable rather than drawing a false crossing. Existing reversal shading and
non-monotonic warnings are retained.

## FP diagnostic reference markers

Race-preparation figures may also show supported FP1, FP2 and FP3 MEDIUM
degradation posterior coordinates as explicitly labelled diagnostic markers.
The production cutoff curves, S:M:H degradation ratios, performance deltas and
initial crossing calculations are unchanged; the markers may extend x-axis
coverage along that model without replacing any tyre input.

This is a deliberate contextual coverage contract. The current practice model cannot
reliably separate absolute tyre degradation from fuel/load effects, and compound
performance is not independently identifiable without stronger cross-compound
constraints. FP diagnostics therefore do not become production tyre inputs or an
independent FP strategy prediction. Their values, intervals, support and structural
identifiability are emitted in deterministic report JSON so report generation does
not infer numbers from the figure. Fuel/load confounding remains a limitation even when coverage is complete.

## Assumptions

- Predicted MEDIUM degradation is non-zero and provides the scale denominator.
  A near-zero prediction is rejected because compound degradation ratios cannot
  then be preserved reliably.
- MEDIUM is the sole scan variable; the other dry compounds move only through
  their fixed ratio to MEDIUM.
- The first refined crossing in ascending degradation is the primary cutoff.
- The same search semantics apply to unrestricted and rules-compliant runs;
  only `StrategyRules` differ.
- Zero is a valid boundary evaluation but not a positive-degradation cutoff.

## Compromises

- Lower-bound extension targets missing 0→1 and 1→2 cutoffs. It does not expand
  the search for 2→3, by design, to preserve existing 2→3 output semantics.
- Extension uses deterministic halving rather than an adaptive model-derived
  lower bound. This is simple, reproducible, and guarantees termination at zero,
  but may evaluate more points than a specialised bracketing method.
- Upper extension points contain all active permitted stop classes; stop4 is
  activated only when required.
- Status values are strings for artifact compatibility rather than a dedicated
  enum type.

## To-do

- Consider a versioned status enum if more cutoff-search outcomes are added.
- Add more end-to-end regression fixtures for events with crossings exactly on
  scan points or near zero.

The tyre prediction model is intentionally outside the scope of this analysis;
this feature consumes its output without changing how it is produced.
