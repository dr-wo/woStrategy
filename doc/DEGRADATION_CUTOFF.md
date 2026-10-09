# Degradation cutoff analysis

`wostrategy.analysis.degradation_cutoff.calculate_degradation_cutoffs` compares
the best exact zero-, one-, two-, and three-stop strategies while varying MEDIUM
tyre degradation. It produces the strategy-envelope sweep, refined primary
0→1, 1→2 and 2→3 crossings, non-monotonic reversal regions, warnings, and the
rules used by the fixed-stop optimiser.

## Function behaviour

The public inputs remain the strategy model and state, pit loss, optional
`StrategyRules`, and optional scan controls. The normal automatic scan is
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

The 2→3 primary cutoff deliberately continues to use brackets from the initial
scan only. This preserves the previous 2→3 definition and avoids changing two
cutoff behaviours in the same revision.

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

`cutoff_table` serializes all three transitions as report-ready rows with the
supplied value and status. `no_positive_degradation_crossing` means the automatic
search reached zero and found no crossing at positive degradation. Equality
exactly at zero does not count as a positive cutoff.

## Tested coverage and boundary comparisons

The search bounds and numerical cutoff calculation are unchanged. In addition
to the existing initial/effective lower bounds, results now include
`initial_scan_max` and `effective_scan_max`. Bounds are MEDIUM absolute degradation
rates in seconds per tyre-age lap (labelled `s/lap`); S/M/H rates scale together
at fixed effective-model ratios. They are not compound performance differences.

Each `cutoff_table` row adds `coverage` and a report-ready `display_text`:

- `coverage.scan_min/max` are the coarse samples used to test that transition.
  For 0→1 and 1→2 they include any lower extension. For 2→3 they exclude the
  lower-extension samples, consistent with its unchanged initial-scan selection.
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
initial scan. It does not guarantee either class was feasible, does not require
either to be globally optimal, and does not establish `cutoff > scan_max`.
Use coverage status and boundary evidence to explain this outcome rather than
collapsing it into `N/A`.

For example, saved R16 effective inputs still scan MEDIUM 0.00975–0.117 s/lap.
Both 2-stop and 3-stop classes are feasible at the upper boundary. The unrestricted
margin is +8.373 s and the rules-compliant margin is approximately +8.386396 s.
The report says the cutoff was not found through 0.117 and that two stops remained
lower-cost at that boundary. It does not claim a mathematical lower bound on the
cutoff. No automatic upper extension is implemented.

The separate [upper-range R&D study](DEGRADATION_CUTOFF_UPPER_RANGE_STUDY.md)
compares historical coverage and future policies. Its proposed adaptive behavior
is not part of production.

The same supplied text is shown below the cutoff figure for unresolved
transitions. Curves and real cutoff lines are unchanged; no fake line is drawn.
Unrestricted 0→1 remains plotted and rules-compliant 0→1 remains explicitly
unavailable. Report bundles retain all additive fields. Older saved artifacts
without those fields remain readable; authoring instructions require explicit
`scan coverage unavailable`, not invented bounds or boundary costs.

The cutoff plot labels predicted MEDIUM degradation and each available 0→1,
1→2 and 2→3 cutoff. Rules-compliant plots explicitly label zero-stop and 0→1 as
unavailable rather than drawing a false crossing. Existing reversal shading and
non-monotonic warnings are retained.

## FP diagnostic reference markers

Race-preparation figures may also show supported FP1, FP2 and FP3 MEDIUM
degradation posterior coordinates as explicitly labelled diagnostic markers.
The production cutoff curves, S:M:H degradation ratios, performance deltas and
crossing calculations are unchanged; the markers merely locate practice evidence
on that existing envelope.

This is a deliberate V1 engineering compromise. The current practice model cannot
reliably separate absolute tyre degradation from fuel/load effects, and compound
performance is not independently identifiable without stronger cross-compound
constraints. FP diagnostics therefore do not become production tyre inputs or an
independent FP strategy prediction. Their values, intervals, support and structural
identifiability are emitted in deterministic report JSON so report generation does
not infer numbers from the figure. Future work may replace this reference-only
presentation once fuel/load estimation and FP identifiability are stronger.

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
- Extension points contain costs for all four stop counts so the emitted sweep
  remains internally consistent.
- Status values are strings for artifact compatibility rather than a dedicated
  enum type.

## To-do

- Consider a versioned status enum if more cutoff-search outcomes are added.
- Profile repeated exact-search evaluations and add safe memoisation if cutoff
  scans become a material runtime cost.
- Decide in a separate change whether 2→3 should gain an independently
  configurable extension policy.
- Add more end-to-end regression fixtures for events with crossings exactly on
  scan points or near zero.

The tyre prediction model is intentionally outside the scope of this analysis;
this feature consumes its output without changing how it is produced.
