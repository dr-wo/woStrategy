# Degradation cutoff analysis

`wostrategy.analysis.degradation_cutoff.calculate_degradation_cutoffs` compares
the best exact one-, two-, and three-stop strategies while varying MEDIUM tyre
degradation. It produces the strategy-envelope sweep, refined primary 1→2 and
2→3 crossings, non-monotonic reversal regions, warnings, and the rules used by
the fixed-stop optimiser.

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
5. The existing exact fixed-stop optimiser evaluates the best one-, two-, and
   three-stop strategy under either unrestricted or supplied sporting rules.
6. Crossings are refined with the existing bisection refinement.

If the initial automatic range has no positive 1→2 bracket, the lower bound is
progressively halved. Each added interval uses the configured coarse step as a
maximum point spacing. The search stops when it brackets a positive crossing or
evaluates zero. It never evaluates negative degradation.

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
- `primary_1_to_2_status`: one of:
  - `found_initial_scan`;
  - `found_after_lower_extension`;
  - `no_positive_degradation_crossing`;
  - `not_found_above_explicit_scan_min`.

`no_positive_degradation_crossing` means the automatic search reached zero and
found no crossing at positive degradation. Equality exactly at zero does not
count as a positive cutoff.

The cutoff plot labels predicted MEDIUM degradation and each available 1→2 and
2→3 cutoff. Existing reversal shading and non-monotonic warnings are retained.

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

- Lower-bound extension targets only the missing 1→2 cutoff. It does not expand
  the search for 2→3, by design, to preserve existing 2→3 output semantics.
- Extension uses deterministic halving rather than an adaptive model-derived
  lower bound. This is simple, reproducible, and guarantees termination at zero,
  but may evaluate more points than a specialised bracketing method.
- Extension points contain costs for all three stop counts so the emitted sweep
  remains internally consistent, although only the 1→2 bracket drives extension.
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
