# Race Monte Carlo and FP Tyre Architecture

Last reviewed: 2026-08-22

This document records the assumptions and compromises behind the current Race
Monte Carlo, live-retro model, experimental Race-specific model, and production
FP tyre-evidence API. Read it before changing dimensions, priors, likelihoods,
reference compounds, or live/replay behaviour.

## Production Race path

The production live path uses `run_live_retro_model` in
`model/live_retro_performance.py`. It prepares the available clean Race laps and
runs `MonteCarloRacePerformanceAlgorithm` with the same parameterization and
weight semantics as the retrospective race-performance review.

The active coordinates depend on observed Race data:

```text
Fuel rate
Track rate
one base degradation per observed compound
one compound delta per observed non-reference compound
one team-compound degradation variation per observed team/compound pair
profiled driver or team baseline intercepts
```

FP1/FP2/FP3 likelihood rows and FP-local nuisance parameters are not part of this
Race likelihood. Pre-race values enter Planner only as an uninformed baseline;
Race-informed values remain controlled by the existing Race evidence path.

The model-quality payload intentionally separates:

- numerical quality: ESS, ESS fraction, max/top-five weight, evaluator timing;
- information: clean Race laps/runs, observed compounds, tyre-age support and
  posterior P10-P90 widths.

Never interpret high ESS alone as high confidence.

## Accelerated evaluator

`MonteCarloRacePerformanceConfig.evaluator_backend` defaults to `accelerated`.
`legacy` remains available as a regression oracle. The accelerated backend:

- draws the same unit-cube dimensions in the same order;
- applies the same bounds and hot-track ordering transforms;
- profiles the same baseline groups;
- calculates the same RMSE and weight function;
- materializes the same public result tables;
- evaluates candidates in NumPy chunks controlled by `candidate_chunk_size`
  (default 2000) to bound temporary memory.

Acceleration is an implementation change, not a modelling change. Any change to
sampling order, ordering transforms, group profiling, references, or weight
normalization must be checked against the legacy backend with fixed seeds. Timing
data in `last_timings` is operational telemetry and does not affect inference.

## Scheduled race length

The fuel proxy requires an `Lmax`. woPlanner resolves it before calling this
package in this order:

```text
live LapCount.TotalLaps
configured/manual override
current leader lap fallback
```

The fallback is only for incomplete input and becomes progressively less useful
early in a Race. Never replace scheduled distance with the maximum currently
completed `LapNumber` when `TotalLaps` is available.

## Identifiability

`model/correction_identifiability.py` supplies component sensitivities and
profiled design-rank diagnostics. Rank/null-space results describe the likelihood
geometry at the supplied observations; they are not sampler diagnostics.

Fuel, track evolution and compound degradation can be strongly correlated. In
run-intercepted FP fits there is an exact common-slope gauge:

```text
Fuel  -> Fuel + t
Deg_H -> Deg_H + t
Deg_M -> Deg_M + t
Deg_S -> Deg_S + t
```

Absolute FP degradation is therefore not uniquely learned. Gauge-invariant
contrasts such as `Deg_H - Deg_M` are the defensible outputs. Changing a reference
compound only changes coordinates when the observation graph is connected; it
cannot manufacture information for an unused compound.

## Experimental Race-specific model

`model/race_specific_performance.py` is retained for architecture diagnostics and
is not the default production live model. It uses six Race-state coordinates:

```text
net lap slope
Deg_HARD, Deg_MEDIUM, Deg_SOFT
Perf_HARD - Perf_MEDIUM
Perf_SOFT - Perf_MEDIUM
```

Its transfer priors are split-normal approximations built from marginal pre-race
P10/median/P90 values. The current pre-race artifact does not contain the full
joint covariance, so this construction cannot preserve pre-race correlations.
The degradation samples are sorted to enforce physical order; that operation
introduces dependence and can move the marginal distribution of an unseen
compound. These are explicit prototype compromises and reasons not to silently
switch production from `live-retro` to `race-specific`.

The `per_driver` and `global_driver` baseline switches span the same fitted
subspace. `per_run` removes cross-run compound offsets and therefore cannot learn
cross-stint performance contrasts. Compound performance is only identifiable when
the driver-compound observation graph contains connectors.

## Production FP tyre evidence

`analysis/fp_tyre_evidence.py` is the stable production library API. It has a
strict separation between:

- `load_fp_tyre_evidence`: local cache read only;
- `local_fp_tyre_evidence_status`: local fingerprint comparison only;
- `calculate_fp_tyre_evidence`: explicit session load/analyse/atomic write;
- `analyse_fp_tyre_evidence_sessions`: pure in-memory calculation.

The calculation inventories Setting, QualiSim, LongRun and Uncertain physical
pit-to-pit runs. Only individually clean laps inside strict LongRuns enter the
quantitative fit. Setting, QualiSim and Uncertain contribute zero observations.
FP track-evolution coordinates are disabled.

The production outputs are `Delta_HM`, `Delta_SM`, and `Delta_SH`, including
strict intervals, classifier sensitivities, leave-one-run/team ranges, profile
information, rank/nullity, and importance-weight diagnostics. One-gap and robust
variability variants are sensitivity results only; the implementation does not
pick whichever result looks most physical.

Direct compound performance requires an estimable same-driver, same-session
strict-LongRun connector. Same-team connectors are reported for diagnosis but are
not used without an explicit driver correction. An unsupported value is preferred
to a QualiSim, Setting, Q-normalized cross-team, or invented proxy.

For the unchanged local 2026 R11 fixture the regression reference is:

```text
8 strict LongRuns; 58 clean laps
rank 3; nullity 1
H-M degradation about -0.0561 s/lap
S-M degradation about -0.1897 s/lap
H-M and S-M direct performance unsupported
```

These values are test expectations, not production constants.

## Artifacts and staleness

The versioned artifact is stored at:

```text
WODATA_ROOT/wostrategy/weekend_model/v1/
  year=<year>/round=<round>/fp_tyre_evidence_v1.json
```

Writes are atomic and occur only after successful analysis. Staleness compares
the artifact's source hashes with locally present FP lap pickles. It does not ask
FastF1 or the network whether newer data exist. A partial FP1+FP2 artifact remains
displayable when FP3 later appears locally and is then marked stale until the user
explicitly recalculates.

## Debugging checklist

When results jump between leader laps, record both numerical and information
diagnostics:

1. Compare the clean-lap fingerprint, scheduled `Lmax` and observed compounds.
2. Compare parameter dimensions and team-compound pairs; a newly observed pair
   changes the search space.
3. Compare fixed-seed accelerated and legacy evaluator outputs.
4. Report ESS/max weight and P10-P90 widths separately.
5. Inspect rank/null vectors and fuel/track/degradation correlations.
6. Confirm reference-compound connectivity before interpreting performance.
7. Confirm no FP raw likelihood or FP-local coordinate entered the Race model.
8. Treat diagnostic scripts as investigation aids, never production imports.

The scripts under `woPlanner/tools/` reproduce the completed audits. Production
code must not import them or depend on their CSV/plot outputs.
