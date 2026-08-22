# Cross-event tyre prediction V1

## Status and scope

This document records the final V1 architecture, assumptions, compromises, validation
rules, and prospective operating procedure for the cross-event tyre predictor. The model
formulations are intentionally frozen after the 2026 R1-R11 retrospective development
sample. Further model redesign is deferred until additional genuinely prospective results
are available.

The workflow is downstream-only:

```text
Pirelli preview descriptors
        +
existing completed Race Retro artifacts
        ↓
supported historical observations
        ↓
P0/P1 performance and D0/D1 degradation regressions
        ↓
versioned pre-race predictions
        ↓
later prospective scoring when Retro becomes available
```

It does not change or invoke Retro MC and does not feed predictions into FP, Race MC,
strategy priors, or live-race state.

## Input semantics

### Pirelli previews

The pipeline consumes accepted preview records produced by `wodata.pirelli_preview`.
The V1 regressions use only:

- `tyre_stress` and `asphalt_grip` for performance;
- `tyre_stress`, `asphalt_abrasion`, and `asphalt_grip` for degradation;
- the absolute C1-C5 allocation behind the HARD, MEDIUM, and SOFT event roles.

Traction, lateral characteristics, longitudinal/lateral heatmaps, weather, temperature,
team identity, and detailed tyre-map information are deliberately omitted. They may be
physically relevant, but the current history is too small to support a larger identifiable
linear model.

The published 1-5 descriptors are ordinal categories, not precise equally spaced physical
measurements. V1 still encodes them numerically as a deliberate simplification. Values 1
and 5 are treated diagnostically as saturated boundary bins: the underlying physical
quantity may extend beyond the nominal endpoint. No Euclidean or ordinal descriptor
distance is treated as physical distance, and descriptor-domain warnings never change,
scale, weight, or widen a prediction.

### Retro observations

Historical performance and degradation values are inferred Retro MC quantities, not
measured ground truth. The extractor reads existing saved artifacts only.

- Performance target: `CompoundDeltaMedianSeconds`.
- Performance coordinate: event HARD is exactly zero.
- Degradation target: `DegradationMedianSecondsPerLap` from global compound rows only.
- Team-specific degradation variation is excluded.
- P10 and P90 are retained with `W80 = P90 - P10` for diagnosis.

P10-P90 concentration is not assumed to be calibrated physical error. A narrow interval
may still be incorrectly centred, so inverse-variance weighting is not used.

Unsupported compounds remain absent. The extractor does not fabricate, interpolate, or
backfill a missing Retro result from the cross-event model. Consequently, an event may
contribute fewer than three rows.

## Performance formulations

Performance remains directly compatible with Retro's HARD-relative sign convention.

### P0: historical baseline

P0 fits four transferable adjacent-compound gap intercepts:

```text
alpha_C1_C2
alpha_C2_C3
alpha_C3_C4
alpha_C4_C5
```

Wider compound differences are signed sums of the relevant adjacent gaps. For an event
using HARD=C2, MEDIUM=C3, and SOFT=C4:

```text
HARD   = 0
MEDIUM = gap(C2,C3)
SOFT   = gap(C2,C3) + gap(C3,C4)
```

### P1: Pirelli-informed performance

P1 preserves the same four gap intercepts and adds two environmental coefficients shared
by every adjacent gap:

```text
gap_j = alpha_j + beta_stress * tyre_stress + beta_grip * asphalt_grip
```

Sharing sensitivities is a small-sample identifiability compromise, not a physical claim
that every compound pair truly responds identically. The preceding compound-specific
environmental design had rank 12/16 and was rejected rather than hidden with Ridge.

The current fits use zero Ridge penalty by default. A configurable model abstraction is
retained, but regularisation is not tuned against the validation set.

HARD identity rows are emitted as exactly zero but excluded from informative performance
metrics.

## Degradation formulations

### D0: historical baseline

D0 fits one absolute intercept for each C1-C5 compound:

```text
D(event, compound) = alpha_compound
```

### D1: Pirelli-informed degradation

D1 retains the existing absolute linear formulation:

```text
D(event, compound) =
    alpha_compound
    + beta_stress * tyre_stress
    + beta_abrasion * asphalt_abrasion
    + beta_grip * asphalt_grip
```

There are no interactions, polynomials, nonlinear transforms, compound-specific
sensitivities, or team terms.

## Observation weighting

Regression formulation and observation weighting are separate metadata concepts. P0,
for example, remains the same physical equation under all supported policies:

- `uniform`: every supported observation receives weight one;
- `event_normalised`: every event has total regression weight one;
- `evidence_event_normalised`: within-event quality is `sqrt(clean_lap_count)`, followed
  by event-total normalisation to one.

The selected direct evidence field is compound-level clean-lap count derived from the
already persisted `clean_laps.csv`. Square-root transformation provides diminishing
returns. Persisted run and stint counts remain diagnostics. Event-level ESS, weighted
Retro RMSE, MC sample count, and weight sum are sampler/fit diagnostics rather than direct
compound evidence and are not primary weights.

Event normalisation prevents an event with three supported compounds from automatically
having more influence on event-level Pirelli coefficients than an event with two. Better
supported compounds may still receive a larger fraction within an event.

Evidence and event-normalised policies remain comparison diagnostics. The compatibility
default is fixed at P0/uniform and D0/uniform; a numerically tiny validation advantage is
not sufficient for automatic switching.

## Readiness and identifiability

Performance and degradation readiness are independent. The default checks include:

- all C1-C5 compounds represented;
- a connected performance comparison graph;
- every adjacent performance gap represented;
- useful variation in the model's Pirelli features;
- full design rank;
- at least 8 informative non-HARD performance rows;
- at least 12 degradation rows.

The reduced full-data P1 design is rank 6/6 with 20 informative R1-R11 observations. D1
is rank 8/8 with 30 observations. Full mathematical rank is necessary but not evidence of
physical coefficient stability. Weighted design condition numbers and complete rolling
coefficient histories are therefore persisted.

## Historical rolling validation

Validation reproduces the pre-race information state:

```text
train through R(k) → predict R(k+1)
```

There is no random train/test split and no later round can enter an earlier fit. A
prediction is emitted only after that formulation passes readiness at the cutoff.
Historical rolling and genuine prospective scores are labelled and aggregated separately.

Each residual record includes the event and compound, prediction and observation,
descriptors, allocation, evidence support, P10/median/P90/W80, training cutoff and hash,
and descriptor-domain state. Performance records also retain deterministic adjacent-gap
decomposition where available.

Current uniform rolling results are:

| Model | Count | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| P0 | 11 | 0.11827 | 0.17322 | -0.02583 |
| P1 | 11 | 0.19224 | 0.26203 | +0.06009 |
| D0 | 16 | 0.08437 | 0.10782 | -0.03622 |
| D1 | 16 | 0.10091 | 0.18783 | -0.05943 |

These results establish current validation context; they do not prove that the baseline
is physically correct or that descriptor dependence does not exist.

## Descriptor-domain diagnostics

For performance, the domain tuple is `(tyre_stress, asphalt_grip)`. For degradation it
is `(tyre_stress, asphalt_abrasion, asphalt_grip)`.

At every rolling cutoff the pipeline records:

- whether each used descriptor equals saturated boundary level 1 or 5;
- whether each exact level appeared in prior training events;
- whether the exact joint tuple appeared previously;
- the number of distinct prior events with the exact tuple.

P0 is classified using the same event domain as P1, and D0 using the same domain as D1,
so comparisons remain paired even though the baselines do not use descriptors in their
equations.

Current split results show P1 approximately matching P0 on four interior observations
but performing worse on seven boundary-exposed observations. D1 beats D0 on six
interior/seen-tuple observations but performs worse on ten boundary/unseen observations.
For degradation, boundary exposure and tuple novelty coincide in the current sample, so
their effects cannot be separated. These small, confounded subsets are diagnostic only;
they do not trigger a new model, correction, uncertainty multiplier, or promotion rule.

## Dual prediction output

Every upcoming-event compound exposes two transparent families:

```text
historical_baseline:
    performance = P0
    degradation = D0

pirelli_informed:
    performance = P1
    degradation = D1
```

The historical baseline currently has lower overall rolling error. The Pirelli-informed
family expresses the intended simplified environmental dependence but has not yet shown
better overall rolling prediction. Neither family is labelled correct or incorrect.

Existing flat `performance_mean` and `degradation_mean` fields remain aliases of the
historical baseline for compatibility. `default_prediction_family` is therefore
`historical_baseline`, while the complete Pirelli-informed alternative and both families'
validation metrics remain visible in the same artifact.

Upcoming events also include boundary warnings and prior tuple counts. Warnings describe
the ordinal descriptor information state only and do not invalidate a prediction.

## Prospective observe mode

Predictions are written to an immutable fingerprinted version path as well as the current
compatibility path. When a previously missing Retro event appears, the pipeline:

1. reads the latest frozen pre-race version for that round;
2. scores informative P0/P1 performance and all D0/D1 degradation predictions;
3. appends idempotent `prospective` residual records;
4. leaves the frozen prediction unchanged;
5. rebuilds historical rolling diagnostics and coefficients through the new event;
6. generates predictions for the next accepted preview without Retro support.

If metadata-only versions exist for the same round and training cutoff, only the latest
pre-race information state is scored, avoiding duplicate prospective influence. The
fingerprinted versions remain available as provenance.

Prospective evidence should become more important than the R1-R11 retrospective sample.
Promotion of P1/D1 or another weighting policy requires an explicit future development
decision based on sustained evidence; there is no automatic promotion rule.

## Artifacts and reproducibility

Artifacts live below:

```text
woData/wostrategy/tyre_prediction/schema_v1/year=<year>/
```

Important files are:

- `retro_tyre_observations_<year>.csv`: supported global Retro observations;
- `training_data.csv`: joined observations and accepted descriptors;
- `validation.json`: model definitions, readiness, rolling records, and metrics;
- `validation_predictions.csv`: enriched historical residual diagnostics;
- `coefficient_history.csv`: one row per eligible fit, cutoff, formulation, and policy;
- `diagnostic_summary.json`: rolling metrics, stability, residual rankings, and domain
  splits;
- `prospective_validation.csv`: genuine frozen-prediction scores only;
- `predictions/round=<round>/prediction.json`: latest compatibility artifact;
- `predictions/round=<round>/versions/<fingerprint>.json`: immutable evidence versions;
- `prediction_history.csv`: idempotent user-facing prediction history.

Training hashes cover the observations and features used by a fit. Preview and Retro
artifacts retain their own source hashes and timestamps. Atomic replacement is used for
reproducible derived tables; immutable prediction versions are never rewritten.

Run the pipeline with:

```bash
PYTHONPATH=../woData/src:src python -m wostrategy.script.tyre_prediction_pipeline \
  --year 2026
```

## Deliberately deferred work

V1 does not add or tune:

- nonlinear, ordinal, censored, GAM, GP, tree, neural, or ensemble models;
- descriptor interactions or polynomial terms;
- compound-specific environmental sensitivities;
- traction, lateral, energy-map, temperature, weather, or team features;
- MC uncertainty recalibration or inverse-variance weighting;
- automatic uncertainty inflation at descriptor boundaries;
- FP/Race MC prior integration or live-race updating;
- automatic model or weighting-policy promotion.

These omissions are intentional small-data compromises, not claims that the omitted
physics is unimportant.
