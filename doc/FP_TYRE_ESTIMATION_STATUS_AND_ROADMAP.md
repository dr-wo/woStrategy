# FP Tyre Estimation: Status and Roadmap

## Purpose and scope

This document freezes the current engineering interpretation of public-data FP tyre
evidence. It separates what is available for reporting from what is approved for
Strategy Prediction. The current FP degradation and performance work remains
diagnostic: no FP value automatically replaces P0/P1 performance or D0/D1 degradation.

Canonical live artifacts are under
`woData/wostrategy/tyre_prediction/schema_v1/year=<year>/`:

- `fp_degradation_calibration/latest_calibration.json` is the compact state for later events;
- `fp_degradation_calibration/versions/through_round=<N>.json` freezes each Retro cutoff;
- `fp_race_diagnostic/fp_degradation_calibration_history.csv` retains the full response history;
- `fp_race_diagnostic/fp_race_diagnostic_summary.json` retains detailed research results.

The implementation entry points are
`src/wostrategy/analysis/fp_race_diagnostic.py` and
`src/wostrategy/analysis/fp_pre_race_report.py`.

## Implemented functionality and deliberate compromises

The completed implementation adds the following functionality:

- FP1, FP2 and FP3 session artifacts are causal snapshots. Their posterior,
  analysis identity and manifest provenance cannot be conditioned on a later FP
  session or Race, while the aggregate weekend artifact retains its intended
  all-available-information behaviour.
- Historical FP-to-Race replay evaluates direct degradation movement and
  leave-one-compound-out partial-K transfer on the fixed V1 grid, separately by
  session and prior family, with event metrics and leave-one-event-out stability.
- Race Retro completion automatically refreshes the calibration history and a
  compact `diagnostic_only` snapshot for the next event. Through-round versions
  are immutable, and a pre-race report can only load a cutoff earlier than its
  target round.
- The weekend workflow writes a separate FP tyre-evidence JSON/Markdown report.
  It displays both D0 and D1 comparisons, provenance, support and limitations,
  but does not mutate the frozen prediction artifact or planner inputs.
- Corrected-intercept and historical team/session programme-offset diagnostics
  are retained for future performance research, including strict rolling
  eligibility rather than in-sample claims.

The engineering compromise is intentionally conservative:

- Useful FP degradation evidence is surfaced, versioned and automatically
  recalibrated, but it is not promoted into Strategy Prediction. A displayed
  weight or partial update is a diagnostic candidate, never an automatic input.
- The fixed five-point grid is retained instead of fitting a continuous weight.
  This gives an auditable response surface at the cost of coarse calibration
  until substantially more independent events exist.
- FP sessions stay independent rather than adopting an FP2-only architecture.
  Current weak FP1/FP3 results remain empirical states that future evidence can
  change.
- Partial-K transfer remains visible but experimental; it is not treated as a
  measured physical constant or enabled for uninformed compounds in production.
- Fuel-burn and track-index effects are corrected where identifiable, but unknown
  starting fuel, engine mode and programme convention are not fabricated. This
  leaves FP compound-performance updating paused because the present same-team
  rolling evidence is insufficient.
- Missing sessions, sprint formats and weak support fall back to the pre-race
  priors with an explanation rather than manufacturing an FP adjustment.

## Current degradation architecture

The information flow is:

```text
Historical Race Retro + Pirelli preview/history
                    ↓
       pre-race D0 and D1 degradation priors
                    ↓
       causal FP1 / FP2 / FP3 evidence states
                    ↓
 session × prior-family × update-type calibration
                    ↓
                  Race
                    ↓
               Race Retro
                    ↓
 refresh calibration for the next event only
```

FP1, FP2 and FP3 are independent calibration tracks. Neither FP1 nor FP3 is
permanently fixed to zero. The current evidence can change as prospective races are
added.

For a directly observed compound, the diagnostic candidate is

```text
D_updated = (1 - w_session) D_prior + w_session D_FP
```

For experimental transfer to an uninformed compound,

```text
K_effective = 1 + w_session (K_event - 1)
D_updated   = K_effective D_prior
```

Direct and partial-K tracks remain distinct for each prior family. `K_event` is an
empirical event-severity ratio, not a measured physical constant. Neither equation is
currently a production Strategy Prediction rule.

## Calibration V1

`calibration_v1` uses the fixed candidates `0, 0.25, 0.50, 0.75, 1.00` and the method
`fixed_grid_event_mean_mae_with_loeo_v1`. It persists event counts, validation error,
bias, event-mean MAE and leave-one-event-out selection stability through each completed
Retro round. Its status is always `diagnostic_only`; a grid preference is not a
production-selected weight.

The compact snapshot is rebuilt automatically after Retro persistence and downstream
rolling validation. Race N updates the snapshot through N for Race N+1. It never edits
the frozen Race N prediction or its version artifact.

## Report-facing degradation names

Human-readable FP tyre-evidence reports use neutral degradation labels:

- `Historical degradation` for the internal D0 / `historical_baseline` family;
- `Pirelli-informed degradation` for the internal D1 / `pirelli_informed` family.

This is deliberately a display-only mapping. The machine-readable JSON, calibration
joins, immutable prediction artifacts, and `selected_default` field retain the existing
technical IDs. That compatibility compromise avoids an artifact-schema migration and
keeps historical calibration data readable, but means report consumers must not derive
presentation text by simply displaying or title-casing the internal family identifier.
Any additional report surface should use the same explicit mapping.

Once substantially more independent events exist, a later calibration version may
test a statistically defined continuous relationship such as

```text
Race degradation correction ~ intercept + w × (FP - prior)
```

That regression is intentionally not implemented now.

## Current degradation evidence

The present conclusions are:

- FP2 contains the clearest repeatable useful degradation information.
- Partial movement is more defensible than complete FP replacement.
- Complete replacement generally strengthens the observed negative bias.
- FP1 and FP3 do not currently show reliable aggregate improvement.
- Preferred grid strength changes by prior family, direct versus K transfer and held-out event.
- No permanent production weight has been selected.
- All three sessions continue to be independently recalibrated.

Do not copy event counts from this document into decisions. Read the canonical compact
snapshot, whose cutoff and counts evolve after each Race Retro.

## Pre-race reporting policy

The FP tyre-evidence report is written beside the causal weekend model. It shows D0 and
D1 separately, the FP estimate, FP-minus-prior difference, and—when causally available—a
clearly labelled experimental candidate. It also shows calibration version, Retro
cutoff, event count and LOEO stability.

Missing sessions and unsupported compounds produce an explicit “prior unchanged” state.
This includes sprint formats without the normal FP schedule. Report generation does not
mutate planner-facing or Strategy Prediction inputs.

The report wording must preserve the distinction between “FP2 currently has the
strongest evidence” and “only FP2 is valid.” The latter is not supported.

## Why FP performance updating is frozen

An FP long-run intercept contains unresolved contributions from:

- team/car performance;
- compound performance;
- unknown starting fuel load;
- engine mode;
- setup and programme convention;
- track state;
- residual modelling error.

Fuel-burn slope can be estimated more readily than absolute starting fuel. Removing
run-lap fuel burn and track-index effects therefore does not put different runs onto a
fully comparable absolute pace coordinate.

Historical tests found catastrophically poor cross-team baseline-corrected recovery and
poor same-team recovery. Reconstructed fuel/track-corrected intercepts improved the two
supported same-team cases only slightly; error remained of order one second, while P0/P1
were dramatically better on the same cases. Run-intercept-based performance updating is
therefore paused. The sample is extremely small, so this is not proof that useful FP
performance information is physically impossible to recover.

## Programme-offset experiment

The tested hypothesis was that teams may use sufficiently repeatable long-run reference
programmes for longitudinal comparison. A historical team × session effective offset
might then normalize later FP pace.

The offset is an effective pace quantity containing starting load, engine mode,
programme convention and possibly setup effects. It is not inferred fuel mass, and the
hypothesis does not require identical starting kilograms at every event.

Current stability is mixed. Some team/session histories are relatively narrow, while
others vary by many seconds. Most variance remains in team/run-specific residual and
noise. Only two same-team compound bridges exist, and neither has two preceding matching
events for a strict rolling prediction. The transfer hypothesis is therefore unvalidated,
not conclusively disproven. Cross-team FP performance development remains paused.

## Future Race-pace-informed performance research

The intended next phase, once enough prospective cases exist, is:

```text
completed historical Race performance
                 ↓
learn a team's effective FP programme reference
                 ↓
next-event corrected FP run pace
                 ↓
subtract the strictly historical programme offset
                 ↓
estimate current-event team-normalised FP pace
                 ↓
test same-team compound-performance recovery
```

Conceptually:

```text
corrected_FP_intercept(event, team, session)
  ≈ current team/car performance
  + compound performance
  + transferable team/session programme offset
  + residual
```

Strict chronology is mandatory. Before Race N, only completed Race results through N-1
may estimate its programme offset. Race N may update the state only after completion for
Race N+1.

Future work must determine the most appropriate car-performance reference. Candidates
include the existing Race team baseline, qualifying/team pace information, or a rolling
team-performance estimate. No choice is made here. The criterion is future-event
compound recovery, not in-sample fit.

## Validation gate and prospective principle

Restart performance development only after enough strict rolling cases exist to compare
historical programme-offset correction against held-out next-event same-team compound
targets. Same-team recovery remains the first gate. Cross-team baseline correction may
be reconsidered only if that gate becomes reasonably accurate.

Future weekends should primarily add prospective evidence:

1. freeze the pre-race prediction and each causal FP information state;
2. run the Race and persist final Retro evidence;
3. score the frozen states without rewriting them;
4. append FP-to-Race calibration evidence;
5. rebuild calibration for later events.

This prevents repeated retrospective model changes from masquerading as prospective
performance.

## Rejected or paused approaches

- Permanent FP2-only architecture: rejected; calibration must remain session-specific.
- Hardcoded FP1/FP3 zero weights: rejected; future evidence must be able to move them.
- Full FP degradation replacement: not supported by current bias and hold-out evidence.
- Automatic partial-K production transfer: paused pending more events.
- Continuous weight regression: deferred because the independent-event sample is too small.
- Literal starting-fuel inference from public FP: unsupported.
- Current run-intercept performance updates and new cross-team baselines: paused at the same-team gate.
