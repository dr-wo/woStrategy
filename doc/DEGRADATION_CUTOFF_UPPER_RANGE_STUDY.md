# Degradation cutoff upper-range study — 2026-10-09

**R&D/design only. Automatic upper extension is not implemented.** Production
still uses the existing upper bound and respects an explicit `scan_max`.
Phase 1 adds coverage reporting, documented in [DEGRADATION_CUTOFF.md](DEGRADATION_CUTOFF.md).

## Sample and reproducibility

The read-only study used eight material saved pre-race inputs across six events,
R11–R16. Multiple runs with the same effective compound vector, performance
deltas, race distance, pit loss and cutoff settings were deduplicated. Two R12
alternatives retained different pit-loss/model assumptions. Synthetic smoke runs,
partial runs and the duplicate R16 research regeneration were excluded. No saved
run, report or canonical prediction was regenerated or overwritten.

Representative runs are under
`woData/woweekend/runs/schema_v1/event=2026-RR/workflow=race_preparation/runs/`:

| Case | Run ID | Effective M | Saved automatic M | Laps / pit loss (s) | Initial upper |
|---|---|---:|---:|---|---:|
| R11 latest | `20260823T095519.658161Z-f96572a2a4` | 0.150213 | unavailable | 70 / 20.5 | 0.300000 |
| R12 baseline, pit 20.5 | `20260823T113808.965065Z-ddc218ac09` | 0.150213 | 0.150213 | 72 / 20.5 | 0.450638 |
| R12 baseline, pit 18.1 | `20260823T112054.300518Z-22efbe2cf3` | 0.150213 | 0.150213 | 72 / 18.1 | 0.450638 |
| R12 latest manual | `20260824T205159.998590Z-b7af8c8d80` | 0.070000 | 0.150213 | 72 / 18.1 | 0.210000 |
| R13 latest | `20261003T095822.772769Z-d474706af5` | 0.223719 | 0.223719 | 53 / 23.0 | 0.671158 |
| R14 latest | `20260913T124237.981225Z-62151e7f88` | 0.139196 | 0.139196 | 57 / 24.2 | 0.417589 |
| R15 latest | `20260926T095545.276998Z-9809d20b84` | 0.039000 | 0.219426 | 51 / 19.3 | 0.117000 |
| R16 latest | `20261003T225359.171082Z-f370e12cf1` | 0.039000 | 0.144070 | 56 / 23.7 | 0.117000 |

All rates are seconds per tyre-age lap. Automatic M comes from the saved
effective artifact's `compounds.MEDIUM.degradation_seconds_per_lap.automatic_value`,
not a replacement read from the current prediction cache. R11's manual-only
fallback lacks that value. R11 explicitly scans 0.05–0.30 with step 0.05 and
tolerance 0.005; the other scenarios use automatic scan controls. The initial
automatic lower bound is 0.25 × effective M. Main strategy max stops is 3 for
R11/R12-pit20.5, 4 for the other scenarios; the cutoff search remains capped at 3.

The exact fixed-stop optimizer uses the saved effective S/M/H ratios, unchanged
compound deltas, fresh starts, any initial compound and separate unrestricted
(U) / two-dry-compound rules-compliant (C) runs. Sweeps and refinements ran in
diagnostic memory only. Ratios against automatic M are coordinate-scale comparisons,
not evaluations of the automatic compound vector when its ratios differ.

## Historical cutoffs and legacy schema

| Case | Current U 0→1 | Saved U 1→2 / 2→3 | Saved C 1→2 / 2→3 |
|---|---|---|---|
| R11 | Not found above explicit 0.05; legacy field absent | not found / 0.145313 | not found / 0.126563 |
| R12 baseline 20.5 | 0.010855, lower extension; legacy field absent | 0.050892 / 0.137812 | 0.036673 / 0.120600 |
| R12 baseline 18.1 | 0.009095, lower extension; legacy field absent | not found / 0.121461 | not found / 0.106831 |
| R12 latest manual | 0.011758, lower extension; legacy field absent | 0.035346 / 0.073846 | 0.036549 / 0.075852 |
| R13 | 0.031024, lower extension | 0.092211 / 0.292532 | 0.100273 / 0.252840 |
| R14 | 0.019846, lower extension; legacy field absent | 0.124515 / 0.248922 | 0.105376 / 0.217023 |
| R15 | 0.018586 | 0.088664 / not found | 0.070992 / not found |
| R16 | 0.020414 | 0.091102 / not found | 0.093539 / not found |

C 0→1 is unavailable under the two-dry-compound rule in every scenario. An absent
legacy field is not evidence of numerical failure. All saved numeric 1→2/2→3
values reproduced exactly at the saved resolution. For R12 baseline/pit18.1,
the old missing 1→2 becomes 0.027871485 U / 0.031392094 C under the calculator's
already-existing lower-extension behavior. This historical version difference
is unrelated to Phase 1's reporting-only changes.

## Higher-stop coverage and fine crossings

Both fixed 2-stop and 3-stop classes were feasible at every initial upper boundary.
Margins are `cost3 - cost2`: positive means 2-stop is cheaper at that boundary.

| Case | Initial boundary margin U / C (s) | Fine 2→3 U / C | Cutoff / effective M U / C | Cutoff / automatic M U / C |
|---|---|---|---|---|
| R11 | −21.698428 / −29.531134 | 0.145740 / 0.127113 | 0.970225 / 0.846219 | unavailable |
| R12 baseline 20.5 | −46.616040 / −59.720906 | 0.137643 / 0.120286 | 0.916323 / 0.800772 | 0.916323 / 0.800772 |
| R12 baseline 18.1 | −49.016040 / −62.120906 | 0.121529 / 0.107028 | 0.809047 / 0.712507 | 0.809047 / 0.712507 |
| R12 latest manual | −11.060000 / −16.265000 | 0.073956 / 0.075929 | 1.056519 / 1.084695 | 0.492344 / 0.505475 |
| R13 | −29.769235 / −41.423456 | 0.292531 / 0.252790 | 1.307580 / 1.129942 | 1.307580 / 1.129942 |
| R14 | −16.351440 / −23.171815 | 0.249206 / 0.217385 | 1.790319 / 1.561720 | 1.790319 / 1.561720 |
| R15 | +6.664000 / +5.020594 | 0.178704 / 0.182778 | 4.582147 / 4.686621 | 0.814415 / 0.832984 |
| R16 | +8.373000 / +8.386396 | 0.182551 / 0.192744 | 4.680794 / 4.942157 | 1.267096 / 1.337847 |

Only R15/R16 U/C lacked initial 2→3 brackets. Diagnostic extension appended
0.025 s/lap upper-range increments from 0.117, with a discretionary 0.30 audit
stress cap. That cap was not a calibrated physical ceiling and was never reached:

| Branch | Margin at 0.142 | At 0.167 | At 0.192 | At 0.217 | First bracket / tested upper |
|---|---:|---:|---:|---:|---|
| R15 U | +3.964000 | +1.264000 | −1.436000 | not needed | 0.167–0.192 / 0.192 |
| R15 C | +3.844953 | +1.553547 | −0.894143 | not needed | 0.167–0.192 / 0.192 |
| R16 U | +5.098000 | +1.823000 | −0.588154 | not needed | 0.167–0.192 / 0.192 |
| R16 C | +5.598577 | +2.702423 | +0.066000 | −2.159865 | 0.192–0.217 / 0.217 |

Across all 16 branch scenarios, local finer-resolution refinements at 1e−6 and
1e−7 agree within 4.6e−7 s/lap. At cutoff ±1e−5, margins have the expected
positive/negative signs; higher stops remain cheaper at cutoff +0.001/+0.005.
Near-crossing stints span 11–26 laps; none is a one-lap/tiny-stint optimizer
pathology. In the four extended scenarios they span 12–19 laps:

| Branch | Two-stop near crossing: compounds / stint lengths | Three-stop near crossing |
|---|---|---|
| R15 U | MMM / 17,17,17 | MMMM / 12,13,13,13 |
| R15 C | HMM / 17,17,17 | HMMM / 13,12,13,13 |
| R16 U | HHM / 19,19,18 | MMMM / 14,14,14,14 |
| R16 C | HHM / 19,19,18 | HMMM / 14,14,14,14 |

Equal-cost ordering permutations are benign. Numerical stability and ordinary
stint lengths do not validate actual tyre availability or physical degradation.

| Distribution | Fine cutoff min / median / max | Cutoff/effective M min / median / max |
|---|---|---|
| Six latest events U | 0.073956 / 0.180627 / 0.292531 | 0.970225 / 1.548949 / 4.680794 |
| Six latest events C | 0.075929 / 0.187761 / 0.252790 | 0.846219 / 1.345831 / 4.942157 |
| Eight material inputs U | 0.073956 / 0.162222 / 0.292531 | 0.809047 / 1.182050 / 4.680794 |
| Eight material inputs C | 0.075929 / 0.154946 / 0.252790 | 0.712507 / 1.107319 / 4.942157 |

For the five latest events with saved automatic M, coordinate ratios have
min/median/max 0.492344/1.267096/1.790319 U and
0.505475/1.129942/1.561720 C. With both R12 alternatives included, the seven
known-automatic scenarios have medians 0.916323 U / 0.832984 C. These scenarios
are neither independent race outcomes nor physical cutoff observations.

The saved R16 automatic-model appendix separately reports compliant 2→3
0.236177784 at upper 0.432210973, margin −20.395334 s. It uses a different S:M:H
vector and is not a ninth independent event in this sample.

## What evidence says about validity

The current 2026 prediction training table contains 41 inferred Retro observations
across R1–15, including 15 MEDIUM-role medians. Their range is
0.052887–0.270673, median 0.150410, descriptive q10/q90 0.064251/0.233861.
The largest MEDIUM P90 is 0.445963 for R10 with 19 clean supporting laps.
These heterogeneous event/absolute-compound model coordinates are context for
sensitivity studies, not measured validity limits. R13's stable unrestricted
cutoff 0.292531 already exceeds the maximum training MEDIUM median.

At R16's effective ray, setting M to automatic M = 0.144070 produces
S ≈ 0.229035 and H ≈ 0.121906. The actual automatic vector instead has
S ≈ 0.204601 and H ≈ 0.091417. Including an automatic scalar in coverage does
not recreate that counterfactual model.

[Prediction documentation](CROSS_EVENT_TYRE_PREDICTION_V1.md) states that Retro
targets are inferred quantities, posterior widths are not calibrated physical
error, descriptor distances are not physical distances and descriptor-domain
warnings do not alter predictions. Selected D0 outputs provide no predictive
uncertainty that could calibrate a ceiling. The Retro producer script's default
0–0.5 degradation sampling bounds are configurable inference priors, not a
physical limit.

The [exact strategy cost](../src/wostrategy/algorithm/exact_strategy_search.py)
integrates linear degradation over tyre age. Compound diversity/availability
rules permit repeated compounds and replacements default to fresh age zero.
There is no complete tyre-set inventory, wear cliff, thermal-state model or
calibrated lifetime limit. Long stints at high rates can therefore be
mathematically costed without establishing useful engineering validity.

A continuous, fresh, single-compound equal-stint approximation gives
`C_s = (s−1)P + d[L²/(2s) + L/2]`, hence `d_2→3 ≈ 24P/L²`.
R16 gives approximately 0.181378, explaining why a 0.117 window can miss 2→3.
This is an explanatory cost scale, not a bound for the actual multi-compound
envelope with sporting rules and integer stint lengths.

## Candidate policies

| Policy | Evidence and tradeoff |
|---|---|
| A: fixed multiple of effective M | Required observed ratios span approximately 0.71–4.94. A 5× multiple happens to cover these cases but is not calibrated. Common rescaling of manual compound rates preserves the sensitivity ray/cutoff while shrinking an effective-M-based upper bound, so any fixed multiplier can fail. Larger multiples also inflate coverage for already-high inputs. |
| B: include automatic/canonical scale | Helps expose low manual scenarios, but automatic M is only a coordinate reference on the effective ray. It supplies neither the automatic vector nor a validity ceiling. Retain as explicit diagnostic/operator context rather than an automatic physical bound. |
| C: adaptive extension with explicit ceiling | Avoids doing extra searches for already-resolved transitions. Phase 1 supplies the feasibility, comparability and boundary margin needed to decide whether extension is useful. A separate configured absolute ceiling makes coverage reproducible without pretending it is physically calibrated. |
| D: explicit `scan_max` only | Preserves current operation and gives exact operator control. Requires event-specific input and can miss useful crossings if omitted; Phase 1 now communicates that omission honestly. This remains the available production behavior. |

## Recommended future design — not implemented

Recommend **opt-in adaptive extension requiring an explicit configured engineering
ceiling**. Until a separate implementation is authorized, use existing explicit
`scan_max` for desired higher-stop coverage. No defensible universal default
physical ceiling can be established from this sample or repository evidence.

Proposed behavior:

1. Start from the unchanged initial scan. Trigger only for a requested unresolved
   higher-stop transition with both boundary classes feasible, sufficient
   comparable adjacent samples and a positive higher-minus-lower boundary margin.
   If the higher-stop class already wins without a bracket, diagnose lower
   coverage/sampling instead of blindly extending the upper range.
2. Append blocks at the existing coarse resolution; a block of five coarse
   intervals equals the study's 0.025 s/lap increments for R15/R16. This is a
   computational step choice, not a modelling ceiling. Keep initial samples and
   already-found cutoffs unchanged; do not coarsen the grid just because the
   covered range grows.
3. Cap each block at a finite operator-configured absolute upper-extension
   ceiling. Require that ceiling to cover the initial range. An absent ceiling
   means no automatic extension; never silently assign the audit's 0.30 value.
4. Stop at the first stable positive-to-negative pairwise crossing after local
   finer-resolution verification, the configured ceiling, unavailable/nonfinite
   class costs, insufficient comparable samples, or unstable/reversal evidence.
   Preserve warnings rather than forcing a simplified cutoff.
5. A user-supplied `scan_max` remains authoritative and disables extension past
   that value, even if a separate extension ceiling is configured. Do not
   reinterpret an explicit maximum as permission to scan farther.
6. Future results need `found_after_upper_extension` for an extended finding,
   and distinct stop reasons for configured ceiling reached, unavailable classes,
   insufficient comparable samples and unstable crossings. Record configured
   ceiling/provenance, expansion history, resolution, verification tolerance and
   stop reason alongside the Phase 1 initial/actual bounds and boundary evidence.
   At a ceiling with no finding, continue to say `not found through`, never infer
   `cutoff > ceiling`.

These are proposals only. Search procedures can be reproducible without a
calibrated validity ceiling; the operator ceiling must therefore be labelled
an engineering/stress assumption. Further work would need calibrated event
degradation uncertainty, inventory-aware strategies and tyre lifetime/model
validation before claiming a default physical ceiling.
