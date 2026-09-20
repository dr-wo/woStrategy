# Push–Cool Sequence Offline Evidence Set

## Outcome and scope

The offline evidence-set and identifiability/sensitivity diagnostic are complete.
This directory freezes the two earlier Push–Cool Sequence Analysis rounds as a
small, reproducible R&D evidence set. It contains exactly five manually reviewed
2026 Spanish Grand Prix FP2 `PitOut -> ... -> PitIn` runs for HAM, RUS, GAS, TSU
and SAI: 34 physical laps and eight push-to-push intervals. It does not search
other sessions or substitute samples.

The evidence can test whether the *sum* of a proposed FP decomposition's fuel,
physical tyre-age, session-time and thermal contributions is compatible with an
observed within-run net pace change. It is not a fuel or tyre truth set, a
calibrated cool-lap benefit, or a standalone decomposition model.

## Evidence files

- [`sequence_runs.csv`](sequence_runs.csv): five run identities, physical pit
  boundaries, selected push/cool-like laps and within-run dependence warning.
- [`sequence_laps.csv`](sequence_laps.csv): every physical lap in each run,
  including outlaps, inlaps and intermediate cool-like laps. Raw `LapTime` and
  elapsed time derived from consecutive `LapStartTime` values are separate.
- [`push_to_push_intervals.csv`](push_to_push_intervals.csv): the eight observed
  push-to-push deltas and their cool count, spacing, elapsed time and descriptive
  telemetry summaries.
- [`excluded_candidates.csv`](excluded_candidates.csv): PER and VER candidates
  excluded because a selected push was deleted. This preserves the production
  flagger's historical output without changing it.
- [`identifiability_sensitivity.csv`](identifiability_sensitivity.csv): the
  previously reported design-rank and normalized assumption-sensitivity snapshot.
- [`source_manifest.csv`](source_manifest.csv): relative source paths, SHA-256
  hashes, table dimensions, code hashes, git HEAD and traffic metadata.

Intervals carrying the same `RunId` are related observations, not independent
replicates. RUS, GAS and TSU each contribute two intervals.

## Manual review semantics

The run boundary requires `PitOutTime` on the declared first lap, `PitInTime` on
the declared last lap, and no intervening pit marker. Selected push laps reproduce
the existing quick-and-clean push flag, are green, accurate and not deleted.
A `cool_like` lap reproduces the existing `slow_lap` role, is green and traffic-ok,
and has lower descriptive mean throttle and speed than its adjacent selected push
running. That label does **not** assert tyre-temperature recovery, a known fuel
rate or a stable operational mode.

The traffic fields are explicitly the historical `unified-traffic-v2` evaluator
with `compatibility_min` combination mode. They must not be described as results
from the current v3 evaluator. Eleven physical laps have missing raw `LapTime`;
those values remain missing. A separately named elapsed value is available for
nine of them from the next `LapStartTime`, but never replaces `LapTime`.

The observed intervals are:

| Driver | Pushes | Cool-like laps between | Observed later-minus-earlier (s) | Cool elapsed (s) |
|---|---:|---:|---:|---:|
| HAM | 2 -> 4 | 3 | -0.767 | 180.282 |
| RUS | 5 -> 7 | 6 | -0.114 | 164.577 |
| RUS | 7 -> 9 | 8 | -0.493 | 161.185 |
| GAS | 2 -> 4 | 3 | -1.660 | 155.277 |
| GAS | 4 -> 6 | 5 | -0.014 | 162.230 |
| TSU | 2 -> 4 | 3 | -1.105 | 157.490 |
| TSU | 4 -> 7 | 5;6 | -0.481 | 304.188 |
| SAI | 5 -> 8 | 6;7 | -1.310 | 289.908 |

SAI lap 9 is a retained trailing cool-like lap inside the full run, but is not
inside the selected lap 5-to-8 interval.

## Minimal formulation and identifiability

For interval \(i\), let \(d_i\) be the directly observed later-push minus
earlier-push raw lap time, \(c_i\) the number of intervening cool-like laps,
\(\Delta t_i\) elapsed session time, and \(z_i\) any thermal descriptor. A
deliberately simple normalized representation is

\[
d_i = -F(1 + \rho c_i) + T(1 + c_i) + S\Delta t_i + H(z_i) + \epsilon_i.
\]

Here `F` is a normalized push-lap fuel-depletion pace contribution, `rho` is the
unknown cool-to-push fuel-exposure ratio, `T` is a linear physical tyre-age
contribution, `S` a session-time contribution and `H` a thermal nuisance term.
None is a physical kg/lap or s/kg quantity. This representation does not alter
the Canonical Pace Contract: Race `fuel_rate` remains seconds per
fuel-proxy-remaining-lap, and an FP per-run intercept is not common intrinsic
pace.

With no thermal term and unknown `rho`, the data identify only

\[
A = T-F, \qquad B = T-\rho F,
\]

through \(d_i=A+B c_i+S\Delta t_i+\epsilon_i\). They do not separately identify
`F`, `T` and `rho`. Fixing `rho < 1` makes the three-column design formally rank
3, but its scaled condition number is about 102–141; at `rho = 1`, fuel exposure
and physical tyre-age exposure are identical and rank falls to 2.

The natural variation does not supply strong independent excitation:

- six intervals contain one cool-like lap and two contain two;
- physical tyre-age delta equals `1 + cool count`, so their correlation is 1;
- interval elapsed and cool elapsed correlate at 0.999786;
- mean cool throttle varies (about 20.9–36.8%), but it is descriptive telemetry,
  not a validated fuel or thermal state proxy;
- eight intervals come from only five runs, with three paired within-run sets.

The no-thermal combination fit has rank 3, scaled condition 86.9 and in-sample
RMSE 0.505 s. Adding a cool-duration thermal column raises formal rank to 4 and
reduces in-sample RMSE to 0.323 s, but condition worsens to 735 and coefficients
swing sharply. A low-throttle-elapsed thermal column gives rank 4, condition 173
and RMSE 0.461 s. These are sensitivity diagnostics, not evidence for either
thermal law. Mapping the same no-thermal combinations under assumed `rho` moves
the normalized `F` value from 0.034 at `rho=0` to 0.340 at `rho=0.9`, showing
that separate fuel/tyre interpretation is assumption-dependent even though the
observed deltas are unchanged.

The robust conclusion is therefore limited: pit-bounded differencing removes a
run-level intercept and constrains a net within-run contribution. Separate fuel,
tyre-age, session and thermal effects remain confounded. The most valuable added
observables would be (1) a credible lap-resolved fuel/energy-use exposure and
(2) tyre surface/carcass temperature at push boundaries. Either supplies state
variation not algebraically determined by cool count and elapsed time.

## Reproduction

Run from the `woStrategy` repository with the existing local `woData` checkout:

```bash
MPLCONFIGDIR=/private/tmp \
PYTHONPYCACHEPREFIX=/private/tmp/pycache-push-cool \
PYTHONPATH=src:../woData/src \
.venv/bin/python -m wostrategy.script.push_cool_sequence_evidence \
  --data-root ../woData \
  --output-dir doc/push_cool_sequence_evidence
```

To verify a clean rebuild without overwriting this directory:

```bash
rebuild=/private/tmp/push_cool_sequence_evidence_rebuild
mkdir -p "$rebuild"
MPLCONFIGDIR=/private/tmp \
PYTHONPYCACHEPREFIX=/private/tmp/pycache-push-cool \
PYTHONPATH=src:../woData/src \
.venv/bin/python -m wostrategy.script.push_cool_sequence_evidence \
  --data-root ../woData --output-dir "$rebuild"
for name in sequence_runs sequence_laps push_to_push_intervals \
  excluded_candidates identifiability_sensitivity source_manifest; do
  diff -u "doc/push_cool_sequence_evidence/$name.csv" "$rebuild/$name.csv"
done
```

The generator reads only the two cache paths recorded in `source_manifest.csv`.
Missing input fails explicitly; it never downloads data or silently changes the
sample. The manifest records the base repository HEAD at generation time,
`b2ed14d755de5fbb4a88fbb92505325b5d4d6db7`, plus exact hashes of the R&D
generator and the production code it calls. The generator reuses
`wostrategy.analysis.push_laps.add_push_lap_flags`; it does not alter that flagger.

Run the focused checks with:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pycache-push-cool \
PYTHONPATH=src:../woData/src \
.venv/bin/pytest -q tests/test_push_cool_sequence_evidence.py
```

## Correct use and limits

For a proposed decomposition, calculate every contribution under explicit
assumptions and compare their sum with `ObservedPaceDeltaSeconds`, retaining a
residual and grouping intervals by `RunId`. If these rows participated in fitting,
that comparison is only an in-sample compatibility check—not independent
validation.

Do not use these rows as true fuel load, kg/lap, s/kg, tyre degradation or tyre
temperature labels. Do not infer a universal cool-lap benefit or treat a
cross-driver shared coefficient as validated. Existing fuel/tyre-corrected FP
outputs must not be used to validate the assumptions that produced them.

The current disposition is to retain the evidence only as an auxiliary
within-run net-change constraint and model-consistency check. The available data
and assumptions cannot stably separate fuel, tyre, thermal and session evolution,
so an independent physical decomposition is deferred rather than treated as a
committed feature. Re-evaluate only if new independent information becomes
available or there is a specific, testable assumption worth checking; this
conclusion creates no implementation commitment or expanded backlog.
