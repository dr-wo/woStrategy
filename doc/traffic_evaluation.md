# Shared traffic evaluation

Traffic and clean-lap calculations live in
`wostrategy.analysis.traffic`. Retrospective FastF1 loading remains in the core
loading layer, and live/archive parsing remains in `woData`. The numerical core
accepts normalized telemetry, but default live Race clean-lap decisions no longer
use the old zero-origin reconstruction.

## Current live and FP contract (R&D-2 Part 1)

Default automatic live traffic returns `traffic-unavailable`, including repeated
requests. Existing Race model state, evidence counts and run/parameter history
are retained. There is **no automatic sector fallback**. Sector traffic requires
explicit selection and keeps its `SECTOR_BOUNDARY_FALLBACK` provenance; prepared
rows cannot bypass the quarantine or impersonate sector traffic.

`Position.z` history supports the track map and a diagnostic position-traffic
shadow. Position-derived gaps are **not production-active**. woPlanner's track-map
store and causal replay builder use the same woData retention implementation:
latest 1,200 sample-time seconds, capped at 6,000 samples per driver. Delayed
batches are sorted by embedded timestamps, with unknown status for conflicting
samples. Diagnostic cuts reconstruct only packets through the ingress boundary
and then filter sample time. A warmed later snapshot cannot establish historical
arrival causality; raw positions do not enter MC inputs.

Fresh traffic-derived FP evidence requires `unified-traffic-v4` and
`compatibility_min` provenance. Cache loaders invalidate stale identities;
materialized frames and direct diagnostic/calibration consumers reject v2, v3,
unversioned or contradictory evidence before fresh quantitative analysis. Loading
a frame cannot certify it: genuine derivation is required. Input hashes and
contract metadata protect provenance. Saved models and calibrations remain
historical artefacts and are not rewritten or retroactively certified as v4.

Part 1 is complete with independent final gate **PASS**. Position-gap activation
remains a separate unresolved task. Rear-filter relaxation belongs to Part 2 and
was not implemented. See [closure evidence](../research/traffic_position_part1/CLOSURE.md).

## Normalized input and finalization

The minimum evaluator input is one timestamped table containing all available
drivers:

| Column | Meaning |
| --- | --- |
| `Driver` | Stable driver number or code within the session |
| `SessionTime` | Comparable session timestamp, in seconds or timedelta form |
| `LapNumber` | Official lap attribution for anchoring and output grouping |
| `Speed` | Car speed in km/h |

`Year`, `Round`, and `SessionName` are optional grouping keys. `Distance` may be
supplied by retrospective FastF1 telemetry and is used as a phase anchor; it is
not required by the normalized live contract. `CarData.z` supplies speed and
`Position.z` is retained separately for the track map and diagnostic shadow.
The retrospective telemetry estimator does not require X/Y coordinates or FastF1's precomputed
`DriverAhead` columns.

The direct component may remain unknown for an incomplete current lap. Input is
sorted by sample time. Conflicting target lap labels at one timestamp are barriers
to rear ownership; packet order cannot decide the target lap. These numerical
finalization rules do not lift the default live quarantine.

"Shared" means that equivalent normalized input produces the same evaluation and
clean-lap decision. It does not mean the upstream feeds are currently identical.
The retrospective adapter still uses FastF1 `lap.get_telemetry()`, which merges
car and position channels and may resample them. Live/replay uses the original
`CarData.z` speed messages retained by `woData`. Retaining `get_telemetry()` was a
compatibility compromise: other retrospective consumers still use its distance,
position, and legacy `DriverAhead` columns, and changing that general loader in
the traffic migration would have altered unrelated plots and caches. Consequently,
batch/replay parity is guaranteed for the same normalized rows, while raw
FastF1-versus-live comparisons require separate validation. Equivalent input
parity is not evidence that live zero-origin gaps are valid.

## Estimator components

### `DIRECT_FASTF1_STYLE`

This component adapts the numerical approach of FastF1
`Telemetry.calculate_driver_ahead` under FastF1's MIT licence. Speed is
integrated once for every driver/lap. For each target lap, the other driver's
finish-line-relevant lap is selected, its reusable progress trace is
interpolated directly onto the target timestamps, and the nearest non-negative
distance is selected. The derived distance is converted to time with the target
lap's distance/time curve.

This preserves FastF1's classification-lap anchoring for compatibility. It is
therefore not the authority for nearby lapped traffic. Retrospective output also
retains legacy FastF1-supplied columns, when present, under `LegacyFastF1...`
names so migration differences remain inspectable.

Rear observations belong to the target driver, sample time and target lap, never
the follower's lap. Exact target labels or agreeing bounded brackets establish
ownership; conflicting or unsupported labels remain unavailable. Session grouping
and legacy/fallback paths obey the same rule. This ownership change and the v4
contract land together so stale traffic semantics cannot survive in fresh FP evidence.

### `PHYSICAL_CIRCULAR`

Speed is integrated from the normalized timestamped samples, once per lap, with
no additional fixed-frequency upsampling inside the evaluator. A complete lap is
normalized to the estimated circuit length; supplied distance can anchor its
starting phase. Other-driver progress is interpolated once at the target
timestamps. Ahead and behind gaps are circular:

```text
ahead  = (other_phase - target_phase) mod circuit_length
behind = (target_phase - other_phase) mod circuit_length
```

Official lap numbers help select and anchor telemetry but are removed from the
physical phase comparison. Consequently, a car multiple classification laps
behind can still be the physically nearest car ahead or behind, including
across the finish line.

### Selection and fallback

The selected telemetry metric currently uses the versioned
`combination_mode=compatibility_min` rule. Direct and physical values are saved
separately, and the selected value is the smaller available value on each side.
This conservative historical behavior is intentionally a compatibility option,
not an irreversible definition of traffic.

Sector-boundary traffic is not numerically merged into telemetry mean gaps.
It is available only through explicit sector selection; missing telemetry never
selects it automatically. Every selected result exposes:

- `TrafficMethod`
- `TrafficStatus`
- `TrafficStatusReason`
- `CoverageFraction`
- `TrafficEvaluatorVersion`
- `TrafficCombinationMode`

Component values use `Direct...` and `Physical...` prefixes. Missing telemetry
is represented as `unknown` or `partial_coverage`, never as an infinite clean
gap. The shared clean-lap mask requires usable traffic evidence and applies the
unchanged ahead/behind thresholds where valid selected evidence is available.
Default live/replay automatic traffic supplies no such evidence.

## Performance assumptions

For `L` laps, `D` drivers, and `S` telemetry timestamps per lap, nearest-driver
comparison is approximately `O(L × D² × S)`. A 72-lap, 22-driver, 80-second,
5 Hz race contains about 13.9 million candidate point comparisons and 633,600
telemetry rows. These comparisons are performed in NumPy matrices. Speed
integration occurs once per driver/lap, and finish-line progress suffixes are
cached per driver/lap rather than rebuilt for every opponent.

On the development machine, a synthetic workload of that full size evaluates
the direct component in roughly six seconds. This is a diagnostic expectation,
not a portable performance guarantee. Runtime scales with telemetry rows and
drivers; the Monte Carlo sample count (for example 80,000 race-model samples) is
a separate later calculation and does not multiply traffic-estimator work.

woData's general store defaults to a 12-lap car-telemetry window. woPlanner's
worker retains generation message history for causal replay and disables that
car-telemetry lap limit. Position retention is separately bounded as above:
at most 132,000 position samples per store for 22 driver identities. At 5 Hz the
6,000-sample cap covers about 20 minutes; at 10 Hz it covers about 10 minutes.
The cap therefore remains meaningful even for dense timestamp streams, while
recent complete laps remain available to the track map. This is a bound on the
position buffers, not the separate raw recording or replay-message history.
Recalculation requests occur at leader-lap boundaries; automatic requests return
the quarantine status without fitting a new Race update.

## Known assumptions and limits

- All drivers' `SessionTime` values must share the same clock.
- Interpolation is rejected when source samples are too old or the bracketing
  interval is too wide; coverage records the resulting missing samples.
- Lap normalization assumes enough plausible speed samples to represent a lap.
- Pit-lane, stopped-car, timing dropout, and lap-attribution anomalies can lower
  coverage; they never trigger automatic sector fallback.
- Run-continuity rules and numerical threshold tuning are outside this traffic
  implementation and remain unchanged.

## Decisions and compromises

The following choices are intentional and should be reconsidered explicitly
rather than changed as incidental refactoring:

- **Local FastF1-style implementation instead of calling a Session method.**
  FastF1's method requires a loaded `Session`, lap tables, and car data around a
  completed time window. The local component preserves its finish-line anchoring
  and distance comparison, but accepts normalized buffers so retrospective,
  archive, and live callers share the numerical core.
- **Algorithm adaptation rather than a literal per-lap copy.** Speed is integrated
  continuously once per driver, reusable finish-line trace suffixes are cached,
  and each opponent is interpolated directly onto target timestamps. This avoids
  repeatedly building pandas telemetry for every `lap × target × opponent` while
  preserving the finish-line integration interval that FastF1 uses.
- **Keep direct and physical components.** The direct component provides migration
  compatibility and comparison with historic FastF1-derived caches. Its official
  lap anchoring is unsuitable as the sole lapped-car detector, so the circular
  component separately locates physical neighbours without retaining whole-lap
  classification differences.
- **Keep `compatibility_min` for now.** Git history showed that `min(direct,
  physical)` was deliberately introduced as a conservative traffic rule. It can
  over-select the more pessimistic component and is not claimed to be physically
  optimal. Versioning and separate component columns make a later selector change
  measurable and cache-safe.
- **Do not blend sector gaps into telemetry means.** Four boundary observations
  have different sampling semantics from continuous telemetry. Sector data is an
  explicitly selected source with provenance so a selected mean never mixes
  incomparable estimators.
- **Keep position gaps diagnostic.** Track-map visualization and the position
  shadow use retained XYZ samples, but neither validates production time-gap
  thresholds. Production activation requires a separate decision and evidence.
- **Preserve causal position retention.** The shared capped/time-window policy
  supports both track-map references and diagnostic analysis cuts. Replay
  reconstruction respects arrival order before filtering embedded sample times.
- **Do not infer valid live gaps from speed origins.** `CarData.z` lap attribution
  follows timing state and can lag a physical crossing. Independent lap-labelled
  speed origins are not a shared track coordinate. The default live path is
  quarantined instead of inferring circuit length and automatically using sectors.
