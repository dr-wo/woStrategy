# Shared traffic evaluation

Traffic and clean-lap calculations live in
`wostrategy.analysis.traffic`. Retrospective FastF1 loading remains in the core
loading layer, and live/archive parsing remains in `woData`; those adapters
normalize their samples before calling the same evaluator and clean-lap mask.

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
`Position.z` may still be retained by archive tooling, but the shared estimator
does not require X/Y position coordinates or FastF1's precomputed
`DriverAhead` columns.

Live and replay results are provisional while a lap can still receive samples.
They become comparable with retrospective output after the lap is complete and
the telemetry watermark has advanced beyond it. The direct component may remain
unknown for an incomplete current lap. Batch and replay delivery order do not
affect a finalized result because input is sorted and duplicate timestamps keep
the latest sample.

"Shared" means that equivalent normalized input produces the same evaluation and
clean-lap decision. It does not mean the upstream feeds are currently identical.
The retrospective adapter still uses FastF1 `lap.get_telemetry()`, which merges
car and position channels and may resample them. Live/replay uses the original
`CarData.z` speed messages retained by `woData`. Retaining `get_telemetry()` was a
compatibility compromise: other retrospective consumers still use its distance,
position, and legacy `DriverAhead` columns, and changing that general loader in
the traffic migration would have altered unrelated plots and caches. Consequently,
batch/replay parity is guaranteed for the same normalized rows, while raw
FastF1-versus-live results are expected to be close rather than bit-for-bit equal.

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

Sector-boundary traffic is not numerically merged into telemetry mean gaps. It
is automatically selected only when telemetry is unavailable or does not pass
basic quality/coverage checks. A manual force-sector mode exists for diagnostics.
Every selected result exposes:

- `TrafficMethod`
- `TrafficStatus`
- `TrafficStatusReason`
- `CoverageFraction`
- `TrafficEvaluatorVersion`
- `TrafficCombinationMode`

Component values use `Direct...` and `Physical...` prefixes. Missing telemetry
is represented as `unknown` or `partial_coverage`, never as an infinite clean
gap. The shared clean-lap mask requires usable traffic evidence and applies the
same ahead/behind thresholds to retrospective and finalized live/replay laps.

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

The live store retains a rolling 12-lap telemetry window per driver, and model
recalculation is scheduled at leader-lap boundaries rather than for every
incoming telemetry message. The retained window is re-evaluated so clean-lap
selection stays deterministic; it does not grow to a full 72-lap race. Archive
batch evaluation may contain the full session and follows the same numerical
path.

## Known assumptions and limits

- All drivers' `SessionTime` values must share the same clock.
- Interpolation is rejected when source samples are too old or the bracketing
  interval is too wide; coverage records the resulting missing samples.
- Lap normalization assumes enough plausible speed samples to represent a lap.
- Pit-lane, stopped-car, timing dropout, and lap-attribution anomalies can lower
  coverage and trigger sector fallback.
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
  all-or-nothing fallback with explicit provenance so a selected mean never mixes
  incomparable estimators.
- **Archive Position.z but do not require it.** XYZ coordinates would need circuit
  projection, orientation, and track-map calibration before they yield along-track
  gaps. Speed integration already matches the retrospective estimator contract;
  parsing Position.z into the traffic hot path would add memory and replay cost
  without solving those calibration problems.
- **Bound live memory to 12 laps per driver.** Clean-run fitting needs recent laps,
  not the full race telemetry history. The window keeps recalculation cost bounded
  while leaving enough completed laps for current clean-run selection. It is a
  scheduling/storage choice, not a change to the evaluator.
- **Infer live lap attribution from timing state.** `CarData.z` does not carry the
  strategy lap number, so `woData` assigns samples to the driver's next incomplete
  lap. This assumes timing and car messages are processed chronologically. Samples
  around a delayed finish-line update can be attributed to the adjacent lap;
  coverage/fallback handles gross failures, but this remains a known source-adapter
  limitation.
- **Infer live circuit length from plausible completed speed integrations.** Live
  data has no required static circuit-length input. The median plausible lap span
  avoids coupling the reducer to event metadata; until enough data exists, the
  sector estimator is used. A known circuit length can replace this inference later
  without changing the normalized contract.
