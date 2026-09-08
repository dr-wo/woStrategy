# Local Pairwise Race-Performance Diagnostic

The local pairwise workflow is an offline diagnostic for asking how stable the
production Race performance conclusion is when comparisons are restricted to
teams running contemporaneously. It does not replace or modify the production
Monte Carlo estimator.

## Inputs and scope

For each event the workflow reads the persisted production clean-lap table,
compound-offset posterior medians, team/compound degradation posterior medians,
and team-baseline posterior samples. It does not load FastF1 or recalculate the
production model.

The diagnostic deliberately does not infer fuel, extrapolate observations to
tyre age zero, refit tyre parameters, or connect separate race windows. Results
therefore describe local contemporaneous support rather than a new whole-race
absolute pace estimate.

## Calculation

Laps are assigned to fixed windows of 3, 5, or 10 laps by default. Within each
window, a pairwise observation is created only when two teams have accepted
clean laps on the same race lap. The raw lap-time difference is corrected by
the difference between the two production tyre effects:

```text
tyre effect = compound offset + team/compound degradation * tyre age
corrected pairwise delta = raw delta - (tyre effect A - tyre effect B)
```

Multiple driver pairings on the same race lap are averaged before edge counts
and edge uncertainty are calculated, so a two-driver-versus-two-driver overlap
still contributes one independent race-lap observation rather than four.

Each window becomes a weighted team graph. Connected components are solved
independently. A component containing Mercedes is expressed relative to
Mercedes; another component remains relative only to its own deterministic
reference node. Teams may connect to Mercedes directly or through an indirect
same-window path. No information is propagated between windows.

The diagnostic also records edge residuals, graph standard errors, connected
components, and triangular loop-closure errors. Raw and tyre-corrected results
are retained together so model-induced movement is visible.

## Command

```bash
python -m wostrategy.script.local_pairwise_race_performance \
  --year 2026 \
  --first-round 1 \
  --last-round 13 \
  --window-sizes 3 5 10 \
  --input-root ../woData/wostrategy/race_performance_review/schema_v1 \
  --output-dir ../temp/local_pairwise_race_performance
```

The default output directory is temporary and is not intended for version
control.

## Outputs

The command writes:

- `local_pairwise_observations.csv`: exact-lap driver-pair observations;
- `local_pairwise_edges.csv`: team edges aggregated by independent race lap;
- `local_pairwise_team_windows.csv`: fitted team values and connectivity;
- `local_pairwise_components.csv`: component support and residual diagnostics;
- `local_pairwise_cycles.csv`: triangle loop-closure diagnostics;
- `local_pairwise_window_summary.csv`: per-window graph coverage;
- `local_pairwise_team_variation.csv`: within-event movement across windows;
- `local_pairwise_global_summary.csv`: aggregate diagnostics by window size;
- `local_pairwise_known_cases.csv`: focused comparisons for configured cases;
- `production_relative_medians.csv`: paired production posterior medians used as
  the whole-race comparison reference.

Large local movement, weak Mercedes connectivity, high residuals, or poor cycle
closure indicate that a local estimate is weak or internally inconsistent.
They are diagnostic warnings, not evidence that the production posterior should
automatically be replaced.
