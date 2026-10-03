"""Position traffic shadow. No result from this module is clean-lap evidence.

All time arguments use one explicitly chosen clock. For live Position.z that is
normally UTC epoch seconds; replay adapters must map reference times to their
session clock. Geometry is supplied, never inferred from the analysed Race.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PositionReference:
    points_m: tuple[tuple[float, float], ...]
    coordinate_frame: str
    source: str
    source_sha256: str
    observed_until: float
    available_from: float
    availability_basis: str
    raw_unit_to_m: float = 0.1

    def __post_init__(self):
        points = np.asarray(self.points_m, float)
        if (len(points) < 4 or points.shape != (len(points), 2)
                or not np.isfinite(points).all() or not np.array_equal(points[0], points[-1])):
            raise ValueError("reference must be a finite, explicitly closed, directed loop")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0):
            raise ValueError("reference contains a zero-length segment")
        if not all((self.coordinate_frame, self.source, self.source_sha256, self.availability_basis)):
            raise ValueError("reference provenance and availability are required")
        if (not np.isfinite([self.observed_until, self.available_from, self.raw_unit_to_m]).all()
                or self.raw_unit_to_m <= 0 or self.available_from < self.observed_until):
            raise ValueError("invalid reference chronology/units")


@dataclass(frozen=True)
class PositionQuality:
    # Shadow quality controls, not ahead/rear filtering thresholds.
    maximum_cross_track_m: float = 20.0
    maximum_bracket_s: float = 1.0
    maximum_passage_age_s: float = 10.0
    maximum_segment_speed_mps: float = 120.0

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not np.isfinite(values).all() or any(value <= 0 for value in values):
            raise ValueError("position quality limits must be finite and positive")


@dataclass(frozen=True)
class PositionShadow:
    samples: pd.DataFrame
    laps: pd.DataFrame
    status: str
    reason: str
    reference: PositionReference | None


def _project(points, reference):
    """Nearest segment projection, as in woData PitLaneGeometry.project.

    Vectorised here for all-car history. No pit-lane geometry is substituted for
    the circuit; separated, similarly close branches are explicitly ambiguous.
    """
    xy = np.asarray(reference.points_m, float)
    segment = np.diff(xy, axis=0)
    lengths = np.linalg.norm(segment, axis=1)
    starts = np.r_[0, lengths.cumsum()[:-1]]
    length = lengths.sum()
    result = []
    for chunk in np.array_split(points, max(1, int(np.ceil(len(points) / 2000)))):
        rel = chunk[:, None, :] - xy[None, :-1, :]
        frac = np.clip(np.einsum('ijk,jk->ij', rel, segment) / lengths**2, 0, 1)
        distances = np.linalg.norm(rel - frac[:, :, None] * segment, axis=2)
        phases = (starts + frac * lengths) % length
        best = distances.argmin(axis=1)
        ix = np.arange(len(chunk))
        phase = phases[ix, best]
        separation = abs((phases - phase[:, None] + length / 2) % length - length / 2)
        ambiguous = ((separation > 50) & (distances <= distances[ix, best, None] + 1)).any(axis=1)
        result.append(np.column_stack((phase, distances[ix, best], ambiguous)))
    return np.concatenate(result), float(length)


def _passage(history, phase, time, length, quality):
    """Past leader passage at the follower's current place, not gap/speed.

    Both bracketing observations must already be at/before the query time.
    Reject reverse/jump/long brackets; never extrapolate from a single sample.
    """
    for left, right in reversed(list(zip(history, list(history)[1:]))):
        dt = right[0] - left[0]
        ds = (right[1] - left[1] + length / 2) % length - length / 2
        goal = (phase - left[1]) % length
        if 0 < ds and goal <= ds:
            if dt <= 0 or dt > quality.maximum_bracket_s or ds / dt > quality.maximum_segment_speed_mps:
                return np.nan, "invalid_passage_bracket"
            crossing = left[0] + goal / ds * dt
            age = time - crossing
            if not 0 <= age <= quality.maximum_passage_age_s:
                return np.nan, "passage_outside_history"
            return age, "diagnostic_only"
    return np.nan, "insufficient_passage_history"


def position_traffic_shadow(
    positions: pd.DataFrame,
    *,
    reference: PositionReference | None,
    coordinate_frame: str,
    cutoff: float,
    expected_drivers: set[str],
    targets: set[str] | None = None,
    lap_windows: pd.DataFrame | None = None,
    quality: PositionQuality = PositionQuality(),
) -> PositionShadow:
    """Evaluate exact common-time observations from an ingress-bounded snapshot.

    Required columns: Driver, SessionTime, X, Y, Z, OnTrack. Do not carry stale
    cars forward: an incomplete common-time field cannot establish the nearest
    neighbour. Partial-field rows expose candidates among available cars only.
    Explicit off-track cars are excluded, unknown routes invalidate
    completeness. Lapped cars remain physical candidates. InPit, if supplied,
    excludes a car even if Position.z claims OnTrack. Missing pit context remains
    an explicit limitation of this shadow, never permission to select clean laps.

    Optional lap windows use the target's own (Start, End] publication interval.
    Boundary source must be named; these labels do not calibrate a finish line.
    """
    reason = None
    if reference is None:
        reason = "reference_unavailable"
    elif reference.coordinate_frame != coordinate_frame:
        reason = "reference_frame_mismatch"
    elif reference.observed_until > cutoff or reference.available_from > cutoff:
        reason = "reference_not_available_at_cut"
    if reason:
        return PositionShadow(pd.DataFrame(), pd.DataFrame(), "unavailable", reason, reference)
    if not np.isfinite(cutoff) or not expected_drivers:
        raise ValueError("finite cutoff and explicit expected field required")
    data = positions.copy()
    if data.empty:
        return PositionShadow(data, pd.DataFrame(), "unavailable", "position_history_missing", reference)
    data["Driver"] = data.Driver.astype(str)
    data = data.loc[data.SessionTime.le(cutoff)].sort_values("SessionTime", kind="stable")
    if data.empty:
        return PositionShadow(data, pd.DataFrame(), "unavailable", "no_positions_at_cut", reference)
    duplicate = data.duplicated(["Driver", "SessionTime"], keep=False)
    valid = np.isfinite(data[["SessionTime", "X", "Y", "Z"]]).all(axis=1) & data[["X", "Y", "Z"]].ne(0).any(axis=1)
    data["Phase"] = np.nan
    data["CrossTrack"] = np.nan
    data["Ambiguous"] = False
    if valid.any():
        projected, length = _project(data.loc[valid, ["X", "Y"]].to_numpy() * reference.raw_unit_to_m, reference)
        data.loc[valid, "Phase"] = projected[:, 0]
        data.loc[valid, "CrossTrack"] = projected[:, 1]
        data.loc[valid, "Ambiguous"] = projected[:, 2].astype(bool)
    else:
        return PositionShadow(data, pd.DataFrame(), "unavailable", "no_valid_coordinates", reference)
    data["Valid"] = valid & ~duplicate & ~data.Ambiguous & data.CrossTrack.le(quality.maximum_cross_track_m)
    history = {}
    rows = []
    for time, frame in data.groupby("SessionTime", sort=False):
        seen = set(frame.Driver)
        excluded = frame.OnTrack.eq(False)
        if "InPit" in frame:
            excluded |= frame.InPit.eq(True)
        usable = frame.Valid & frame.OnTrack.eq(True) & ~excluded
        complete = expected_drivers.issubset(seen) and bool((usable | excluded).all())
        active = frame.loc[usable].drop_duplicates("Driver").set_index("Driver")
        # Co-located cars are common in placeholders; never invent an ordering.
        coincident = active.Phase.duplicated(keep=False).any()
        complete = complete and not coincident
        for driver in seen:
            hist = history.setdefault(driver, deque())
            while hist and hist[0][0] < time - quality.maximum_passage_age_s - quality.maximum_bracket_s:
                hist.popleft()
            if driver in active.index:
                hist.append((time, float(active.loc[driver, "Phase"])))
            else:
                hist.clear()  # do not bridge an observed invalid/pit interval
        for driver in sorted(targets if targets is not None else expected_drivers):
            row = dict(Driver=driver, SessionTime=time, NeighbourStatus="unavailable",
                       Reason="incomplete_or_invalid_field", DriverAhead=None, DriverBehind=None,
                       RearPassageSeconds=np.nan, AheadPassageSeconds=np.nan,
                       RearPassageStatus="unavailable", AheadPassageStatus="unavailable",
                       PitContext="explicit" if "InPit" in frame else "position_status_only",
                       TrafficStatus="unknown", EvidenceLevel="position_shadow")
            row["FieldComplete"] = complete
            row["UnavailableDrivers"] = ",".join(sorted((expected_drivers - seen) | set(frame.loc[~(usable | excluded), "Driver"])))
            if not coincident and driver in active.index and len(active) >= 2:
                phase = float(active.loc[driver, "Phase"])
                others = active.drop(driver)
                ahead = (others.Phase - phase) % length
                rear = (phase - others.Phase) % length
                a, r = ahead.idxmin(), rear.idxmin()
                row.update(NeighbourStatus="common_time" if complete else "partial_field",
                           Reason="" if complete else "nearest_among_available_cars_only", DriverAhead=a, DriverBehind=r,
                           DistanceAheadM=float(ahead[a]), DistanceBehindM=float(rear[r]), Phase=phase)
                row["AheadPassageSeconds"], row["AheadPassageStatus"] = _passage(
                    history[a], phase, time, length, quality)
                row["RearPassageSeconds"], row["RearPassageStatus"] = _passage(
                    history[driver], float(active.loc[r, "Phase"]), time, length, quality)
            rows.append(row)
    samples = pd.DataFrame(rows)
    samples["LapNumber"] = np.nan
    samples["LapBoundaryBasis"] = "unavailable"
    if lap_windows is not None:
        for window in lap_windows.itertuples(index=False):
            member = samples.Driver.eq(str(window.Driver)) & samples.SessionTime.gt(window.Start) & samples.SessionTime.le(window.End)
            if samples.loc[member, "LapNumber"].notna().any():
                raise ValueError("overlapping target lap windows")
            samples.loc[member, "LapNumber"] = window.LapNumber
            samples.loc[member, "LapBoundaryBasis"] = window.Basis
    laps = samples.dropna(subset=["LapNumber"]).groupby(["Driver", "LapNumber"], as_index=False).agg(
        PositionSamples=("SessionTime", "size"),
        RearPassageSamples=("RearPassageSeconds", "count"),
        MeanDiagnosticRearPassageSeconds=("RearPassageSeconds", "mean"),
        MinDiagnosticRearPassageSeconds=("RearPassageSeconds", "min"),
    )
    laps["TrafficStatus"] = "unknown"
    laps["EvidenceLevel"] = "position_shadow"
    return PositionShadow(samples, laps, "diagnostic_only", "gap_decisions_not_validated", reference)
