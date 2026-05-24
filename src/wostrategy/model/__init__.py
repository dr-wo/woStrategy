"""Mathematical model implementations used by analysis workflows."""

from .fuel_consumption import FixedRateFuelCorrection, FuelCorrection
from .track_evolution import (
    EXPONENTIAL_TRACK_EVOLUTION_MODEL,
    LINEAR_TRACK_EVOLUTION_MODEL,
    TRACK_EVO_CORRECTED_LAP_TIME,
    TRACK_EVO_CORRECTED_LAP_TIME_SECONDS,
    TRACK_EVO_CORRECTION_SECONDS,
    TRACK_EVOLUTION_FIT_MODEL,
    TRACK_EVOLUTION_SECONDS_PER_LAP,
    ExponentialTrackEvolutionModel,
    LinearTrackEvolutionModel,
    TrackEvolutionFit,
    TrackEvolutionModel,
    add_track_evolution_correction,
    dominant_compound,
    fit_compound_track_evolution,
    get_track_evolution_model,
)

__all__ = [
    "EXPONENTIAL_TRACK_EVOLUTION_MODEL",
    "LINEAR_TRACK_EVOLUTION_MODEL",
    "TRACK_EVO_CORRECTED_LAP_TIME",
    "TRACK_EVO_CORRECTED_LAP_TIME_SECONDS",
    "TRACK_EVO_CORRECTION_SECONDS",
    "TRACK_EVOLUTION_FIT_MODEL",
    "TRACK_EVOLUTION_SECONDS_PER_LAP",
    "ExponentialTrackEvolutionModel",
    "FixedRateFuelCorrection",
    "FuelCorrection",
    "LinearTrackEvolutionModel",
    "TrackEvolutionFit",
    "TrackEvolutionModel",
    "add_track_evolution_correction",
    "dominant_compound",
    "fit_compound_track_evolution",
    "get_track_evolution_model",
]
