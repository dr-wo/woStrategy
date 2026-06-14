from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from wostrategy.model.fuel_consumption import get_fuel_consumption_term_config
from wostrategy.model.track_evolution import (
    EXPONENTIAL_TRACK_EVOLUTION_MODEL,
    LINEAR_TRACK_EVOLUTION_MODEL,
    get_track_evolution_term_config,
)
from wostrategy.model.tyre_degragation import get_tyre_degradation_term_config

LINEAR_LONG_RUN_TERM = "linear"
EXPONENTIAL_LONG_RUN_TERM = "exponential"

DEFAULT_LONG_RUN_MODEL_CONFIG: dict[str, Any] = {
    "name": "linear_components",
    "terms": {
        "tyre": get_tyre_degradation_term_config(),
        "fuel": {
            **get_fuel_consumption_term_config(),
            "fit": False,
            "coefficient": 0.0,
        },
        "track": {
            **get_track_evolution_term_config(LINEAR_TRACK_EVOLUTION_MODEL),
            "fit": False,
            "coefficient": 0.0,
        },
    },
    "intercept": {"parameter": "intercept_seconds"},
}

EXPONENTIAL_TRACK_LONG_RUN_MODEL_CONFIG: dict[str, Any] = {
    "name": "linear_components_exponential_track",
    "terms": {
        "tyre": get_tyre_degradation_term_config(),
        "fuel": {
            **get_fuel_consumption_term_config(),
            "fit": False,
            "coefficient": 0.0,
        },
        "track": get_track_evolution_term_config(EXPONENTIAL_TRACK_EVOLUTION_MODEL),
    },
    "intercept": {"parameter": "intercept_seconds"},
}

LONG_RUN_MODEL_PRESETS: dict[str, dict[str, Any]] = {
    str(DEFAULT_LONG_RUN_MODEL_CONFIG["name"]): DEFAULT_LONG_RUN_MODEL_CONFIG,
    str(EXPONENTIAL_TRACK_LONG_RUN_MODEL_CONFIG["name"]): (
        EXPONENTIAL_TRACK_LONG_RUN_MODEL_CONFIG
    ),
}


@dataclass(frozen=True)
class LongRunLapTimeFit:
    model_name: str
    config: dict[str, Any]
    parameters: dict[str, float]
    x_columns: tuple[str, ...]
    y_column: str
    rmse_seconds: float
    formula: str

    def predict_frame(self, laps: pd.DataFrame) -> np.ndarray:
        _require_columns(laps, self.x_columns)
        prediction = np.zeros(len(laps), dtype="float64")
        for term_name, term_config in _iter_terms(self.config):
            prediction = prediction + _predict_term(
                laps,
                term_name=term_name,
                term_config=term_config,
                parameters=self.parameters,
            )
        intercept_config = self.config.get("intercept", {})
        intercept_parameter = intercept_config.get("parameter", "intercept_seconds")
        if intercept_parameter in self.parameters:
            prediction = prediction + self.parameters[intercept_parameter]
        return prediction

    def predict_values(self, values: dict[str, float]) -> float:
        frame = pd.DataFrame([{column: values[column] for column in self.x_columns}])
        return float(self.predict_frame(frame)[0])


def get_long_run_model_config(name: str) -> dict[str, Any]:
    normalized = name.strip()
    if normalized not in LONG_RUN_MODEL_PRESETS:
        options = ", ".join(sorted(LONG_RUN_MODEL_PRESETS))
        raise ValueError(f"Unknown long-run model preset {name!r}. Options: {options}")
    return _deepcopy_config(LONG_RUN_MODEL_PRESETS[normalized])


def fit_long_run_lap_time_model(
    laps: pd.DataFrame,
    *,
    model_config: dict[str, Any],
    y_column: str = "LapTimeSeconds",
) -> LongRunLapTimeFit:
    config = normalize_long_run_model_config(model_config)
    x_columns = tuple(_term_x_columns(config))
    _require_columns(laps, (*x_columns, y_column))
    fit_laps = laps.dropna(subset=[*x_columns, y_column])
    if len(fit_laps) < _minimum_points(config):
        raise ValueError(
            f"At least {_minimum_points(config)} laps are required to fit "
            f"{config['name']}."
        )

    y = fit_laps[y_column].to_numpy(dtype="float64")
    fixed_prediction = _fixed_term_prediction(fit_laps, config=config)
    parameters, fitted_adjusted = _fit_parameters(
        fit_laps,
        y=y - fixed_prediction,
        config=config,
    )
    fitted = fitted_adjusted + fixed_prediction
    rmse = float(np.sqrt(np.mean((y - fitted) ** 2)))
    return LongRunLapTimeFit(
        model_name=str(config["name"]),
        config=config,
        parameters=parameters,
        x_columns=x_columns,
        y_column=y_column,
        rmse_seconds=rmse,
        formula=_formula_label(config, parameters),
    )


def normalize_long_run_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    config = _deepcopy_config(model_config)
    if "name" not in config:
        config["name"] = "custom_long_run_model"
    if "terms" not in config or not isinstance(config["terms"], dict):
        raise ValueError("Long-run model config must define a 'terms' dictionary.")
    if not config["terms"]:
        raise ValueError("Long-run model config must include at least one term.")
    config.setdefault("intercept", {"parameter": "intercept_seconds"})

    for term_name, term_config in _iter_terms(config):
        if "model" not in term_config:
            raise ValueError(f"Term {term_name!r} must define a model.")
        if "x_column" not in term_config:
            raise ValueError(f"Term {term_name!r} must define an x_column.")
        term_config.setdefault("label", term_name)
        model = str(term_config["model"]).lower()
        term_config["model"] = model
        if model == LINEAR_LONG_RUN_TERM:
            term_config.setdefault("parameter", f"{term_name}_slope")
            term_config.setdefault("fit", True)
            if not bool(term_config["fit"]) and "coefficient" not in term_config:
                raise ValueError(
                    f"Fixed linear term {term_name!r} must define coefficient."
                )
        elif model == EXPONENTIAL_LONG_RUN_TERM:
            term_config.setdefault("amplitude_parameter", f"{term_name}_amplitude")
            term_config.setdefault("decay_parameter", f"{term_name}_decay_rate")
            term_config.setdefault("decay_grid_size", 400)
            term_config.setdefault("fit", True)
            if not bool(term_config["fit"]):
                missing = [
                    key
                    for key in ("amplitude", "decay_rate")
                    if key not in term_config
                ]
                if missing:
                    raise ValueError(
                        f"Fixed exponential term {term_name!r} must define "
                        f"{', '.join(missing)}."
                    )
        else:
            options = ", ".join([LINEAR_LONG_RUN_TERM, EXPONENTIAL_LONG_RUN_TERM])
            raise ValueError(
                f"Unknown model {term_config['model']!r} for term {term_name!r}. "
                f"Options: {options}"
            )
    return config


def reference_values_from_config(
    model_config: dict[str, Any],
    *,
    fallback_values: dict[str, float],
) -> dict[str, float]:
    config = normalize_long_run_model_config(model_config)
    values: dict[str, float] = {}
    for _, term_config in _iter_terms(config):
        x_column = str(term_config["x_column"])
        if "reference_value" in term_config:
            values[x_column] = float(term_config["reference_value"])
        elif x_column in fallback_values:
            values[x_column] = float(fallback_values[x_column])
        else:
            raise ValueError(
                f"No reference value configured or inferred for column {x_column!r}."
            )
    return values


def _fit_parameters(
    laps: pd.DataFrame,
    *,
    y: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, float], np.ndarray]:
    exponential_terms = [
        (term_name, term_config)
        for term_name, term_config in _iter_terms(config)
        if (
            term_config["model"] == EXPONENTIAL_LONG_RUN_TERM
            and bool(term_config.get("fit", True))
        )
    ]
    decay_candidates = [
        _decay_grid(laps[str(term_config["x_column"])], term_config=term_config)
        for _, term_config in exponential_terms
    ]
    if not decay_candidates:
        decay_combinations = [()]
    else:
        decay_combinations = product(*decay_candidates)

    best_error = float("inf")
    best_parameters: dict[str, float] | None = None
    best_fitted: np.ndarray | None = None
    for decay_values in decay_combinations:
        decay_by_term = {
            term_name: float(decay)
            for (term_name, _), decay in zip(exponential_terms, decay_values)
        }
        design, parameter_names = _design_matrix(
            laps,
            config=config,
            decay_by_term=decay_by_term,
        )
        try:
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        fitted = design @ coefficients
        error = float(np.mean((y - fitted) ** 2))
        if error >= best_error:
            continue
        parameters = {
            parameter_name: float(coefficient)
            for parameter_name, coefficient in zip(parameter_names, coefficients)
        }
        for term_name, term_config in exponential_terms:
            decay_parameter = str(term_config["decay_parameter"])
            parameters[decay_parameter] = decay_by_term[term_name]
        best_error = error
        best_parameters = parameters
        best_fitted = fitted

    if best_parameters is None or best_fitted is None:
        raise ValueError(f"Could not fit long-run model {config['name']!r}.")
    best_parameters.update(_fixed_parameters(config))
    return best_parameters, best_fitted


def _fixed_parameters(config: dict[str, Any]) -> dict[str, float]:
    parameters: dict[str, float] = {}
    for _, term_config in _iter_terms(config):
        if bool(term_config.get("fit", True)):
            continue
        if term_config["model"] == LINEAR_LONG_RUN_TERM:
            parameters[str(term_config["parameter"])] = float(term_config["coefficient"])
        elif term_config["model"] == EXPONENTIAL_LONG_RUN_TERM:
            parameters[str(term_config["amplitude_parameter"])] = float(
                term_config["amplitude"]
            )
            parameters[str(term_config["decay_parameter"])] = float(
                term_config["decay_rate"]
            )
    return parameters


def _design_matrix(
    laps: pd.DataFrame,
    *,
    config: dict[str, Any],
    decay_by_term: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    parameter_names: list[str] = []
    for term_name, term_config in _iter_terms(config):
        x = laps[str(term_config["x_column"])].to_numpy(dtype="float64")
        if term_config["model"] == LINEAR_LONG_RUN_TERM:
            if not bool(term_config.get("fit", True)):
                continue
            columns.append(x)
            parameter_names.append(str(term_config["parameter"]))
        elif term_config["model"] == EXPONENTIAL_LONG_RUN_TERM:
            if not bool(term_config.get("fit", True)):
                continue
            decay_rate = decay_by_term[term_name]
            columns.append(np.exp(-decay_rate * x))
            parameter_names.append(str(term_config["amplitude_parameter"]))
    intercept_config = config.get("intercept")
    if intercept_config:
        columns.append(np.ones(len(laps), dtype="float64"))
        parameter_names.append(str(intercept_config.get("parameter", "intercept_seconds")))
    if not columns:
        raise ValueError("Long-run model must include at least one fitted parameter.")
    design = np.column_stack(columns)
    rank = int(np.linalg.matrix_rank(design))
    if rank < len(parameter_names):
        names = ", ".join(parameter_names)
        raise ValueError(
            "Long-run model is underdetermined for this lap chunk. "
            f"Fitted parameters are linearly dependent: {names}. "
            "Fix one or more component coefficients in the model config, or fit "
            "only the combined stint slope."
        )
    return design, parameter_names


def _predict_term(
    laps: pd.DataFrame,
    *,
    term_name: str,
    term_config: dict[str, Any],
    parameters: dict[str, float],
) -> np.ndarray:
    x = laps[str(term_config["x_column"])].to_numpy(dtype="float64")
    if term_config["model"] == LINEAR_LONG_RUN_TERM:
        return parameters[str(term_config["parameter"])] * x
    if term_config["model"] == EXPONENTIAL_LONG_RUN_TERM:
        amplitude = parameters[str(term_config["amplitude_parameter"])]
        decay_rate = parameters[str(term_config["decay_parameter"])]
        return amplitude * np.exp(-decay_rate * x)
    raise ValueError(f"Unknown term model for {term_name!r}: {term_config['model']!r}")


def _fixed_term_prediction(
    laps: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> np.ndarray:
    prediction = np.zeros(len(laps), dtype="float64")
    for _, term_config in _iter_terms(config):
        if bool(term_config.get("fit", True)):
            continue
        x = laps[str(term_config["x_column"])].to_numpy(dtype="float64")
        if term_config["model"] == LINEAR_LONG_RUN_TERM:
            prediction = prediction + (float(term_config["coefficient"]) * x)
        elif term_config["model"] == EXPONENTIAL_LONG_RUN_TERM:
            prediction = prediction + (
                float(term_config["amplitude"])
                * np.exp(-float(term_config["decay_rate"]) * x)
            )
    return prediction


def _formula_label(config: dict[str, Any], parameters: dict[str, float]) -> str:
    parts = []
    for term_name, term_config in _iter_terms(config):
        label = str(term_config["label"])
        if term_config["model"] == LINEAR_LONG_RUN_TERM:
            parameter = str(term_config["parameter"])
            fixed = " fixed" if not bool(term_config.get("fit", True)) else ""
            parts.append(f"{parameters[parameter]:.5f}*{label}{fixed}")
        elif term_config["model"] == EXPONENTIAL_LONG_RUN_TERM:
            amplitude = str(term_config["amplitude_parameter"])
            decay = str(term_config["decay_parameter"])
            fixed = " fixed" if not bool(term_config.get("fit", True)) else ""
            parts.append(
                f"{parameters[amplitude]:.5f}*exp(-{parameters[decay]:.6f}*{label})"
                f"{fixed}"
            )
        else:
            raise ValueError(f"Unknown term model for {term_name!r}.")
    intercept_config = config.get("intercept")
    if intercept_config:
        intercept_parameter = str(intercept_config.get("parameter", "intercept_seconds"))
        parts.append(f"{parameters[intercept_parameter]:.5f}")
    return "y = " + " + ".join(parts)


def _decay_grid(
    values: pd.Series,
    *,
    term_config: dict[str, Any],
) -> np.ndarray:
    x = values.to_numpy(dtype="float64")
    x_span = float(np.nanmax(x) - np.nanmin(x))
    if x_span <= 0:
        raise ValueError("Cannot fit exponential term with zero x span.")
    grid_size = int(term_config.get("decay_grid_size", 400))
    lower_k = float(term_config.get("min_decay_rate", 1e-6 / x_span))
    upper_k = float(term_config.get("max_decay_rate", 10.0 / x_span))
    return np.geomspace(lower_k, upper_k, num=grid_size)


def _minimum_points(config: dict[str, Any]) -> int:
    parameter_count = sum(
        1
        for _, term_config in _iter_terms(config)
        if bool(term_config.get("fit", True))
    )
    if config.get("intercept"):
        parameter_count += 1
    parameter_count += sum(
        1
        for _, term_config in _iter_terms(config)
        if (
            term_config["model"] == EXPONENTIAL_LONG_RUN_TERM
            and bool(term_config.get("fit", True))
        )
    )
    return parameter_count


def _term_x_columns(config: dict[str, Any]) -> list[str]:
    return [str(term_config["x_column"]) for _, term_config in _iter_terms(config)]


def _iter_terms(config: dict[str, Any]):
    yield from config["terms"].items()


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Laps are missing required columns: {', '.join(sorted(missing))}")


def _deepcopy_config(config: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            copied[key] = _deepcopy_config(value)
        else:
            copied[key] = value
    return copied


__all__ = [
    "DEFAULT_LONG_RUN_MODEL_CONFIG",
    "EXPONENTIAL_LONG_RUN_TERM",
    "EXPONENTIAL_TRACK_LONG_RUN_MODEL_CONFIG",
    "LINEAR_LONG_RUN_TERM",
    "LONG_RUN_MODEL_PRESETS",
    "LongRunLapTimeFit",
    "fit_long_run_lap_time_model",
    "get_long_run_model_config",
    "normalize_long_run_model_config",
    "reference_values_from_config",
]
