from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

RANDOM_SAMPLER = "random"
LATIN_HYPERCUBE_SAMPLER = "latin-hypercube"
HALTON_SAMPLER = "halton"
SAMPLING_STRATEGIES = (RANDOM_SAMPLER, LATIN_HYPERCUBE_SAMPLER, HALTON_SAMPLER)


class UnitCubeSampler(Protocol):
    name: str

    def sample(
        self,
        *,
        sample_count: int,
        dimension_count: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Return a sample_count x dimension_count array with values in [0, 1)."""


@dataclass(frozen=True)
class RandomSampler:
    name: str = RANDOM_SAMPLER

    def sample(
        self,
        *,
        sample_count: int,
        dimension_count: int,
        seed: int | None = None,
    ) -> np.ndarray:
        _validate_sample_shape(sample_count=sample_count, dimension_count=dimension_count)
        rng = np.random.default_rng(seed)
        return rng.random((sample_count, dimension_count), dtype="float64")


@dataclass(frozen=True)
class LatinHypercubeSampler:
    name: str = LATIN_HYPERCUBE_SAMPLER

    def sample(
        self,
        *,
        sample_count: int,
        dimension_count: int,
        seed: int | None = None,
    ) -> np.ndarray:
        _validate_sample_shape(sample_count=sample_count, dimension_count=dimension_count)
        rng = np.random.default_rng(seed)
        samples = np.empty((sample_count, dimension_count), dtype="float64")
        for dimension in range(dimension_count):
            permutation = rng.permutation(sample_count)
            jitter = rng.random(sample_count, dtype="float64")
            samples[:, dimension] = (permutation + jitter) / sample_count
        return samples


@dataclass(frozen=True)
class HaltonSampler:
    name: str = HALTON_SAMPLER

    def sample(
        self,
        *,
        sample_count: int,
        dimension_count: int,
        seed: int | None = None,
    ) -> np.ndarray:
        del seed
        _validate_sample_shape(sample_count=sample_count, dimension_count=dimension_count)
        bases = _first_primes(dimension_count)
        samples = np.empty((sample_count, dimension_count), dtype="float64")
        for dimension, base in enumerate(bases):
            samples[:, dimension] = [
                _radical_inverse(index, base) for index in range(1, sample_count + 1)
            ]
        return samples


def get_unit_cube_sampler(name: str) -> UnitCubeSampler:
    normalized = name.strip().lower().replace("_", "-")
    samplers: dict[str, UnitCubeSampler] = {
        RANDOM_SAMPLER: RandomSampler(),
        LATIN_HYPERCUBE_SAMPLER: LatinHypercubeSampler(),
        HALTON_SAMPLER: HaltonSampler(),
    }
    try:
        return samplers[normalized]
    except KeyError as exc:
        options = ", ".join(SAMPLING_STRATEGIES)
        raise ValueError(f"Unknown sampling strategy {name!r}. Options: {options}") from exc


def scale_unit_sample(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = float(bounds[0]), float(bounds[1])
    return lower + (float(value) * (upper - lower))


def _validate_sample_shape(*, sample_count: int, dimension_count: int) -> None:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if dimension_count <= 0:
        raise ValueError("dimension_count must be positive.")


def _radical_inverse(index: int, base: int) -> float:
    inverse = 0.0
    fraction = 1.0 / base
    while index > 0:
        inverse += fraction * (index % base)
        index //= base
        fraction /= base
    return inverse


def _first_primes(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if _is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    factor = 3
    while factor * factor <= value:
        if value % factor == 0:
            return False
        factor += 2
    return True
