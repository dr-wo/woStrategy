from __future__ import annotations

import numpy as np

from wostrategy.algorithm.sampling import (
    HALTON_SAMPLER,
    LATIN_HYPERCUBE_SAMPLER,
    RANDOM_SAMPLER,
    get_unit_cube_sampler,
)


def test_random_sampler_is_seed_reproducible():
    sampler = get_unit_cube_sampler(RANDOM_SAMPLER)

    first = sampler.sample(sample_count=4, dimension_count=3, seed=1)
    second = sampler.sample(sample_count=4, dimension_count=3, seed=1)

    assert first.shape == (4, 3)
    assert np.allclose(first, second)
    assert ((first >= 0.0) & (first < 1.0)).all()


def test_latin_hypercube_sampler_uses_each_bin_once_per_dimension():
    sampler = get_unit_cube_sampler(LATIN_HYPERCUBE_SAMPLER)

    samples = sampler.sample(sample_count=8, dimension_count=3, seed=2)
    bins = np.floor(samples * 8).astype(int)

    assert samples.shape == (8, 3)
    for dimension in range(3):
        assert sorted(bins[:, dimension].tolist()) == list(range(8))


def test_halton_sampler_starts_with_expected_low_discrepancy_values():
    sampler = get_unit_cube_sampler(HALTON_SAMPLER)

    samples = sampler.sample(sample_count=3, dimension_count=2, seed=99)

    expected = np.array(
        [
            [0.5, 1.0 / 3.0],
            [0.25, 2.0 / 3.0],
            [0.75, 1.0 / 9.0],
        ]
    )
    assert np.allclose(samples, expected)
