"""Seeded permutation and bootstrap replicates are reproducible."""

import numpy as np
import pandas as pd
import pytest

from snpio import PhylipReader, PopGenStatistics
from tests.test_pop_gen_statistics import (
    generate_phylip_file,
    generate_population_map_file,
)


@pytest.fixture(scope="module")
def popgen(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("seed") / "seed"
    genotype_data = PhylipReader(
        filename=generate_phylip_file(
            num_samples=30, num_loci=100, default_rng=np.random.default_rng(42)
        ),
        popmapfile=generate_population_map_file(),
        prefix=str(prefix),
        save_plots=False,
    )
    return PopGenStatistics(genotype_data)


def assert_same(a, b):
    """Recursive equality for the nested results (dicts, frames, arrays)."""
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b)
    elif isinstance(a, (np.ndarray, list, tuple)):
        np.testing.assert_array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object))
    else:
        assert a == b or (pd.isna(a) and pd.isna(b))


@pytest.mark.parametrize("method", ["permutation", "bootstrap"])
def test_summary_statistics_seed(popgen, method):
    run = lambda seed, n_jobs=1: popgen.summary_statistics(
        method=method, n_reps=20, n_jobs=n_jobs, save_plots=False,
        include_nei=True, seed=seed,
    )[0]
    first = run(5)
    assert_same(first, run(5))
    assert_same(first, run(5, n_jobs=2))


def test_fst_distance_seed(popgen):
    run = lambda seed: popgen.fst_distance(
        method="bootstrap", n_reps=20, suppress_plot=True, seed=seed
    )
    first = run(5)
    assert_same(first, run(5))
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(first["lower_ci"], run(6)["lower_ci"])


def test_neis_genetic_distance_seed(popgen):
    run = lambda seed: popgen.neis_genetic_distance(
        method="bootstrap", n_reps=20, suppress_plot=True, seed=seed
    )
    first = run(5)
    assert_same(first, run(5))
    with pytest.raises(AssertionError):
        assert_same(first, run(6))
