"""Vectorised Weir & Cockerham Fst and ``save_plots`` behaviour."""

import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import Mock

import numpy as np
import pytest

from snpio import Plotting
from snpio.popgenstats import fst_distance
from snpio.popgenstats.fst_distance import FstDistance


@pytest.fixture(scope="module")
def genotypes():
    """Mixed encodings, multi-allelic sites, all missing spellings, tiny pops."""
    rng = np.random.default_rng(7)
    vals = ["A", "C", "G", "T", "R", "Y", "S", "W", "K", "M", "A/C", "C|T",
            "G/G", "T/A", "N", "-", "?", ".", "NA", "", None, "a/g"]
    mat = rng.choice(np.array(vals, dtype=object), size=(40, 400))
    mat[:, :20] = "A"                        # monomorphic block
    mat[rng.random(mat.shape) < 0.2] = "N"   # extra missingness
    mat[:3, 100:120] = "N"                   # one population entirely missing
    pops = {"p0": np.arange(0, 3), "p1": np.arange(3, 5),
            "p2": np.arange(5, 20), "p3": np.arange(20, 40)}
    return mat, pops


def test_vectorised_components_match_reference(genotypes):
    mat, pops = genotypes
    encoded = FstDistance.encode_genotypes(mat)
    for p, q in itertools.combinations(sorted(pops), 2):
        new = FstDistance._fst_variance_components_per_locus(pops[p], pops[q], encoded)
        ref = FstDistance._fst_variance_components_per_locus_python(pops[p], pops[q], mat)
        for a, b in zip(new, ref):
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14, equal_nan=True)


def test_multilocus_fst_accepts_raw_or_encoded(genotypes):
    mat, pops = genotypes
    encoded = FstDistance.encode_genotypes(mat)
    raw = FstDistance._compute_multilocus_fst(pops["p2"], pops["p3"], mat)
    enc = FstDistance._compute_multilocus_fst(pops["p2"], pops["p3"], encoded)
    assert raw == pytest.approx(enc, rel=1e-12)


def test_pool_task_matches_serial_worker(genotypes):
    mat, pops = genotypes
    encoded = FstDistance.encode_genotypes(mat)
    state = {"pop_indices": pops, "encoded": encoded}
    serial = FstDistance._permutation_worker(("p2", "p3"), pops, encoded, 5, 123)

    with ProcessPoolExecutor(
        max_workers=2, initializer=fst_distance._set_worker_state, initargs=(state,)
    ) as pool:
        pooled = pool.submit(fst_distance._permutation_task, ("p2", "p3"), 5, 123).result()

    assert pooled[0] == serial[0]
    assert pooled[1]["fst"] == serial[1]["fst"]
    assert pooled[1]["pvalue"] == serial[1]["pvalue"]
    np.testing.assert_array_equal(pooled[1]["perm_dist"], serial[1]["perm_dist"])


def _bare_plotting(tmp_path, save_plots):
    plotting = object.__new__(Plotting)
    plotting.output_dir_analysis = tmp_path / "plots"
    plotting.report_dir_analysis = tmp_path / "reports"
    plotting.plot_format = "png"
    plotting.show = False
    plotting.save_plots = save_plots
    plotting.snpio_mqc = Mock()
    plotting.logger = Mock()
    return plotting


@pytest.mark.parametrize("save_plots", [True, False])
def test_permutation_dist_exports_data_without_plots(tmp_path, save_plots):
    plotting = _bare_plotting(tmp_path, save_plots)
    plotting.plot_permutation_dist(0.1, np.array([0.01, 0.02, 0.2]), "a", "b")

    exported = list((tmp_path / "reports").rglob("*.json"))
    assert len(exported) == 1
    assert json.loads(exported[0].read_text())["Observed Fst"] == 0.1
    images = list(tmp_path.rglob("*.png"))
    assert bool(images) is save_plots
