"""Boolean-mask missingness and the per-locus violin/histogram switch."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from snpio import PhylipReader
from snpio.read_input import genotype_data as gd_module
from snpio.read_input.genotype_data import MAX_VIOLIN_LOCI, GenotypeData


@pytest.fixture(scope="module")
def phylip(tmp_path_factory):
    """12 samples x 60 sites with every missing symbol and uneven populations."""
    rng = np.random.default_rng(3)
    seqs = rng.choice(list("ACGTRYSWKM"), size=(12, 60))
    seqs[rng.random(seqs.shape) < 0.25] = "N"
    seqs[0, :5] = ["-", "?", ".", "N", "-"]
    seqs[:, 10] = "N"  # a locus missing in every sample
    seqs[1, 20:] = "?"  # one heavily missing sample
    d = tmp_path_factory.mktemp("miss")
    names = [f"S{i}" for i in range(12)]
    (d / "aln.phy").write_text(
        "12 60\n" + "".join(f"{n}\t{''.join(s)}\n" for n, s in zip(names, seqs))
    )
    pops = ["pA"] * 2 + ["pB"] * 9 + ["pC"]
    (d / "aln.popmap").write_text("".join(f"{n}\t{p}\n" for n, p in zip(names, pops)))
    return PhylipReader(
        filename=str(d / "aln.phy"),
        popmapfile=str(d / "aln.popmap"),
        prefix=str(d / "out"),
        save_plots=False,
    )


@pytest.mark.parametrize("use_pops", [True, False])
def test_mask_matches_na_frame(phylip, use_pops):
    snp = phylip.snp_data
    na_frame = pd.DataFrame(snp).replace(to_replace=phylip.missing_vals, value=pd.NA)
    mask = pd.DataFrame(np.isin(snp, phylip.missing_vals))
    old = phylip.calc_missing(na_frame, use_pops=use_pops)
    new = phylip.calc_missing(mask, use_pops=use_pops)

    for field in ("per_locus", "per_individual", "per_population"):
        a, b = getattr(old, field), getattr(new, field)
        assert (a is None) == (b is None)
        if a is not None:
            assert_series_equal(a, b, check_exact=True)
    for field in ("per_population_locus", "per_individual_population"):
        a, b = getattr(old, field), getattr(new, field)
        assert (a is None) == (b is None)
        if a is not None:
            assert_frame_equal(a, b, check_exact=True)
    assert_frame_equal(old.summary(), new.summary(), check_exact=True)
    assert new.per_locus.max() == 1.0  # the all-missing locus


def _bare_genotype_data():
    gd = object.__new__(GenotypeData)
    gd.snpio_mqc = Mock()
    return gd


def _violin_kwargs():
    return dict(
        panel_id="locus_missingness",
        section="missing_data",
        title="Per-locus Missingness",
        description="violin",
        index_label="Locus ID",
        pconfig={"id": "locus_missingness"},
    )


def test_violin_up_to_cutoff(tmp_path):
    gd = _bare_genotype_data()
    df = pd.DataFrame({"Percent Missing": np.zeros(MAX_VIOLIN_LOCI)})
    out = tmp_path / "loci.tsv.gz"
    gd._queue_locus_distribution(
        df=df, export_path=out, description_prefix="", **_violin_kwargs()
    )
    gd.snpio_mqc.queue_violin.assert_called_once()
    assert gd.snpio_mqc.queue_violin.call_args.kwargs["df"] is df
    gd.snpio_mqc.queue_linegraph.assert_not_called()
    assert not out.exists()


def test_histogram_above_cutoff(tmp_path):
    gd = _bare_genotype_data()
    n = MAX_VIOLIN_LOCI + 1
    rng = np.random.default_rng(5)
    values = rng.integers(0, 401, size=(n, 2)) / 400 * 100
    values[:3, 0] = [0.1 * 100, 100.0, 0.0]  # float noise on an edge, both ends
    values[3, 1] = np.nan
    df = pd.DataFrame(values, columns=["pA", "pB"], index=[f"l{i}" for i in range(n)])
    out = tmp_path / "loci.tsv.gz"
    gd._queue_locus_distribution(
        df=df, export_path=out, description_prefix="Pre: ", **_violin_kwargs()
    )

    gd.snpio_mqc.queue_violin.assert_not_called()
    data = gd.snpio_mqc.queue_linegraph.call_args.args[0]
    assert set(data) == {"pA", "pB"}
    assert sum(data["pA"].values()) == n
    assert sum(data["pB"].values()) == n - 1  # NaN is not counted
    assert len(data["pA"]) == 50
    first = pd.Series(data["pA"])
    expected = np.histogram(df["pA"].round(6), bins=np.linspace(0, 100, 51))[0]
    assert first.to_numpy().tolist() == expected.tolist()

    written = pd.read_csv(out, sep="\t", index_col="Locus ID")
    assert list(written.columns) == ["pA", "pB"]
    np.testing.assert_allclose(written.to_numpy(), df.round(2).to_numpy(), equal_nan=True)


@pytest.mark.parametrize(
    "column, series_name, expected",
    [("Percent Missing", "All loci", "All loci"), ("pA", None, "pA")],
)
def test_single_column_series_label(tmp_path, monkeypatch, column, series_name, expected):
    monkeypatch.setattr(gd_module, "MAX_VIOLIN_LOCI", 2)
    gd = _bare_genotype_data()
    df = pd.DataFrame({column: [0.0, 50.0, 100.0]})
    gd._queue_locus_distribution(
        df=df,
        export_path=tmp_path / "x.tsv.gz",
        description_prefix="",
        series_name=series_name,
        **_violin_kwargs(),
    )
    data = gd.snpio_mqc.queue_linegraph.call_args.args[0]
    assert list(data) == [expected]
    assert data[expected][1.0] == 1 and data[expected][51.0] == 1
    assert data[expected][99.0] == 1
