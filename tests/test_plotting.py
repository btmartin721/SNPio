import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

from snpio import NRemover2, PhylipReader, Plotting


class TestPlotting(unittest.TestCase):
    def setUp(self):

        self.phylip_data = tempfile.NamedTemporaryFile(
            delete=False, suffix=".phy", mode="w"
        )
        self.popmap = tempfile.NamedTemporaryFile(
            delete=False, suffix=".popmap", mode="w"
        )

        self.test_popmap_content = [
            "Sample1\tpop1",
            "Sample2\tpop1",
            "Sample3\tpop2",
            "Sample4\tpop2",
            "Sample5\tpop2",
        ]

        with open(self.phylip_data.name, "w") as f:
            f.write(
                textwrap.dedent(
                    """\
                    5 10
                    Sample1\tACGTACGTAC
                    Sample2\tTCGATCGATA
                    Sample3\tGCTAGCTAGC
                    Sample4\tATCGATCGAT
                    Sample5\tCGATCGATCG
                    """
                )
            )

        with open(self.popmap.name, "w") as f:
            f.write("\n".join(self.test_popmap_content))

        self.genotype_data = PhylipReader(
            filename=self.phylip_data.name,
            popmapfile=self.popmap.name,
            prefix="test_read_phylip",
            verbose=False,
            plot_format="png",
        )

        self.plotting = Plotting(self.genotype_data)

    def test_visualize_missingness(self):

        self.genotype_data.missingness_reports()

        expected_file = Path(
            f"{self.genotype_data.prefix}_output/nremover/plots/gtdata/missingness_report.png"
        )

        self.assertTrue(expected_file.exists())
        self.assertTrue(expected_file.is_file())
        self.assertTrue(expected_file.stat().st_size > 0)

    def test_sankey_plot(self):

        nrm = NRemover2(self.genotype_data)
        gd_filt = nrm.filter_missing(0.8).filter_monomorphic().resolve()

        nrm.plot_sankey_filtering_report()

        expected_file = Path(
            Path(f"{self.genotype_data.prefix}_output")
            / "nremover"
            / "plots"
            / "gtdata"
            / "sankey_plots"
            / "filtering_results_sankey_mqc.html"
        )

        self.assertTrue(expected_file.exists())
        self.assertTrue(expected_file.is_file())
        self.assertTrue(expected_file.stat().st_size > 0)

    def tearDown(self):
        if Path(self.phylip_data.name).exists():
            Path(self.phylip_data.name).unlink(missing_ok=True)
        if Path(self.popmap.name).exists():
            Path(self.popmap.name).unlink(missing_ok=True)

        for f in Path(".").glob("tmp*phy"):
            if f.is_file():
                f.unlink(missing_ok=True)

        dir = Path(f"{self.genotype_data.prefix}_output")
        if dir.is_dir():
            shutil.rmtree(dir)

        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
