import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import toytree as tt

from snpio import PhylipReader, TreeParser


class TestTreeParser(unittest.TestCase):
    def setUp(self):
        """Set up a random tree for testing."""
        # Create a random tree with 10 tips
        self.tree = tt.tree(data="((a,b),(c,d),(e,f),(g,h),(i,j));")
        self.filename = tempfile.NamedTemporaryFile(delete=False, suffix=".phy")

        self.snp_data = np.random.choice(["A", "C", "G", "T"], (10, 50), replace=True)
        self.num_snps = 50

        sampleids = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

        header = "10 50"
        data = "\n".join(
            [
                f"{sid}\t" + "".join(row)
                for row, sid in zip(self.snp_data.tolist(), sampleids)
            ]
        )
        data = f"{header}\n{data}".encode()

        self.filename.write(data)
        self.filename.seek(0)

        self.treefile = tempfile.NamedTemporaryFile(delete=False, suffix=".nwk")
        self.treefile.write(f"{self.tree.write()}".encode())
        self.treefile.seek(0)

        # Mock GenotypeData object
        self.mock_genotype_data = PhylipReader(filename=self.filename.name)

        self.siterates = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        self.qmatrix = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")

        self.site_rates = np.random.random(50).tolist()
        rates = "\n".join(map(str, self.site_rates)).encode()
        self.siterates.write(rates)
        self.siterates.seek(0)

        self.siterates_iqtree = "snpio/example_data/trees/test_n50.rate"

        self.qmat = [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.1, 0.2],
            [0.2, 0.1, 0.0, 0.1],
            [0.3, 0.2, 0.1, 0.0],
        ]

        qmat = "\n".join(["\t".join(map(str, row)) for row in self.qmat]).encode()

        self.qmatrix.write(qmat)
        self.qmatrix.seek(0)

        self.qmatrix_iqtree = "snpio/example_data/trees/test.iqtree"

        # Create TreeParser instance
        self.parser = TreeParser(
            genotype_data=self.mock_genotype_data,
            treefile=self.treefile.name,
            qmatrix=self.qmatrix.name,
            siterates=self.siterates.name,
            verbose=True,
            debug=True,
        )

        self.parser_iqtree = TreeParser(
            genotype_data=self.mock_genotype_data,
            treefile=self.treefile.name,
            qmatrix=self.qmatrix_iqtree,
            siterates=self.siterates_iqtree,
            verbose=True,
            debug=True,
        )

        self.qmat = pd.DataFrame(
            self.qmat, columns=["A", "C", "G", "T"], index=["A", "C", "G", "T"]
        )

        self.qmat_iqtree = [
            [-1.482, 0.2476, 1.066, 0.169],
            [0.1237, -0.7085, 0.06871, 0.5161],
            [0.5769, 0.07446, -0.7699, 0.1186],
            [0.1889, 1.155, 0.2448, -1.588],
        ]

        self.qmat_iqtree = pd.DataFrame(
            self.qmat_iqtree, columns=["A", "C", "G", "T"], index=["A", "C", "G", "T"]
        )

        self.qmat_iqtree.index.name = "nuc"

        self.site_rates_iqtree = [
            0.65244,
            0.71905,
            0.66729,
            2.65438,
            0.45287,
            0.70798,
            0.70532,
            0.43881,
            0.46221,
            0.69352,
            0.72661,
            0.59677,
            0.45332,
            0.45439,
            0.58785,
            0.41904,
            0.41957,
            0.45609,
            0.46272,
            2.78146,
            0.46007,
            3.10141,
            0.65251,
            0.41980,
            0.45148,
            0.42018,
            0.46300,
            0.41985,
            0.45384,
            0.64632,
            0.57909,
            0.45676,
            0.42065,
            11.38813,
            0.57922,
            0.41990,
            0.45883,
            0.46317,
            0.42012,
            0.45216,
            0.45919,
            0.66636,
            0.45845,
            0.46051,
            0.64517,
            0.42045,
            0.44030,
            0.46461,
            0.71732,
            0.59264,
        ]

    def tearDown(self):
        """Clean up the temporary files."""
        Path(self.treefile.name).unlink()
        Path(self.siterates.name).unlink()
        Path(self.qmatrix.name).unlink()

    class MockGenotypeData:
        """A mock class to simulate GenotypeData"""

        def __init__(self, filename):
            self.filename = filename
            self.popmapfile = None
            self.force_popmap = False
            self.exclude_pops = None
            self.include_pops = None
            self.plot_format = None
            self.prefix = None
            self.verbose = False
            self.debug = False
            self.show_plots = False
            self.logger = self.MockLogger()

        def load_aln(self):
            return np.random.choice(["A", "C", "G", "T"], (10, 50), replace=True)

        @property
        def num_snps(self):
            return 50

        class MockLogger:
            def debug(self, msg):
                pass

            def info(self, msg):
                pass

            def error(self, msg):
                pass

    def test_read_tree(self):
        """Test that the tree is correctly read from file."""
        tree = self.parser.read_tree()
        self.assertEqual(tree.ntips, 10, "Tree should have 10 tips")
        self.assertEqual(tree.nnodes, 16, "Tree should have 16 nodes")

    def test_class_properties(self):
        """Test that the class properties are correctly set."""
        self.parser.siterates = self.siterates.name
        self.parser.qmatrix = self.qmatrix.name
        site_rates = self.parser.site_rates
        qmat = self.parser.qmat

        self.parser.siterates = self.siterates_iqtree

        self.assertEqual(
            self.parser.site_rates,
            site_rates,
            "Site rates did not match expected values",
        )

        pd.testing.assert_frame_equal(
            self.qmat, qmat, check_dtype=True, check_index_type=True
        )

        self.parser_iqtree.siterates = self.siterates_iqtree
        self.parser_iqtree.qmatrix = self.qmatrix_iqtree

        site_rates = self.parser_iqtree.site_rates
        qmat = self.parser_iqtree.qmat

        pd.testing.assert_frame_equal(
            self.qmat_iqtree, qmat, check_dtype=True, check_index_type=True
        )

        self.assertEqual(
            self.site_rates_iqtree,
            site_rates,
            f"Site rates did not match expected values: {self.site_rates_iqtree}, {site_rates}",
        )

    def test_tree_stats(self):
        """Test that the tree statistics are correctly calculated."""
        stats = self.parser.tree_stats()
        self.assertEqual(stats["num_tips"], 10, "Tree should have 10 tips")
        self.assertEqual(stats["num_nodes"], 16, "Tree should have 16 nodes")
        self.assertAlmostEqual(
            stats["max_tree_height"], 2, delta=0.1, msg="Tree height should be around 4"
        )

    def test_prune_tree(self):
        """Test pruning the tree by removing taxa."""
        pruned_tree = self.parser.prune_tree(["a", "b", "c"])
        self.assertEqual(pruned_tree.ntips, 7, "Tree should have 7 tips after pruning")
        self.assertEqual(
            pruned_tree.nnodes, 11, "Tree should have 11 nodes after pruning"
        )

    def test_write_tree(self):
        """Test writing the tree to a file."""
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nwk")
        self.parser.write_tree(save_path=output_file.name)

        # Check if file was created and tree is written correctly
        self.assertTrue(Path(output_file.name).is_file(), "Output file should exist")
        written_tree = tt.tree(output_file.name)
        self.assertEqual(written_tree.ntips, 10, "Written tree should have 10 tips")

        Path(output_file.name).unlink()

    def test_get_distance_matrix(self):
        """Test the computation of the pairwise distance matrix."""
        dist_matrix = self.parser.get_distance_matrix()
        self.assertIsInstance(
            dist_matrix, pd.DataFrame, "Distance matrix should be a pandas DataFrame"
        )
        self.assertEqual(dist_matrix.shape, (16, 16), "Distance matrix should be 16x16")


if __name__ == "__main__":
    unittest.main()
