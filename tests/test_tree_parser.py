import tempfile
import unittest
from pathlib import Path

import pandas as pd
import toytree as tt

from snpio import TreeParser  # Update this import to the correct path


class TestTreeParser(unittest.TestCase):
    def setUp(self):
        """Set up a random tree for testing."""
        # Create a random tree with 10 tips
        self.tree = tt.tree(data="((a,b),(c,d),(e,f),(g,h),(i,j));")
        self.treefile = tempfile.NamedTemporaryFile(delete=False, suffix=".nwk")
        self.tree.write(self.treefile.name)

        # Mock GenotypeData object
        self.mock_genotype_data = self.MockGenotypeData(filename=self.treefile.name)

        # Create TreeParser instance
        self.parser = TreeParser(
            genotype_data=self.mock_genotype_data,
            treefile=self.treefile.name,
            verbose=True,
            debug=True,
        )

    def tearDown(self):
        """Clean up the temporary files."""
        Path(self.treefile.name).unlink()

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

            self.num_snps = 100

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
