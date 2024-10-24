import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import toytree as tt

from snpio.read_input.genotype_data import GenotypeData


class TreeParser(GenotypeData):
    def __init__(
        self,
        genotype_data: Any,
        treefile: str,
        qmatrix: Optional[str] = None,
        siterates: Optional[str] = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:

        # Initialize the parent class GenotypeData
        super().__init__(
            filename=genotype_data.filename,
            filetype="tree",
            popmapfile=genotype_data.popmapfile,
            force_popmap=genotype_data.force_popmap,
            exclude_pops=genotype_data.exclude_pops,
            include_pops=genotype_data.include_pops,
            plot_format=genotype_data.plot_format,
            prefix=genotype_data.prefix,
            verbose=verbose,
            debug=debug,
        )

        self.logger = genotype_data.logger

        self.treefile = treefile
        self.qmatrix = qmatrix
        self.siterates = siterates
        self.qmatrix
        self.verbose = verbose
        self.debug = debug

        self.show_plots = genotype_data.show_plots

        self._tree = None
        self._qmat = None
        self._site_rates = None

    def read_tree(self) -> tt.tree:
        """Read Newick-style phylogenetic tree into toytree object.

        The Newick-style tree file should follow the format type 0 (see toytree documentation).

        Returns:
            toytree.tree object: The input tree as a toytree object.

        Raises:
            FileNotFoundError: If the tree file is not found.
            PermissionError: If the tree file exists but is not readable.
        """
        if not Path(self.treefile).is_file():
            raise FileNotFoundError(f"File {self.treefile} not found!")

        if not os.access(self.treefile, os.R_OK):
            msg = f"Tree file {self.treefile} is unreadable."
            self.logger.error(msg)
            raise PermissionError(msg)

        return tt.tree(self.treefile)

    def write_tree(
        self, save_path: Optional[str] = None, nexus: bool = False
    ) -> Optional[str]:
        """Write the phylogenetic tree to a file.

        Args:
            save_path (str, optional): Path to save the tree file. If not provided (left as None), then a string representation of the tree is returned. Defaults to None.
            nexus (bool, optional): Whether to save the tree in NEXUS format.If False, then Newick format is used. Defaults to False.

        Returns:
            Optional[str]: The string representation of the tree if save_path is None. Otherwise, None is returned.
        """
        if self._tree is None:
            self._tree = self.read_tree()

        tree_str = self._tree.write(path=save_path)

        if tree_str is not None:
            return tree_str

        self.logger.info(f"Tree saved to {save_path}")

    def tree_stats(self) -> dict:
        """Calculate basic statistics for the phylogenetic tree.

        Returns:
            Dict[str, Any]: Dictionary containing tree statistics such as the number of tips, number of nodes, and total tree height.
        """
        if self._tree is None:
            self._tree = self.read_tree()

        stats = {
            "num_tips": self._tree.ntips,
            "num_nodes": self._tree.nnodes,
            "max_tree_height": self._tree.get_node_data()["height"].max(),
            "tree_length": self._tree.get_node_data()["height"].sum(),
            "is_rooted": self._tree.is_rooted(),
            "is_bifurcating": self._tree.is_bifurcating(),
        }

        self.logger.debug(f"Tree statistics: {stats}")
        return stats

    def get_subtree(self, regex: str) -> tt.tree:
        """Get a subtree rooted at a specified node or tip.

        Args:
            regex (int): Regular expression to match the node or tip name. Regular expressions can be prefixed with '~' to indicate taxa to keep.

        Returns:
            toytree.tree: The subtree rooted at the specified node.
        """
        if self._tree is None:
            self._tree = self.read_tree()

        subtree = self._tree.mod.extract_subtree(regex)
        self.logger.debug(f"Subtree with any tips labeled {regex} obtained.")
        return subtree

    def prune_tree(self, taxa: List[str] | str) -> tt.tree:
        """Prune the tree by removing a set of taxa (leaf nodes).

        Args:
            taxa (Union[List[str], str]): List of taxa names to remove from the tree or a regular expression to match the node or tip name. Regular expressions can be prefixed with '~' to indicate taxa to keep.

        Returns:
            toytree.tree: The pruned tree object.
        """
        if self._tree is None:
            self._tree = self.read_tree()

        if isinstance(taxa, list):
            pruned_tree = self._tree.mod.drop_tips(*taxa)
        else:
            pruned_tree = self._tree.mod.drop_tips(taxa)
        self.logger.debug(f"Pruned tree by removing taxa: {taxa}")
        return pruned_tree

    def visualize_tree(
        self, save_path: Optional[str] = None, show: bool = True
    ) -> None:
        """Visualize the phylogenetic tree.

        Args:
            save_path (Optional[str], optional): Path to save the tree plot. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to True.
        """
        if self._tree is None:
            self._tree = self.read_tree()

        canvas = self._tree.draw()
        if save_path:
            canvas.to_svg(save_path)
            self.logger.info(f"Tree visualization saved to {save_path}")

        if self.show_plots:
            canvas.show()

    def get_distance_matrix(self) -> pd.DataFrame:
        """Calculate the pairwise distance matrix between all tips in the tree.

        Returns:
            pd.DataFrame: Pairwise distance matrix as a pandas DataFrame.
        """
        if self._tree is None:
            self._tree = self.read_tree()

        dist_df = self._tree.distance.get_node_distance_matrix(df=True)
        self.logger.debug("Computed pairwise distance matrix for the tree.")
        return dist_df

    def reroot_tree(self, node: int) -> tt.tree:
        """Reroot the tree at a specific node or tip.

        Args:
            node (int): Index of the node or tip where the tree should be rerooted.

        Returns:
            toytree.tree: The rerooted tree.
        """
        if self._tree is None:
            self._tree = self.read_tree()

        rerooted_tree = self._tree.root(node)
        self.logger.info(f"Tree rerooted at node {node}.")
        return rerooted_tree

    def load_tree_from_string(self, newick_str: str) -> tt.tree:
        """Load a phylogenetic tree from a Newick string.

        Args:
            newick_str (str): The Newick string representing the tree.

        Returns:
            toytree.tree: The loaded tree object.
        """
        tree = tt.tree(newick_str)
        self.logger.debug("Loaded tree from Newick string.")
        return tree

    def _q_from_file(self, header: bool = True) -> pd.DataFrame:
        """Read Q matrix from a file.

        Args:
            header (bool, optional): If True, the first line of the Q matrix file is assumed to be the nucleotide order. Defaults to True.

        Returns:
            pandas.DataFrame: The Q-matrix as a pandas.DataFrame object.

        Raises:
            FileNotFoundError: If the Q matrix file is not found.
        """
        # Initialize a blank Q matrix.
        # This object is a dictionary of dictionaries, where the keys are
        # nucleotides and the values are the substitution rates.
        if not header:
            self.logger.info(
                "Assuming the following nucleotide order because 'header=False': A, C, G, T"
            )

        if not Path(self.qmatrix).is_file():
            raise FileNotFoundError(f"File {self.qmatrix} not found!")

        with open(self.qmatrix, "r") as fin:
            lines = fin.readlines()

            # Check if the file is comma or whitespace separated
            sep = "," if "," in lines[1] else "\s+"

        # Read the Q matrix file using pandas.
        dfq = pd.read_csv(self.qmatrix, sep=sep, header=header)

        dfq = (
            dfq.set_index(dfq.columns)
            if header
            else dfq.set_index(["A", "C", "G", "T"])
        )

        if header:
            dfq = dfq.set_index(dfq.columns)
        else:
            dfq = dfq.set_index(["A", "C", "G", "T"])
            dfq.columns = ["A", "C", "G", "T"]

        dfq = dfq.astype(float)

        self.logger.debug(f"{dfq=}")

        return dfq

    def _q_from_iqtree(self, iqfile: str) -> pd.DataFrame:
        """Read Q matrix from an IQ-TREE (.iqtree) file.

        The IQ-TREE file contains the standard output of an IQ-TREE run and includes the Q-matrix.

        Args:
            iqfile (str): Path to the IQ-TREE file (.iqtree).

        Returns:
            pandas.DataFrame: The Q-matrix as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the IQ-TREE file could not be found.
            IOError: If the IQ-TREE file could not be read.
        """
        qlines = []
        with open(iqfile, "r") as fin:
            foundLine = False
            matlinecount = 0
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                if "Rate matrix Q" in line:
                    foundLine = True
                    continue
                if foundLine:
                    matlinecount += 1
                    if matlinecount > 4:
                        break
                    qlines.append(line)

        # Check that the Q matrix was found and read
        if not qlines:
            raise IOError(f"Rate matrix Q not found in IQ-TREE file {iqfile}")

        # Populate q matrix with values from the IQ-TREE file
        qlines = [line.split(",") if "," in line else line.split() for line in qlines]

        dfq = pd.DataFrame(
            qlines, columns=["nuc", "A", "C", "G", "T"], skip_blank_lines=True
        )
        dfq = dfq.set_index("nuc")
        dfq = dfq.astype(float)

        self.logger.debug(f"{dfq=}")

        return dfq

    def _siterates_from_iqtree(self, rates: str) -> pd.DataFrame:
        """Read site-specific substitution rates from .rates file.

        The rates file is an optional output file generated by IQ-TREE and contains a table of site-specific rates and rate categories.

        Args:
            rates (str): Path to rates input file that is output by IQ-TREE.

        Returns:
            List[float]: List of site-specific substitution rates.

        Raises:
            FileNotFoundError: If the rates file could not be found.
        """
        if not Path(rates).is_file():
            raise FileNotFoundError(f"File {rates} not found.")

        try:
            dfs = pd.read_csv(rates, sep="\s+", header=True, comment="#")
        except IOError as e:
            self.logger.error(f"Could not read rates file {rates}: {e}")
            raise

        return dfs["Rate"].to_list()

    def _validate_rates(self, rates: pd.DataFrame) -> None:
        """Validate the number of site rates matches the number of SNPs.

        Args:
            rates (pd.DataFrame): Site rates object as a pandas DataFrame

        Raises:
            ValueError: If the number of site rates does not match the number of SNPs.
        """
        if self.snp_data is None:
            _ = self.snp_data

        if len(rates) != self.num_snps:
            msg = "Number of site rates != number of snps in the alignment: {len(rates)} != {self.num_snps}"
            self.logger.error(msg)
            raise ValueError(msg)

    def _siterates_from_file(self, fname: str) -> List[float]:
        """Read site-specific substitution rates from a file.

        Args:
            fname (str): Path to the input file containing site rates. Header is optional.

        Returns:
            List[float]: List of site-specific substitution rates.
        """
        with open(fname, "r") as fin:
            lines = fin.readlines()
            sep = "," if "," in lines else "\s+"
            header = lines[0].isalpha()

            line = lines[1].strip()
            if not line:
                raise ValueError("No site rates found in file.")

            ncol = len(line.split(sep))

            if ncol > 1:
                raise ValueError(
                    "Site rates file must have only one column, unless it is in IQ-TREE format."
                )

        dfs = pd.read_csv(fname, sep=sep, header=header, columns=["Rate"])
        return dfs["Rate"].to_list()

    def _validate_qmat(self, qmat: pd.DataFrame) -> None:
        """Validate the Q matrix.

        Args:
            qmat (pd.DataFrame): Q matrix as a pandas DataFrame.

        Raises:
            TypeError: If the Q matrix is not a pandas DataFrame.
            ValueError: If the Q matrix is empty.
            ValueError: If the Q matrix is not square.
            ValueError: If the Q matrix columns do not equal the index.
            ValueError: If the Q matrix columns are not in the order A, C, G, T.
        """
        if not isinstance(qmat, pd.DataFrame):
            msg = "Q matrix must be a pandas DataFrame, but got: {type(qmat)}"
            self.logger.error(msg)
            raise TypeError(msg)

        if qmat.empty:
            msg = "Q matrix is empty after attempting to load from file."
            self.logger.error(msg)
            raise ValueError(msg)

        if qmat.shape[0] != qmat.shape[1]:
            msg = "Q matrix is not square: {qmat.shape}"
            self.logger.error(msg)
            raise ValueError(msg)

        if not all(qmat.columns == qmat.index):
            msg = (
                "Q matrix columns must equal the index: {qmat.columns} != {qmat.index}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        if not all(qmat.columns == ["A", "C", "G", "T"]):
            msg = "Q matrix columns must be in the order A, C, G, T: {qmat.columns}"
            self.logger.error(msg)
            raise ValueError(msg)

    @property
    def qmat(self) -> pd.DataFrame:
        """Get q-matrix object for a corresponding phylogenetic tree.

        Returns:
            pandas.DataFrame: The Q-matrix as a pandas DataFrame.
        """
        if self._qmat is not None:
            self._validate_qmat(self._qmat)
            return self._qmat

        if self.qmatrix is None:
            msg = "Q matrix file path not provided."
            self.logger.error(msg)
            raise TypeError(msg)

        is_iqtree = False
        with open(self.qmatrix, "r") as fin:
            lines = fin.readlines()
            lines = [line.strip() for line in lines]

            if (
                any(line.startswith("Rate matrix Q") for line in lines)
                and len(lines) >= 5
            ):
                is_iqtree = True

            # Check if the file is comma or whitespace separated
            sep = "," if "," in lines[1] else "\s+"

        self._qmat = (
            self._q_from_iqtree(self.qmatrix) if is_iqtree else self._q_from_file()
        )

        self._validate_qmat(self._qmat)

        return self._qmat

    @qmat.setter
    def qmat(self, value: pd.DataFrame) -> None:
        """Set q-matrix for the corrresponding phylogenetic tree.

        Args:
            value (pd.DataFrame): The Q-matrix as a pandas.DataFrame.
        """
        self._validate_qmat(value)
        self._qmat = value

    @property
    def site_rates(self) -> pd.DataFrame:
        """Get site rate data for phylogenetic tree.

        Returns:
            pd.DataFrame: Site rates for the phylogenetic tree.
        """
        if self._site_rates is not None:
            self._validate_rates(self._site_rates)
            return self._site_rates

        if self.siterates is None:
            msg = "Site rates file path not provided."
            self.logger.error(msg)
            raise TypeError(msg)

        is_iqtree = False
        with open(self.siterates, "r") as fin:
            lines = fin.readlines()
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if line]
            lines = [line for line in lines if not line.startswith("#")]

            if any(line.lower().startswith("rate") for line in lines):
                is_iqtree = True

        if is_iqtree:
            self._site_rates = self._siterates_from_iqtree(self.siterates)
        else:
            self._site_rates = self._siterates_from_file(self.siterates)

        self._validate_rates(self._site_rates)

        # Filter site rates to only include rates for remaining loci after
        # alignment filtering.
        if np.count_nonzero(self.loci_indices) < len(self._site_rates):
            idx = np.where(self.loci_indices)[0]
            self._site_rates = [v for i, v in enumerate(self._site_rates) if i in idx]

        return self._site_rates

    @site_rates.setter
    def site_rates(self, value: pd.DataFrame) -> None:
        """Set site_rates object."""
        self._validate_rates(value)
        self._site_rates = value

    @property
    def tree(self):
        """Get newick tree from provided path.

        Returns:
            toytree.tree: The toytree tree object.
        """
        if self._tree is not None:
            return self._tree

        if self.treefile is None:
            msg = "Tree file path not provided."
            self.logger.error(msg)
            raise TypeError(msg)

        self._tree = self.read_tree()
        return self._tree

    @tree.setter
    def tree(self, value: tt.tree) -> None:
        """Setter for newick tree data.

        Args:
            value (toytree.tree): The tree object to set.
        """
        self._tree = value
