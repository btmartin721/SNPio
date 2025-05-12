import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import toytree as tt

from snpio.read_input.genotype_data import GenotypeData


class TreeParser(GenotypeData):
    """TreeParser class for reading and manipulating phylogenetic trees.

    This class provides methods for reading, writing, and manipulating phylogenetic trees. The TreeParser class inherits from the GenotypeData class and provides additional functionality for working with phylogenetic trees. The TreeParser class can read phylogenetic trees from Newick or NEXUS format files, calculate basic statistics for the tree, extract subtrees, prune the tree, reroot the tree, and calculate pairwise distance matrices.

    Example:
        >>> tp = TreeParser(
        ...     genotype_data=gd_filt,
        ...     treefile="snpio/example_data/trees/test.tre",
        ...     qmatrix="snpio/example_data/trees/test.iqtree",
        ...     siterates="snpio/example_data/trees/test14K.rate",
        ...     show_plots=True,
        ...     verbose=True,
        ...     debug=False,
        ... )
        >>>
        >>> tree = tp.read_tree()
        >>> print(tp.tree_stats())
        >>> tp.reroot_tree("~EA")
        >>> print(tp.get_distance_matrix())
        >>> print(tp.qmat)
        >>> print(tp.site_rates)
        >>> subtree = tp.get_subtree("~EA")
        >>> pruned_tree = tp.prune_tree("~ON")
        >>> print(tp.write_tree(subtree, save_path=None))
        >>> print(tp.write_tree(pruned_tree, save_path=None)

    Attributes:
        genotype_data (GenotypeData): GenotypeData object containing the SNP data.
        treefile (str): Path to the phylogenetic tree file.
        qmatrix (str): Path to the Q matrix file.
        siterates (str): Path to the site rates file.
        verbose (bool): Whether to display verbose output.
        debug (bool): Whether to display debug output.
    """

    def __init__(
        self,
        genotype_data: Any,
        treefile: str,
        qmatrix: str | None = None,
        siterates: str | None = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize the TreeParser object.

        This class provides methods for reading, writing, and manipulating phylogenetic trees. The TreeParser class inherits from the GenotypeData class and provides additional functionality for working with phylogenetic trees. The TreeParser class can read phylogenetic trees from Newick or NEXUS format files, calculate basic statistics for the tree, extract subtrees, prune the tree, reroot the tree, and calculate pairwise distance matrices.

        Args:
            genotype_data (Any): GenotypeData object containing the SNP data.
            treefile (str): Path to the phylogenetic tree file.
            qmatrix (str, optional): Path to the Q matrix file. Defaults to None.
            siterates (str, optional): Path to the site rates file. Defaults to None.
            verbose (bool, optional): Whether to display verbose output. Defaults to False.
            debug (bool, optional): Whether to display debug output. Defaults to False.
        """

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
            verbose=genotype_data.verbose,
            debug=genotype_data.debug,
            plot_fontsize=genotype_data.plot_fontsize,
            plot_dpi=genotype_data.plot_dpi,
            loci_indices=genotype_data.loci_indices,
            sample_indices=genotype_data.sample_indices,
        )

        self.logger = genotype_data.logger

        self.genotype_data = genotype_data
        self.treefile = treefile
        self.qmatrix = qmatrix
        self.siterates = siterates
        self.qmatrix
        self.verbose = verbose
        self.debug = debug

        self._tree = None
        self._qmat = None
        self._site_rates = None

    def read_tree(self) -> tt.ToyTree:
        """Read Newick or NEXUS-style phylogenetic tree into toytree object.

        This method reads a phylogenetic tree from a file and returns it as a toytree object. The tree file can be in Newick or NEXUS format. If the tree file is not found or is unreadable, an exception is raised.

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
        self, tree: tt.ToyTree, save_path: str | None = None, nexus: bool = False
    ) -> str | None:
        """Write the phylogenetic tree to a file.

        This method saves the phylogenetic tree to a file in Newick or NEXUS format. If the save_path argument is not provided, the tree is returned as a string representation.

        Args:
            tree (toytree.tree): The tree object to save.
            save_path (str | None): Path to save the tree file. If not provided (left as None), then a string representation of the tree is returned. Defaults to None.
            nexus (bool): Whether to save the tree in NEXUS format.If False, then Newick format is used. Defaults to False.

        Returns:
            str | None: The string representation of the tree if save_path is None. Otherwise, None is returned.

        Raises:
            TypeError: If the input tree is not a toytree object.
        """
        if not isinstance(tree, tt.ToyTree):
            msg = f"Input tree must be a toytree object, but got: {type(tree)}."
            self.logger.error(msg)
            raise TypeError(msg)

        tree_str = tree.write(path=save_path)

        if tree_str is not None:
            return tree_str

        self.logger.info(f"Tree saved to {save_path}")

    def tree_stats(self) -> Dict[str, Any]:
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

    def get_subtree(self, regex: str) -> tt.ToyTree:
        """Get a subtree rooted at a specified node or tip.

        This method extracts a subtree from the phylogenetic tree rooted at the specified node or tip. The subtree is returned as a toytree object. The regex argument can be a regular expression to match the node or tip name. Regular expressions can be prefixed with '~' to indicate taxa to keep.

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

    def prune_tree(self, taxa: List[str] | str) -> tt.ToyTree:
        """Prune the tree by removing a set of taxa (leaf nodes).

        This method prunes the tree by removing a set of taxa (leaf nodes) from the tree. The taxa argument can be a list of taxa names to remove from the tree or a regular expression to match the node or tip name. Regular expressions can be prefixed with '~' to indicate taxa to keep.

        Args:
            taxa (List[str] | str): List of taxa names to remove from the tree or a regular expression to match the node or tip name. Regular expressions can be prefixed with '~' to indicate taxa to keep.

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

    def get_distance_matrix(self) -> pd.DataFrame:
        """Calculate the pairwise distance matrix between all tips in the tree.

        This method computes the pairwise distance matrix between all nodes and tips in the phylogenetic tree. The distance matrix is returned as a pandas DataFrame object.

        Returns:
            pd.DataFrame: Pairwise distance matrix as a pandas DataFrame.
        """
        if self._tree is None:
            self._tree = self.read_tree()

        dist_df = self._tree.distance.get_node_distance_matrix(df=True)
        self.logger.debug("Computed pairwise distance matrix for the tree.")
        return dist_df

    def reroot_tree(self, node: int | str | List[str]) -> tt.ToyTree:
        """Reroot the tree at a specific node or tip.

        This method reroots the tree at a specific node or tip, changing the root of the tree to the specified node. The rerooted tree is returned as a toytree object.

        Args:
            node (int | str | List[str]): Index of the node or tip where the tree should be rerooted, a regex string to match the node or tip name prefixed by "~", or a list of node or tip names.

        Returns:
            toytree.tree: The rerooted tree.
        """
        if self._tree is None:
            self._tree = self.read_tree()

        is_list = isinstance(node, list)

        tree = self._tree
        mrca = tree.get_mrca_node(*node) if is_list else tree.get_mrca_node(node)

        rerooted_tree = self._tree.root(mrca)
        self.logger.info(f"Tree rerooted at node {node}.")
        return rerooted_tree

    def load_tree_from_string(self, newick_str: str) -> tt.ToyTree:
        """Load a phylogenetic tree from a Newick string.

        This method loads a phylogenetic tree from a Newick string and returns it as a toytree object.

        Args:
            newick_str (str): The Newick string representing the tree.

        Returns:
            toytree.tree: The loaded tree object.
        """
        tree = tt.tree(newick_str)
        self.logger.debug("Loaded tree from Newick string.")
        return tree

    def _q_from_file(self) -> pd.DataFrame:
        """Read Q matrix from a file.

        This method reads the Q matrix from a file and returns it as a pandas DataFrame object. The Q matrix file can be in either comma-separated or whitespace-separated format.

        Returns:
            pandas.DataFrame: The Q-matrix as a pandas.DataFrame object.

        Raises:
            FileNotFoundError: If the Q matrix file is not found.
        """
        if not Path(self.qmatrix).is_file():
            raise FileNotFoundError(f"File {self.qmatrix} not found!")

        with open(self.qmatrix, "r") as fin:
            lines = fin.readlines()

            header = True if "A" in lines[0].upper() else False

            # Check if the file is comma or whitespace separated
            sep = r"," if r"," in lines[1] else r"\s+"

        header_idx = 0 if header else None

        # Read the Q matrix file using pandas.
        dfq = pd.read_csv(
            self.qmatrix, sep=sep, header=header_idx, names=["A", "C", "G", "T"]
        )

        if header:
            dfq = dfq.set_index(dfq.columns)
        else:
            nucs = ["A", "C", "G", "T"]
            dfq.columns = nucs
            dfq.index = nucs
        dfq = dfq.astype(float)

        self.logger.debug(f"{dfq=}")

        return dfq

    def _q_from_iqtree(self) -> pd.DataFrame:
        """Read Q matrix from an IQ-TREE (.iqtree) file.

        The IQ-TREE file contains the standard output of an IQ-TREE run and includes the Q-matrix. This method reads the Q matrix from the IQ-TREE file and returns it as a pandas DataFrame object. The IQ-TREE file should contain the rate matrix Q in the format:

        ```
            Rate matrix Q

            A	C	G	T
            -0.000000	0.000000	0.000000	0.000000
            0.000000	-0.000000	0.000000	0.000000
            0.000000	0.000000	-0.000000	0.000000
            0.000000	0.000000	0.000000	-0.000000
        ```

        The header row and index column are optional and can be omitted. The Q matrix values and header should be separated by whitespace or commas, and the matrix should be square with the columns and index in the order A, C, G, T.

        Args:
            iqfile (str): Path to the IQ-TREE file (.iqtree).

        Returns:
            pandas.DataFrame: The Q-matrix as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the IQ-TREE file could not be found.
            IOError: If the IQ-TREE file could not be read.
        """
        qlines = []
        with open(self.qmatrix, "r") as fin:
            foundLine = False
            matlinecount = 0
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                if "rate matrix q" in line.lower():
                    foundLine = True
                    continue
                if foundLine:
                    matlinecount += 1
                    if matlinecount > 4:
                        break
                    qlines.append(line)

        # Check that the Q matrix was found and read
        if not qlines:
            raise IOError(f"Rate matrix Q not found in IQ-TREE file {self.qmatrix}")

        # Populate q matrix with values from the IQ-TREE file
        qlines = [line.split(",") if "," in line else line.split() for line in qlines]

        dfq = pd.DataFrame(qlines, columns=["nuc", "A", "C", "G", "T"])
        dfq = dfq.set_index("nuc")
        dfq = dfq.astype(float)

        self.logger.debug(f"{dfq=}")

        return dfq

    def _siterates_from_iqtree(self) -> pd.DataFrame:
        """Read site-specific substitution rates from .rates file.

        The rates file is an optional output file generated by IQ-TREE (.rate) and contains a table of site-specific rates and rate categories. This method reads the site rates from the IQ-TREE file and returns them as a list of float values. The rates file should contain the site rates in the format:

        ```
        # Any comment lines can be included here.
        Site    Rate   Cat C_rate
        1       0.0000  1   0.0000
        2       0.0000  1   0.0000
        3       0.0000  1   0.0000
        4       0.0000  1   0.0000
        5       0.0000  1   0.0000
        ```

        The site rates should be in the 'Rate' column and separated by whitespace or commas.

        Returns:
            List[float]: List of site-specific substitution rates.

        Raises:
            FileNotFoundError: If the rates file could not be found.
        """
        if not Path(self.siterates).is_file():
            self.logger.error(f"File {self.siterates} not found.")
            raise FileNotFoundError(f"File {self.siterates} not found.")

        try:
            dfs = pd.read_csv(self.siterates, sep=r"\s+", comment="#")
        except IOError as e:
            msg = f"Could not read rates file {self.siterates}: {e}"
            self.logger.error(msg)
            raise

        return dfs["Rate"].to_list()

    def _validate_rates(self, rates: pd.DataFrame) -> None:
        """Validate the number of site rates matches the number of SNPs.

        This method validates the number of site rates matches the number of SNPs in the alignment. If the number of site rates does not match the number of SNPs, a ValueError is raised.

        Args:
            rates (pd.DataFrame): Site rates object as a pandas DataFrame

        Raises:
            ValueError: If the number of site rates does not match the number of SNPs.
        """
        if self.genotype_data.snp_data is None:
            _ = self.genotype_data.snp_data

        if len(rates) != self.genotype_data.num_snps:
            msg = f"Number of site rates != number of snps in the alignment: {len(rates)} != {self.genotype_data.num_snps}"
            self.logger.error(msg)
            raise ValueError(msg)

    def _siterates_from_file(self) -> List[float]:
        """Read site-specific substitution rates from a file.

        This method reads the site-specific substitution rates from a file and returns them as a list of float values. The site rates file should contain the site rates in a single column, with each rate on a separate line. For example:

        ```
        0.0000
        0.0000
        0.0000
        0.0000
        0.0000
        ```

        Returns:
            List[float]: List of site-specific substitution rates.
        """
        with open(self.siterates, "r") as fin:
            lines = fin.readlines()
            sep = "," if "," in lines else r"\s+"
            header = lines[0] if lines[0].isalpha() else None

            line = lines[1].strip()
            if not line:
                msg = "Site rates file is empty."
                self.logger.error(msg)
                raise ValueError()

            ncol = len(line.split(sep))

            if ncol > 1:
                msg = "Site rates file must have only one column."
                self.logger.error(msg)
                raise ValueError(msg)

        header = 0 if header else None

        dfs = pd.read_csv(self.siterates, sep=sep, header=header, names=["Rate"])
        return dfs["Rate"].to_list()

    def _validate_qmat(self, qmat: pd.DataFrame) -> None:
        """Validate the Q matrix.

        This method validates the Q matrix to ensure it is a square matrix with the correct columns and index. The Q matrix should be a square matrix with columns and index in the order A, C, G, T.

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

        This method reads the Q matrix from a file and returns it as a pandas DataFrame object. The Q matrix file can be in either comma-separated or whitespace-separated format. The Q matrix should be a square matrix with columns and index in the order A, C, G, T.

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
            lines = [line for line in lines if line]

            if (
                lines
                and any("rate matrix q" in line.lower() for line in lines)
                and len(lines) >= 5
            ):
                is_iqtree = True

        self._qmat = self._q_from_iqtree() if is_iqtree else self._q_from_file()
        self._validate_qmat(self._qmat)

        return self._qmat

    @qmat.setter
    def qmat(self, value: pd.DataFrame) -> None:
        """Set q-matrix for the corrresponding phylogenetic tree.

        This method sets the Q matrix for the corresponding phylogenetic tree. The Q matrix should be a square matrix with columns and index in the order A, C, G, T. The Q matrix must be provided as a pandas DataFrame object with the correct columns and index.

        Args:
            value (pd.DataFrame): The Q-matrix as a pandas.DataFrame.
        """
        self._validate_qmat(value)
        self._qmat = value

    @property
    def site_rates(self) -> pd.DataFrame:
        """Get site rate data for phylogenetic tree.

        This method reads the site-specific substitution rates from a file and returns them as a list of float values. The site rates file should either contain the site rates in a single column, with each rate on a separate line, or in a table format with the rates in the 'Rate' column, as output by IQ-TREE. For example:

        ```
        0.0000
        0.0000
        0.0000
        0.0000
        0.0000
        ```

        OR:

        ```
        # Any comment lines can be included here.
        Site    Rate   Cat C_rate
        1       0.0000  1   0.0000
        2       0.0000  1   0.0000
        3       0.0000  1   0.0000
        4       0.0000  1   0.0000
        5       0.0000  1   0.0000
        ```

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

            if any("c_rate" in line.lower() for line in lines) and any(
                "rate" in line.lower() for line in lines
            ):
                is_iqtree = True

        if is_iqtree:
            self._site_rates = self._siterates_from_iqtree()
        else:
            self._site_rates = self._siterates_from_file()

        # Filter site rates to only include rates for remaining loci after
        # alignment filtering.
        if self._loci_indices is None:
            self.loci_indices = np.ones(len(self.snp_data), dtype=bool)

        if np.count_nonzero(self.loci_indices) < len(self._site_rates):
            sr = np.array(self._site_rates)
            self._site_rates = sr[self.loci_indices].tolist()

        self._validate_rates(self._site_rates)
        return self._site_rates

    @site_rates.setter
    def site_rates(self, value: List[float]) -> None:
        """Set site_rates object.

        This method sets the site rates for the corresponding phylogenetic tree. The site rates should be provided as a list of float values.

        Args:
            value (List[float]): The site rates as a list of floats.
        """
        self._validate_rates(value)
        self._site_rates = value

    @property
    def tree(self):
        """Get newick tree from provided path.

        This method reads the phylogenetic tree from the provided tree file path and returns it as a toytree object. If the tree file path is not provided, an exception is raised.

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
    def tree(self, value: tt.ToyTree) -> None:
        """Setter for newick tree data.

        This method sets the phylogenetic tree for the corresponding tree parser object.

        Args:
            value (toytree.tree): The tree object to set.
        """
        self._tree = value
