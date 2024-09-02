import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import toytree as tt

from snpio.read_input.genotype_data import GenotypeData


class TreeParser(GenotypeData):
    def __init__(
        self,
        filename: Optional[str] = None,
        popmapfile: Optional[str] = None,
        force_popmap: bool = False,
        exclude_pops: Optional[List[str]] = None,
        include_pops: Optional[List[str]] = None,
        guidetree: Optional[str] = None,
        qmatrix_iqtree: Optional[str] = None,
        qmatrix: Optional[str] = None,
        siterates: Optional[str] = None,
        siterates_iqtree: Optional[str] = None,
        plot_format: Optional[str] = "png",
        prefix="snpio",
        verbose: bool = True,
        **kwargs,
    ) -> None:

        # Initialize the parent class GenotypeData
        super().__init__(
            filename=filename,
            filetype="tree",
            popmapfile=popmapfile,
            force_popmap=force_popmap,
            exclude_pops=exclude_pops,
            include_pops=include_pops,
            guidetree=guidetree,
            qmatrix_iqtree=qmatrix_iqtree,
            qmatrix=qmatrix,
            siterates=siterates,
            siterates_iqtree=siterates_iqtree,
            plot_format=plot_format,
            prefix=prefix,
            verbose=verbose,
            **kwargs,
        )

        if self.filename:
            self.read_tree(filename)

    def read_tree(self, treefile: str) -> tt.tree:
        """
        Read Newick-style phylogenetic tree into toytree object.

        The Newick-style tree file should follow the format type 0 (see toytree documentation).

        Args:
            treefile (str): Path to the Newick-style tree file.

        Returns:
            toytree.tree object: The input tree as a toytree object.

        Raises:
            FileNotFoundError: If the tree file is not found.
            AssertionError: If the tree file is not readable.
        """
        if not Path(treefile).is_file():
            raise FileNotFoundError(f"File {treefile} not found!")

        assert os.access(treefile, os.R_OK), f"File {treefile} isn't readable"

        return tt.tree(treefile, tree_format=0)

    def q_from_file(self, fname: str, label: bool = True) -> pd.DataFrame:
        """
        Read Q matrix from a file.

        Args:
            fname (str): Path to the Q matrix input file.

            label (bool): True if the nucleotide label order is present, False otherwise.

        Returns:
            pandas.DataFrame: The Q-matrix as a pandas DataFrame object.

        Raises:
            FileNotFoundError: If the Q matrix file is not found.
        """
        q = self._blank_q_matrix()

        if not label:
            print("Warning: Assuming the following nucleotide order: A, C, G, T")

        if not Path(fname).is_file():
            raise FileNotFoundError(f"File {fname} not found!")

        with open(fname, "r") as fin:
            header = True
            qlines = list()
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                if header:
                    if label:
                        order = line.split()
                        header = False
                    else:
                        order = ["A", "C", "G", "T"]
                    continue
                else:
                    qlines.append(line.split())
        fin.close()

        for l in qlines:
            for index in range(0, 4):
                q[l[0]][order[index]] = float(l[index + 1])
        qdf = pd.DataFrame(q)
        return qdf.T

    def q_from_iqtree(self, iqfile: str) -> pd.DataFrame:
        """
        Read Q matrix from an IQ-TREE (\*.iqtree) file.

        The IQ-TREE file contains the standard output of an IQ-TREE run and includes the Q-matrix.

        Args:
            iqfile (str): Path to the IQ-TREE file (\*.iqtree).

        Returns:
            pandas.DataFrame: The Q-matrix as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the IQ-TREE file could not be found.
            IOError: If the IQ-TREE file could not be read.
        """

        q = self._blank_q_matrix()
        qlines = list()
        try:
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
                        stuff = line.split()
                        qlines.append(stuff)
                    else:
                        continue
        except (IOError, FileNotFoundError):
            raise FileNotFoundError(f"Could not open IQ-TREE file {iqfile}")

        # Check that the Q matrix was found and read
        if not qlines:
            raise IOError(f"Rate matrix Q not found in IQ-TREE file {iqfile}")

        # Populate q matrix with values from the IQ-TREE file
        order = [l[0] for l in qlines]
        for l in qlines:
            for index in range(0, 4):
                q[l[0]][order[index]] = float(l[index + 1])

        qdf = pd.DataFrame(q)
        return qdf.T

    def _blank_q_matrix(self, default: float = 0.0) -> Dict[str, Dict[str, float]]:
        """
        Create a blank Q-matrix dictionary initialized with default values.

        Args:
            default (float, optional): Default value to initialize the Q-matrix cells. Defaults to 0.0.

        Returns:
            Dict[str, Dict[str, float]]: Blank Q-matrix dictionary.

        """
        q: Dict[str, Dict[str, float]] = dict()
        for nuc1 in ["A", "C", "G", "T"]:
            q[nuc1] = dict()
            for nuc2 in ["A", "C", "G", "T"]:
                q[nuc1][nuc2] = default
        return q

    def siterates_from_iqtree(self, iqfile: str) -> pd.DataFrame:
        """
        Read site-specific substitution rates from \*.rates file.

        The \*.rates file is an optional output file generated by IQ-TREE and contains a table of site-specific rates and rate categories.

        Args:
            iqfile (str): Path to \*.rates input file.

        Returns:
            List[float]: List of site-specific substitution rates.

        Raises:
            FileNotFoundError: If the rates file could not be found.
            IOError: If the rates file could not be read from.
        """
        s = []
        try:
            with open(iqfile, "r") as fin:
                for line in fin:
                    line = line.strip()
                    if (
                        not line
                        or line.startswith("#")
                        or line.lower().startswith("site")
                    ):
                        continue
                    else:
                        stuff = line.split()
                        s.append(float(stuff[1]))

        except (IOError, FileNotFoundError):
            raise IOError(f"Could not open iqtree file {iqfile}")
        return s

    def _validate_rates(self) -> None:
        """
        Validate the number of site rates matches the number of SNPs.

        Raises:
            ValueError: If the number of site rates does not match the number of SNPs.
        """
        if len(self._site_rates) != self.num_snps:
            raise ValueError(
                "The number of site rates was not equal to the number of snps in the alignment."
            )

    def siterates_from_file(self, fname: str) -> List[float]:
        """
        Read site-specific substitution rates from a file.

        Args:
            fname (str): Path to the input file.

        Returns:
            List[float]: List of site-specific substitution rates.
        """
        s = list()
        with open(fname, "r") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                else:
                    s.append(float(line.split()[0]))
        fin.close()
        return s

    @property
    def q(self):
        """Get q-matrix object for phylogenetic tree."""
        if self.qmatrix_iqtree is not None and self.qmatrix is None:
            self._q = self.q_from_iqtree(self.qmatrix_iqtree)
        elif self.qmatrix_iqtree is None and self.qmatrix is not None:
            self._q = self.q_from_file(self.qmatrix)
        elif self.qmatrix is None and self.qmatrix_iqtree is None and self._q is None:
            raise TypeError(
                "qmatrix or qmatrix_iqtree must be provided at class instantiation or the q property must be set to get the q object."
            )
        return self._q

    @q.setter
    def q(self, value):
        """Set q-matrix for phylogenetic tree."""
        self._q = value

    @property
    def site_rates(self):
        """Get site rate data for phylogenetic tree."""
        if self.siterates_iqtree is not None and self.siterates is None:
            self._site_rates = self.siterates_from_iqtree(self.siterates_iqtree)
            self._validate_rates()
        elif self.siterates_iqtree is None and self.siterates is not None:
            self._site_rates = self.siterates_from_file(self.siterates)
            self._validate_rates()
        elif (
            self.siterates_iqtree is None
            and self.siterates is None
            and self._site_rates is None
        ):
            raise TypeError(
                "siterates or siterates_iqtree must be provided at class instantiation or the site_rates property must be set to get the site_rates object."
            )
        self._site_rates = [
            self._site_rates[i]
            for i in range(len(self._site_rates))
            if i in self.loci_indices
        ]
        return self._site_rates

    @site_rates.setter
    def site_rates(self, value):
        """Set site_rates object."""
        self._site_rates = value

    @property
    def tree(self):
        """Get newick tree provided at class instantiation."""
        if self.guidetree is not None:
            self._tree = self.read_tree(self.guidetree)
        elif self.guidetree is None and self._tree is None:
            raise TypeError(
                "Either a guidetree file must be provided at class instantiation or the tree property must be set to get the tree object."
            )
        return self._tree

    @tree.setter
    def tree(self, value):
        """Setter for newick tree data."""
        self._tree = value
