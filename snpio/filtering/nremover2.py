import sys
import os
import tempfile
import warnings
import pandas as pd
import numpy as np
from Bio.Align import MultipleSeqAlignment
from Bio import SeqUtils
from copy import deepcopy

from snpio.plotting.plotting import Plotting
from snpio.read_input.genotype_data import GenotypeData


class NRemover2:
    """
    A class for filtering alignments based on the proportion of missing data in a genetic alignment.

    The class can filter out sequences (samples) and loci (columns) that exceed a missing data threshold.

    The loci can be filtered by global missing data proportions or if any given population exceeds the missing data threshold.

    A number of informative plots are also generated.

    Note:
        NRemover2 handles the following characters as missing data:
            - 'N'
            - '-'
            - '?'
            - '.'

        Thus, it treats gaps as missing data. Please keep this in mind when using NRemover2.

    Args:
        popgenio (GenotypeData): An instance of the GenotypeData class containing the genetic data alignment, population map, and populations.

    Attributes:
        alignment (list of Bio.SeqRecord.SeqRecord): The input alignment to filter.

        populations (list of str): The population for each sequence in the alignment.

    Properties:
        alignment (property): Property for accessing and setting the alignment.

        msa (property): Property for accessing and setting the msa.

        population_sequences (property): Property for accessing the sequences for each population.

    Methods:
        nremover: Runs the whole NRemover2 pipeline.

        filter_missing: Filters out sequences from the alignment that have more than a given proportion of missing data.

        filter_missing_pop: Filters out sequences from the alignment that have more than a given proportion of missing data in a specific population.

        filter_missing_sample: Filters out samples from the alignment that have more than a given proportion of missing data.

        filter_minor_allele_frequency: Filters out loci (columns) where the minor allele frequency is below the threshold.

        filter_monomorphic: Filters out monomorphic sites.

        filter_singletons: Filters out loci (columns) where the only variant is a singleton.

        filter_non_biallelic: Filter out loci (columns) that have more than 2 alleles.

        get_population_sequences: Returns the sequences for a specific population.

        count_iupac_alleles: Counts the number of occurrences of each IUPAC ambiguity code in a given column.

        count_unique_bases: Counts the number of unique bases in a given column.

        plot_missing_data_thresholds: Plots the proportion of missing data against the filtering thresholds.

        plot_sankey_filtering_report: Makes a Sankey plot showing the number of loci removed at each filtering step.

        print_filtering_report: Prints a summary of the filtering results.
    """

    def __init__(self, popgenio):
        self._msa = popgenio.alignment
        self._alignment = deepcopy(self._msa)
        self.popgenio = popgenio
        self.popmap = popgenio.popmap
        self.popmap_inverse = popgenio.popmap_inverse
        self.populations = list(set(popgenio.populations))
        self.samples = popgenio.samples
        self.poplist = popgenio.populations

        self.loci_indices = None
        self.sample_indices = None

    def nremover(
        self,
        max_missing_global=1.0,
        max_missing_pop=1.0,
        max_missing_sample=1.0,
        min_maf=0.0,
        biallelic=False,
        monomorphic=False,
        singletons=False,
        search_thresholds=True,
        plot_outfile="missingness_report.png",
        suppress_cletus=True,
        plot_dir="plots",
        included_steps=None,
    ):
        """
        Runs the NRemover2 pipeline for filtering alignments based on missing data, minor allele frequency, and monomorphic, non-biallelic, and singleton sites.

        Args:
            max_missing_global (float, optional): The maximum proportion of missing data allowed globally (across all samples). Defaults to 1.0.

            max_missing_pop (float, optional): The maximum proportion of missing data allowed within a population. Defaults to 1.0.

            max_missing_sample (float, optional): The maximum proportion of missing data allowed for a single sample. Defaults to 1.0.

            min_maf (float, optional): The minimum minor allele frequency threshold. Defaults to 0.0.

            biallelic (bool, optional): Whether to filter out non-biallelic loci. Defaults to False.

            monomorphic (bool, optional): Whether to filter out monomorphic loci. Defaults to False.

            singletons (bool, optional): Whether to filter out loci where the only variant is a singleton. Defaults to False.

            search_thresholds (bool, optional): Whether to search across multiple thresholds and make a plot for visualization. Defaults to True.

            plot_outfile (str, optional): The filename for the missingness report plot. Defaults to "missingness_report.png".

            plot_dir (str, optional): The directory to save the plots. Defaults to "plots".

            included_steps (list, optional): The steps to include in the Sankey plot. If None, all steps will be included. Defaults to None.

        Returns:
            GenotypeData: A GenotypeData object containing the filtered alignment, retained loci indices, and retained sample indices.

        """
        if not suppress_cletus:
            self.print_cletus()

        self.alignment = self.msa[:]

        aln_before = deepcopy(self.alignment)
        indices_loci_before = range(len(aln_before[0]))

        Plotting.plot_gt_distribution(
            self.popgenio.genotypes_int, plot_dir=plot_dir
        )

        if search_thresholds:
            self.search_thresholds_ = True
            self.plot_missing_data_thresholds(plot_outfile, plot_dir=plot_dir)

        steps = [
            (
                "Filter missing data (sample)",
                max_missing_sample < 1.0,
                max_missing_sample,
                self.filter_missing_sample,
                5,
            ),
            (
                "Filter monomorphic sites",
                monomorphic,
                monomorphic,
                self.filter_monomorphic,
                0,
            ),
            (
                "Filter singletons",
                singletons,
                singletons,
                self.filter_singletons,
                1,
            ),
            (
                "Filter non-biallelic sites",
                biallelic,
                biallelic,
                self.filter_non_biallelic,
                2,
            ),
            (
                "Filter missing data (global)",
                max_missing_global < 1.0,
                max_missing_global,
                self.filter_missing,
                3,
            ),
            (
                "Filter missing data (population)",
                max_missing_pop < 1.0,
                max_missing_pop,
                self.filter_missing_pop,
                4,
            ),
            (
                "Filter minor allele frequency",
                min_maf > 0.0,
                min_maf,
                self.filter_minor_allele_frequency,
                6,
            ),
        ]

        indices_loci_before = list(range(len(aln_before[0])))
        loci_removed_per_step = []
        filter_idx_dict = {}
        retained_indices = indices_loci_before

        for name, condition, threshold, filter_func, step_idx in steps:
            if condition:
                filtered_alignment, indices = filter_func(
                    threshold, alignment=self.alignment
                )
                loci_removed = len(self.alignment[0]) - len(
                    filtered_alignment[0]
                )

                if name != "Filter missing data (sample)":
                    loci_removed_per_step.append((name, loci_removed))

                    # Update retained_indices to only include the indices that were retained
                    retained_indices = [retained_indices[i] for i in indices]
                    filter_idx_dict[step_idx] = retained_indices
                else:
                    self.sample_indices = indices
                    self.samples = [self.samples[i] for i in indices]
                self.alignment = filtered_alignment
            else:
                loci_removed_per_step.append((name, 0))

        aln_after = deepcopy(self.alignment)

        max_step_idx = max(filter_idx_dict)
        self.loci_indices = filter_idx_dict[max_step_idx]

        self.print_filtering_report(
            aln_before, aln_after, loci_removed_per_step
        )

        if included_steps is None:
            included_steps = [
                step_idx for _, condition, _, _, step_idx in steps if condition
            ]

        Plotting.plot_sankey_filtering_report(
            loci_removed_per_step,
            len(aln_before[0]),
            len(aln_after[0]),
            "sankey_filtering_report.html",
            plot_dir=plot_dir,
            included_steps=included_steps,
        )

        return self.return_filtered_output()

    def return_filtered_output(self):
        """
        Creates a temporary alignment file and a temporary population map file, writes data to them, and returns a new GenotypeData object with the filtered alignment.

        Returns:
            GenotypeData: A new GenotypeData object with the filtered alignment.

        Raises:
            None
        """
        # Create a temporary file and write some data to it
        aln = tempfile.NamedTemporaryFile(delete=False)

        if self.sample_indices is not None:
            indices = list(
                set(self.sample_indices) & set(self.popgenio.sample_indices)
            )
            indices.sort()
            self.sample_indices = indices

        else:
            self.sample_indices = self.popgenio.sample_indices

        samples = [
            self.popgenio.samples[i]
            for i in range(len(self.popgenio.samples))
            if i in self.sample_indices
        ]

        popmap = {
            k: v for k, v in self.popgenio.popmap.items() if k in samples
        }

        if self.popgenio.filetype == "vcf":
            vcf_attributes = self.popgenio.subset_vcf_data(
                self.loci_indices,
                self.sample_indices,
                self.popgenio.vcf_attributes,
                self.popgenio.num_snps,
                self.popgenio.num_inds,
            )
        else:
            vcf_attributes = self.popgenio._vcf_attributes

        aln_filename = aln.name
        self.popgenio.write_phylip(
            aln_filename, snp_data=self.alignment, samples=samples
        )
        aln.close()

        popmap = tempfile.NamedTemporaryFile(delete=False)
        popmap_filename = popmap.name

        if self.sample_indices is not None:
            self.popgenio.popmap = {
                k: v for k, v in self.popgenio.popmap.items() if k in samples
            }

        with open(popmap_filename, "w") as fout:
            for key, value in self.popgenio.popmap.items():
                fout.write(f"{key}\t{value}\n")
        popmap.close()

        inputs = self.popgenio.inputs
        inputs["popmapfile"] = popmap_filename
        inputs["filename"] = aln_filename
        inputs["loci_indices"] = self.loci_indices
        inputs["sample_indices"] = self.sample_indices
        inputs["filetype"] = "phylip"
        inputs["vcf_attributes"] = vcf_attributes
        inputs["verbose"] = False

        # Create a new object with the filtered alignment.
        new_popgenio = GenotypeData(**inputs)

        new_popgenio.filetype = self.popgenio.filetype
        new_popgenio.verbose = self.popgenio.verbose

        if self.popgenio.filetype == "vcf":
            new_popgenio.vcf_attributes = vcf_attributes

        # When done, delete the file manually using os.unlink
        os.unlink(aln_filename)
        os.unlink(popmap_filename)

        return new_popgenio

    def calc_missing_proportions(
        self,
        alignment_array,
        missing_chars=["N", "-", ".", "?"],
        calculate_stdev=False,
        is_sample_filter=False,
    ):
        """Calculates the proportion of missing data in each column or row of the alignment.

        Args:
            alignment_array (numpy.ndarray): The alignment array.

            missing_chars (list, optional): List of characters representing missing data. Defaults to ["N", "-", ".", "?"].

            calculate_stdev (bool, optional): Whether to calculate the standard deviation of the missing data proportions. Defaults to False.

            is_sample_filter (bool, optional): Whether the calculation is for a sample filter. If True, calculates proportions per column (axis=1). If False, calculates proportions per row (axis=0). Defaults to False.

        Returns:
            numpy.ndarray or tuple: The proportion of missing data per column or row. If calculate_stdev is True, returns a tuple with the proportions and the standard deviation.

        Raises:
            None
        """
        axis = 1 if is_sample_filter else 0

        new_missing_counts = np.sum(
            np.isin(alignment_array, missing_chars), axis=axis
        )

        # Calculate the mean missing data proportion among all the columns
        missing_prop = new_missing_counts / alignment_array.shape[axis]

        if calculate_stdev:
            std_missing_prop = np.std(
                new_missing_counts / alignment_array.shape[axis]
            )

        res = (
            (missing_prop, std_missing_prop)
            if calculate_stdev
            else missing_prop
        )

        return res

    def filter_missing(self, threshold, alignment=None, return_props=False):
        """Filters out columns with missing data proportion greater than the given threshold.

        Args:
            threshold (float): The maximum missing data proportion allowed.

            alignment (MultipleSeqAlignment, optional): The alignment to be filtered. Defaults to the stored alignment.

            return_props (bool, optional): Whether to return the mean missing data proportion among all columns after filtering. Defaults to False.

        Returns:
            MultipleSeqAlignment or tuple: The filtered alignment. If return_props is True, returns a tuple with the filtered alignment, the mean missing data proportion, and None.

        Raises:
            TypeError: If threshold is not a float value.

            ValueError: If threshold is not between 0.0 and 1.0 inclusive.
        """
        if alignment is None:
            alignment = self.alignment

        alignment_array = alignment

        missing_counts = np.sum(
            np.isin(alignment_array, ["N", "-", ".", "?"]), axis=0
        )
        mask = missing_counts / alignment_array.shape[0] <= threshold

        # Get the indices of the True values in the mask
        mask_indices = [i for i, val in enumerate(mask) if val]

        # Apply the mask to filter out columns with a missing proportion greater than the threshold
        filtered_alignment_array = alignment_array[:, mask]

        if return_props:
            missing_prop = self.calc_missing_proportions(
                filtered_alignment_array
            )
            return (
                filtered_alignment_array,
                missing_prop,
                None,
            )
        else:
            return filtered_alignment_array, mask_indices

    def filter_missing_pop(
        self, max_missing, alignment, populations=None, return_props=False
    ):
        """Filters out sequences from the alignment that have more than a given proportion of missing data in any given population.

        Args:
            max_missing (float): The maximum missing data proportion allowed.

            alignment (MultipleSeqAlignment): The alignment to be filtered.

            populations (dict, optional): A dictionary mapping population names to sample IDs. Defaults to None.

            return_props (bool, optional): Whether to return the mean and standard deviation of missing data proportions for each population after filtering. Defaults to False.

        Returns:
            MultipleSeqAlignment or tuple: The filtered alignment. If return_props is True, returns a tuple with the filtered alignment, a dictionary of mean missing data proportions for each population, and a dictionary of standard deviations of missing data proportions for each population.

        Raises:
            None
        """
        if populations is None:
            populations = self.popmap_inverse

        alignment_array = alignment

        def missing_data_proportion(column, indices):
            missing_count = sum(
                column[i] in {"N", "-", ".", "?"} for i in indices
            )
            return missing_count / len(indices)

        def exceeds_threshold(column):
            missing_props = {}
            flaglist = []
            for pop, sample_ids in populations.items():
                # Get sampleID indices for given population, if not removed
                # with filter_missing_sample
                indices = [
                    i
                    for i, sid in enumerate(sample_ids)
                    if sid in self.samples
                ]
                missing_prop = missing_data_proportion(column, indices)

                if missing_prop >= max_missing:
                    flagged = True
                else:
                    missing_props[pop] = missing_prop
                    flagged = False
                flaglist.append(flagged)
            return flaglist, missing_props

        mask_and_missing_props = np.array(
            [exceeds_threshold(col) for col in alignment_array.T], dtype=object
        )
        mask = np.all(
            np.array([mmp[0] for mmp in mask_and_missing_props], dtype=bool),
            axis=1,
        )

        # Get the indices of the True values in the mask
        mask_indices = [i for i, val in enumerate(mask) if not val]

        filtered_alignment_array = alignment_array[:, ~mask]

        if return_props:
            mean_missing_props = [mmp[1] for mmp in mask_and_missing_props]

            key_values = {
                key: [
                    d.get(key)
                    for d in mean_missing_props
                    if d.get(key) is not None
                ]
                for key in set().union(*mean_missing_props)
            }

            missing_props = {k: v for k, v in key_values.items()}
            std_missing_props = {k: np.std(v) for k, v in key_values.items()}
            return (
                filtered_alignment_array,
                missing_props,
                std_missing_props,
            )
        else:
            return filtered_alignment_array, mask_indices

    def filter_missing_sample(
        self, threshold, alignment=None, return_props=False
    ):
        """Filters out sequences with missing data proportion greater than the given threshold.

        Args:
            threshold (float): The maximum missing data proportion allowed for each sequence.

            alignment (MultipleSeqAlignment, optional): The alignment to be filtered. Defaults to the stored alignment.

            return_props (bool, optional): Whether to return the mean missing data proportion among all sequences after filtering. Defaults to False.

        Returns:
            MultipleSeqAlignment or tuple: The filtered alignment. If return_props is True, returns a tuple with the filtered alignment, the mean missing data proportion among all sequences, and the indices of the filtered sequences.

        Raises:
            TypeError: If threshold is not a float value.

            ValueError: If threshold is not between 0.0 and 1.0 inclusive.
        """

        if alignment is None:
            alignment = self.alignment

        alignment_array = alignment

        missing_counts = np.sum(
            np.isin(alignment_array, ["N", "-", ".", "?"]), axis=1
        )

        mask = missing_counts / alignment_array.shape[1] <= threshold

        # Apply the mask to filter out sequences with a missing proportion greater than the threshold
        filtered_alignment_array = alignment_array[mask, :]

        # Get the indices of the True values in the mask
        mask_indices = [i for i, val in enumerate(mask) if val]

        # Convert the filtered alignment array back to a list of SeqRecord objects
        filtered_alignment = [
            filtered_alignment_array[i, :]
            for i, index in enumerate(mask_indices)
        ]

        if return_props:
            missing_prop = self.calc_missing_proportions(
                filtered_alignment_array, is_sample_filter=True
            )
            return filtered_alignment, missing_prop, mask_indices
        else:
            return filtered_alignment, mask_indices

    def filter_minor_allele_frequency(
        self, min_maf, alignment=None, return_props=False
    ):
        """Filters out loci (columns) where the minor allele frequency is below the threshold.

        Args:
            min_maf (float): The minimum minor allele frequency allowed.

            alignment (MultipleSeqAlignment, optional): The alignment to be filtered. Defaults to the stored alignment.

            return_props (bool, optional): Whether to return the mean missing data proportion among all columns after filtering. Defaults to False.

        Returns:
            MultipleSeqAlignment or tuple: The filtered alignment. If return_props is True, returns a tuple with the filtered alignment, the mean missing data proportion among all columns, and the minor allele frequencies.

        Raises:
            TypeError: If min_maf is not a float value.

            ValueError: If min_maf is not between 0.0 and 1.0 inclusive.
        """
        if alignment is None:
            alignment = self.alignment
        alignment_array = alignment

        def count_bases(column):
            base_count = {
                "A": 0,
                "C": 0,
                "G": 0,
                "T": 0,
            }
            for base in column:
                if base in base_count:
                    base_count[base] += 1
                elif base not in {"N", "-", ".", "?"}:
                    try:
                        ambig_bases = SeqUtils.IUPACData.ambiguous_dna_values[
                            base
                        ]
                        for ambig_base in ambig_bases:
                            base_count[ambig_base] += 1
                    except KeyError:
                        pass
            return base_count

        def minor_allele_frequency(column):
            counts = count_bases(column)
            # Remove counts of "N", "-", and "." characters from the counts dictionary
            counts = {
                base: count
                for base, count in counts.items()
                if base not in {"N", "-", ".", "?"}
            }

            if not counts or all(v == 0 for v in counts.values()):
                return 0

            # Sort the counts by their values
            sorted_counts = sorted(counts.values(), reverse=True)
            total = sum(sorted_counts)

            # Calculate the frequencies of each allele
            freqs = [count / total for count in sorted_counts]

            # Return the frequency of the second most common allele (the minor allele)
            return freqs[1] if len(freqs) > 1 else 0

        maf = np.apply_along_axis(minor_allele_frequency, 0, alignment_array)
        mask = maf >= min_maf

        # Get the indices of the True values in the mask
        mask_indices = [i for i, val in enumerate(mask) if val]
        filtered_alignment_array = alignment_array[:, mask]

        if return_props:
            missing_prop = self.calc_missing_proportions(
                filtered_alignment_array
            )
            return (
                filtered_alignment_array,
                missing_prop,
                maf,
            )
        else:
            return filtered_alignment_array, mask_indices

    def filter_non_biallelic(
        self, threshold=None, alignment=None, return_props=False
    ):
        """Filters out loci (columns) that are not biallelic.

        Args:
            threshold (None, optional): Not used.

            alignment (MultipleSeqAlignment, optional): The alignment to be filtered. Defaults to the stored alignment.

            return_props (bool, optional): Whether to return additional information. Defaults to False.

        Returns:
            MultipleSeqAlignment or tuple: The filtered alignment. If return_props is True, returns a tuple with the original missing data proportions, filtered missing data proportions, and the mask indicating the biallelic columns.

        Raises:
            None
        """

        if alignment is None:
            alignment = self.alignment

        # Convert the input alignment to a numpy array of sequences
        alignment_array = alignment

        iupac = {
            "R": ("A", "G"),
            "Y": ("C", "T"),
            "S": ("G", "C"),
            "W": ("A", "T"),
            "K": ("G", "T"),
            "M": ("A", "C"),
        }

        def count_unique_bases(column):
            """
            Args:
                column (str): A column of bases from an alignment.

            Returns:
                int: The number of unique bases in the column, excluding ambiguous and missing bases.
            """
            base_count = {
                "A": 0,
                "C": 0,
                "G": 0,
                "T": 0,
                "U": 0,
            }

            for base in column:
                if base in base_count:
                    base_count[base] += 1
                elif base in iupac:
                    base1, base2 = iupac[base]
                    base_count[base1] += 1
                    base_count[base2] += 1
                # Ignore "N", "-", and "." bases

            return len([count for count in base_count.values() if count > 0])

        unique_base_counts = np.apply_along_axis(
            count_unique_bases, 0, alignment_array
        )
        mask = unique_base_counts == 2

        # Get the indices of the True values in the mask
        mask_indices = [i for i, val in enumerate(mask) if val]

        # Apply the mask to filter non-biallelic columns
        filtered_alignment_array = alignment_array[:, mask]

        if return_props:
            orig_missing_prop = self.calc_missing_proportions(alignment_array)
            filt_missing_prop = self.calc_missing_proportions(
                filtered_alignment_array
            )
            return (
                orig_missing_prop,
                filt_missing_prop,
                mask,
            )
        else:
            return filtered_alignment_array, mask_indices

    def count_iupac_alleles(self, column):
        """Counts the number of occurrences of each IUPAC ambiguity code in a column of nucleotide sequences.

        Args:
            column (str): A string representing a column of nucleotide sequences.

        Returns:
            dict: A dictionary with the counts of the unambiguous nucleotide bases.

        Raises:
            None
        """
        iupac = {
            "A": "A",
            "C": "C",
            "G": "G",
            "T": "T",
            "U": "T",
            "R": "AG",
            "Y": "CT",
            "S": "GC",
            "W": "AT",
            "K": "GT",
            "M": "AC",
            "B": "CGT",
            "D": "AGT",
            "H": "ACT",
            "V": "ACG",
            "N": "ACGT",
        }

        counts = {"A": 0, "C": 0, "G": 0, "T": 0}

        for base in column:
            if base in iupac:
                for allele in iupac[base]:
                    counts[allele] += 1

        return counts

    def filter_monomorphic(
        self, threshold=None, alignment=None, return_props=False
    ):
        """Filters out monomorphic sites from an alignment.

        Args:
            alignment (Bio.Align.MultipleSeqAlignment): The alignment to be filtered.

        Returns:
            filtered_alignment (Bio.Align.MultipleSeqAlignment): The filtered alignment.

        Raises:
            ValueError: If no loci remain in the alignment.
        """

        if alignment is None:
            alignment = self.alignment

        def is_monomorphic(column):
            """
            Determines if a column in an alignment is monomorphic.

            Args:
                column: a list of bases representing a column in an alignment

            Returns:
                A boolean indicating whether the column is monomorphic.
            """
            column_list = column.tolist()
            alleles = set(column_list)

            # Remove any ambiguity code
            alleles.discard("N")

            # Count the number of valid alleles
            valid_alleles = [
                allele for allele in alleles if allele not in ["-", ".", "?"]
            ]

            return len(valid_alleles) <= 1

        alignment_array = alignment.astype(str)

        if alignment_array.shape[1] > 0:
            mask = np.apply_along_axis(is_monomorphic, 0, alignment_array)
            filtered_alignment_array = alignment_array[:, ~mask]

            # Get the indices of the True values in the mask
            mask_indices = [i for i, val in enumerate(mask) if not val]

        else:
            raise ValueError(
                "No loci remain in the alignment. Try adjusting the filtering paramters."
            )

        if return_props:
            orig_missing_prop = self.calc_missing_proportions(alignment_array)
            filt_missing_prop = self.calc_missing_proportions(
                filtered_alignment_array
            )
            return (
                orig_missing_prop,
                filt_missing_prop,
                mask,
            )
        else:
            return filtered_alignment_array, mask_indices

    @staticmethod
    def resolve_ambiguity(base):
        """
        Resolves an IUPAC ambiguity code to the set of possible nucleotides it represents.

        Args:
            base (str): A single IUPAC character.

        Returns:
            set: A set of possible nucleotides represented by the IUPAC character.
        """
        iupac_dict = {
            "A": {"A"},
            "C": {"C"},
            "G": {"G"},
            "T": {"T"},
            "U": {"T"},
            "R": {"A", "G"},
            "Y": {"C", "T"},
            "S": {"G", "C"},
            "W": {"A", "T"},
            "K": {"G", "T"},
            "M": {"A", "C"},
            "B": {"C", "G", "T"},
            "D": {"A", "G", "T"},
            "H": {"A", "C", "T"},
            "V": {"A", "C", "G"},
            "N": {"A", "C", "G", "T"},
            "-": {"-"},
        }
        return iupac_dict.get(base.upper(), {"N"})

    def filter_singletons(
        self, threshold=None, alignment=None, return_props=False
    ):
        """
        Filters out singletons from an alignment.

        Args:
            alignment (Bio.Align.MultipleSeqAlignment): The alignment to be filtered.

        Returns:
            filtered_alignment (Bio.Align.MultipleSeqAlignment): The filtered alignment.
        """

        if alignment is None:
            alignment = self.alignment

        def is_singleton(column):
            """
            Determines if a column in an alignment is a singleton.

            Args:
                column: a list of bases representing a column in an alignment

            Returns:
                A boolean indicating whether the column is a singleton, meaning that there is only one
            variant in the column and it appears only once.
            """
            column_list = column.tolist()
            alleles = {
                allele
                for allele in column_list
                if allele not in ["N", "-", ".", "?"]
            }
            allele_count = {
                allele: column_list.count(allele) for allele in alleles
            }

            if len(alleles) == 2:
                min_allele = min(alleles, key=lambda x: allele_count[x])
                return allele_count[min_allele] == 1
            return False

        alignment_array = alignment.astype(str)

        if alignment_array.shape[1] > 0:
            mask = np.apply_along_axis(is_singleton, 0, alignment_array)
            filtered_alignment_array = alignment_array[:, ~mask]
            # Get the indices of the True values in the mask
            mask_indices = [i for i, val in enumerate(mask) if not val]
        else:
            raise ValueError(
                "No loci remain in the alignment. Try adjusting the filtering paramters."
            )

        if return_props:
            orig_missing_prop = self.calc_missing_proportions(alignment_array)
            filt_missing_prop = self.calc_missing_proportions(
                filtered_alignment_array
            )
            return (
                orig_missing_prop,
                filt_missing_prop,
                mask,
            )
        else:
            return filtered_alignment_array, mask_indices

    def get_population_sequences(self, population):
        """
        Returns a list of sequence strings for a specific population.

        Args:
            population: str, the name of the population to retrieve sequences for.

        Returns:
            population_sequences: list, a list of sequence strings for the specified population.

        Raises:
            ValueError: If the specified population is not found in the object's list of populations.
        """
        population_indices = [
            i for i, pop in enumerate(self.populations) if pop == population
        ]
        alignment_array = self.alignment
        population_sequences = alignment_array[population_indices, :]
        return population_sequences.tolist()

    @staticmethod
    def print_filtering_report(
        before_alignment, after_alignment, loci_removed_per_step
    ):
        """
        Print a filtering report to the terminal.

        Args:
            before_alignment (list): The original alignment before filtering.

            after_alignment (list): The alignment after filtering.

            loci_removed_per_step (list of tuples): A list of tuples, where each tuple contains the name of a filtering step and the number of loci removed during that step.

        Returns:
            None.

        Raises:
            ValueError: If there is no data left after filtering, which could indicate an issue with the filtering or with the provided filtering parameters.

        Note:
            The function also raises a warning if none of the filtering arguments were changed from their defaults, in which case the alignment will not be filtered.
        """
        num_loci_before = len(before_alignment[0])
        num_samples_before = len(before_alignment)
        num_loci_after = len(after_alignment[0])
        num_samples_after = len(after_alignment)
        samples_removed = num_samples_before - num_samples_after

        def missing_data_percent(msa):
            total = len(msa) * len(msa[0])
            if total == 0:
                # for name, loci_removed in loci_removed_per_step:
                #     print(f"  {name}: {loci_removed}")
                raise ValueError(
                    "There is no data left after filtering. This can indicate an issue with the filtering or with the provided filtering parameters."
                )
            missing = np.count_nonzero(np.isin(msa, ["N", "-", ".", "?"]))
            return (missing / total) * 100

        missing_data_before = missing_data_percent(before_alignment)
        missing_data_after = missing_data_percent(after_alignment)

        print("Filtering Report:")
        print(f"  Loci before filtering: {num_loci_before}")
        print(f"  Samples before filtering: {num_samples_before}")

        if (
            all([x[1] == 0 for x in loci_removed_per_step])
            and samples_removed == 0
            and missing_data_before == missing_data_after
        ):
            warnings.warn(
                "The alignment was unchanged. Note that if none of the filtering arguments were changed from defaults, the alignment will not be filtered."
            )

        for name, loci_removed in loci_removed_per_step:
            print(f"  {name}: {loci_removed}")
        print(f"  Samples removed: {samples_removed}")
        print(f"  Loci remaining: {num_loci_after}")
        print(f"  Samples remaining: {num_samples_after}")
        print(f"  Missing data before filtering: {missing_data_before:.2f}%")
        print(f"  Missing data after filtering: {missing_data_after:.2f}%\n\n")

    def plot_missing_data_thresholds(
        self,
        output_file,
        num_thresholds=5,
        num_maf_thresholds=10,
        max_maf_threshold=0.2,
        show_plot_inline=False,
        plot_dir="plots",
        plot_fontsize=28,
        plot_ticksize=20,
        plot_ymin=0.0,
        plot_ymax=1.0,
        plot_legend_loc="upper left",
    ):
        """
        Plots the missing data and MAF proportions for different filtering thresholds.

        Args:
            output_file (str): The name of the output plot file.
            num_thresholds (int, optional): The number of thresholds to use for filtering. Defaults to 5.

            num_maf_thresholds (int, optional): The number of minor allele frequency (MAF) thresholds to use for filtering. Defaults to 10.
            max_maf_threshold (float, optional): The maximum MAF threshold to use for filtering. Defaults to 0.2.

            show_plot_inline (bool, optional): Whether to show the plot inline. Defaults to False.

            plot_dir (str, optional): The directory to save the plot. Defaults to "plots".

            plot_fontsize (int, optional): The fontsize for plot labels. Defaults to 28.

            plot_ticksize (int, optional): The fontsize for plot ticks. Defaults to 20.

            plot_ymin (float, optional): The minimum y-axis value for the plot. Defaults to 0.0.

            plot_ymax (float, optional): The maximum y-axis value for the plot. Defaults to 1.0.

            plot_legend_loc (str, optional): The location of the plot legend. Defaults to "upper left".

        Returns:
            None.

        Raises:
            None.
        """
        thresholds = np.linspace(0.1, 1, num=num_thresholds, endpoint=True)
        maf_thresholds = np.linspace(
            0.0, max_maf_threshold, num=num_maf_thresholds, endpoint=True
        )
        sample_missing_data_proportions = []
        global_missing_data_proportions = []
        population_missing_data_proportions = []
        maf_per_threshold = []
        maf_props_per_threshold = []
        mask_indices = []

        for threshold, maf_threshold in zip(thresholds, maf_thresholds):
            mask_idx, sample_missing_prop = self.filter_per_threshold(
                self.filter_missing_sample,
                threshold,
                self.alignment,
                return_props=True,
                is_maf=True,
            )

            global_missing_prop = self.filter_per_threshold(
                self.filter_missing,
                threshold,
                self.alignment,
                return_props=True,
            )

            pop_missing_props = self.filter_per_threshold(
                self.filter_missing_pop,
                threshold,
                self.alignment,
                populations=self.popmap_inverse,
                return_props=True,
            )

            # Get MAF for each threshold
            maf_freqs, maf_props = self.filter_per_threshold(
                self.filter_minor_allele_frequency,
                maf_threshold,
                self.alignment,
                is_maf=True,
                return_props=True,
            )

            sample_missing_data_proportions.append(sample_missing_prop)
            global_missing_data_proportions.append(global_missing_prop)
            population_missing_data_proportions.append(pop_missing_props)
            maf_per_threshold.append(maf_freqs)
            maf_props_per_threshold.append(maf_props)

            # For sample-level filtering.
            mask_indices.append(mask_idx)

        (
            mono_orig_props,
            mono_filt_props,
            mono_mask,
        ) = self.filter_per_threshold(
            self.filter_monomorphic,
            self.alignment,
            is_bool=True,
            return_props=True,
        )

        (
            bi_orig_props,
            bi_filt_props,
            biallelic_mask,
        ) = self.filter_per_threshold(
            self.filter_non_biallelic,
            self.alignment,
            is_bool=True,
            return_props=True,
        )

        (
            sing_orig_props,
            sing_filt_props,
            singleton_mask,
        ) = self.filter_per_threshold(
            self.filter_singletons,
            self.alignment,
            is_bool=True,
            return_props=True,
        )

        def generate_df(props, thresholds, dftype, orig_props=None, mask=None):
            flattened_proportions = []
            flattened_thresholds = []

            if (orig_props is None and mask is not None) or (
                orig_props is not None and mask is None
            ):
                raise TypeError(
                    "orig_props and mask must both be either NoneType or not NoneType"
                )

            if orig_props is None:
                p = deepcopy(props)
            else:
                p = deepcopy(orig_props)

            if mask is None:
                for i, (threshold, array) in enumerate(zip(thresholds, p)):
                    flattened_proportions.extend(array)
                    flattened_thresholds.extend(
                        [f"{threshold:.2f}"] * len(array)
                    )

                df = pd.DataFrame(
                    {
                        "Threshold": flattened_thresholds,
                        "Proportion": flattened_proportions,
                        "Type": dftype,
                    }
                )

            else:
                df = pd.DataFrame(
                    {
                        "Proportion": orig_props,
                        "Filtered": mask,
                        "Type": dftype,
                    }
                )

            return df

        def generate_population_df(props, thresholds):
            df_list = []
            for threshold, prop_dict in zip(thresholds, props):
                for population, proportions in prop_dict.items():
                    temp_df = pd.DataFrame(
                        {
                            "Threshold": [f"{threshold:.2f}"]
                            * len(proportions),
                            "Proportion": proportions,
                            "Type": [population] * len(proportions),
                        }
                    )
                    df_list.append(temp_df)

            df = pd.concat(df_list, ignore_index=True)
            return df

        df_sample = generate_df(
            sample_missing_data_proportions, thresholds, "Sample"
        )

        df_global = generate_df(
            global_missing_data_proportions, thresholds, "Global"
        )

        df_populations = generate_population_df(
            population_missing_data_proportions, thresholds
        )

        df_maf = generate_df(maf_props_per_threshold, maf_thresholds, "MAF")

        df_mono = generate_df(
            mono_filt_props,
            thresholds,
            "Monomorphic",
            mono_orig_props,
            mono_mask,
        )

        df_biallelic = generate_df(
            bi_filt_props,
            thresholds,
            "Biallelic",
            bi_orig_props,
            biallelic_mask,
        )

        df_singleton = generate_df(
            sing_filt_props,
            thresholds,
            "Singleton",
            sing_orig_props,
            singleton_mask,
        )

        # combine the two dataframes
        df = pd.concat([df_sample, df_global])
        df2 = pd.concat([df_mono, df_biallelic, df_singleton])

        Plotting.plot_filter_report(
            df,
            df2,
            df_populations,
            df_maf,
            maf_per_threshold,
            maf_props_per_threshold,
            plot_dir,
            output_file,
            plot_fontsize,
            plot_ticksize,
            plot_ymin,
            plot_ymax,
            plot_legend_loc,
            show_plot_inline,
        )

    def filter_per_threshold(
        self, filter_func, *args, is_maf=False, is_bool=False, **kwargs
    ):
        """
        Filters the alignment using the provided filter function for multiple thresholds.

        Args:
            filter_func (callable): The filtering function to apply.

            args: Positional arguments to pass to the filter function.

            is_maf (bool, optional): Indicates whether the filter is for minor allele frequency (MAF). Defaults to False.

            is_bool (bool, optional): Indicates whether the filter returns boolean values. Defaults to False.

            kwargs: Keyword arguments to pass to the filter function.

        Returns:
            tuple: A tuple containing the filtered results based on the filter function.
                If is_bool is True:
                    (orig_props, filt_props, mask)
                If is_maf is True:
                    (freqs, props)
                Otherwise:
                    props

        Raises:
            None.
        """
        if is_bool:
            orig_props, filt_props, mask = filter_func(*args, **kwargs)
            res = (orig_props, filt_props, mask)
        else:
            _, props, freqs = filter_func(*args, **kwargs)
            res = (freqs, props) if is_maf else props
        return res

    @property
    def alignment(self):
        """
        Gets the alignment data.

        Returns:
            numpy.ndarray: The alignment data as a numpy array.

        Raises:
            None.
        """
        if isinstance(self._alignment, MultipleSeqAlignment):
            a = np.array([list(str(record.seq)) for record in self._alignment])
        else:
            a = np.array(self._alignment)
        return a

    @alignment.setter
    def alignment(self, value):
        """
        Sets the alignment data.

        Args:
            value (MultipleSeqAlignment or numpy.ndarray): The alignment data to be set.

        Returns:
            None.

        Raises:
            None.
        """
        if isinstance(value, MultipleSeqAlignment):
            self._alignment = np.array(
                [list(str(record.seq)) for record in value]
            )
        else:
            self._alignment = value

    @property
    def msa(self):
        """
        Gets the multiple sequence alignment (MSA) data.

        Returns:
            The multiple sequence alignment (MSA) data.

        Raises:
            None.
        """
        return self._msa

    @msa.setter
    def msa(self, value):
        """
        Sets the multiple sequence alignment (MSA) data.

        Args:
            value: The multiple sequence alignment (MSA) data to be set.

        Returns:
            None.

        Raises:
            None.
        """
        self._msa = value

    @property
    def population_sequences(self):
        """
        Returns a dictionary of population sequences.

        The dictionary keys are the names of the populations, and the values are the corresponding sequences for each population.

        Sequences are in the form of a list of strings, where each string is a sequence for a given sample.

        Returns:
            dict: A dictionary of population sequences, where each key is the name of a population and the corresponding value is a list of sequences.

        Raises:
            None.
        """
        population_sequences = {}
        for population_name in self.populations:
            population_sequences[
                population_name
            ] = self.get_population_sequences(population_name)
        return population_sequences

    @classmethod
    def print_cletus(cls):
        """Prints ASCII art of Cletus from the Simpsons (silly inside joke)."""
        # ASCII Cletus
        cletus_ascii = r"""
                                                      T                                            
                                                     M                                             
                                              +MI   :                                              
                                                  ?.M ?MM888DMD,                                   
                                                   M88888888888888DN.                              
                                                 M88888888888888888888N~                           
                                                8888888888888888888888888M                         
                                              ,88888888888888888888888888888O                      
                                             .88888888888888888888888888888888M                    
                                            =88888888888888888888888888888888888M                  
                                           M88888888888888888888888888888888888888M                
                                         D888888888N88888888888888888888888888888888O              
                                             M8888MID888888888888888888888888N8888888D             
                                            M8888MIIM8888M8888888888888888888MIIMD8888D.           
                                           7888MDIIIM888MII888888888888888888IIIIIN ~M8M           
                                           = IIIIIID88TTIIIM888DI8888M888888?IIIIIII.              
                                             MIIIMIIMIIIII?M88M ,888IIMDIDDIIIM      ?             
                                             M?IIIIIIIIII.      M8MMIIIIIIIII.        M            
                                             MINIIIIIIII       ,    NIIIIIII$     R    N           
                                          .MI?N8IIIIIIIN      &7    7IIIIIIM      M7  M           
                                         .IIIIIIIMIIIII=      &&     DIIIIIIIIM        D           
                                         MIIIIIIIIIIIIIM             ?IIIIIIIIIIM     M            
                                         ?IIIDIIIIIIIIIIN           MIIIIIIIIIIIIIN M.             
                                         IIIIIOIIIIIIIIIIM        ,IIIIIIIIIIIIIIIII$              
                                         7IIIIIMIIIIIIIIIII?NMMM8IMIIIIIIIIIIIIIIIIIIM             
                                         MIIIIIIIIIIIIIIIIMMMMM7IIIIIIIIIIIIIIIIIIIIIID            
                                         .IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIM            
                                          NIIIIIIIIIIIIIIMIIIIIIIIIIIIIIIIIIIIMM?IIMM             
                                           NIIIIIIIIIIII7IIIIIIIIIMIIIIIIIIIIIIIIIIIIID.           
                                             IMMNIIIIIIIIIIIIIIII7IIIIMIIIMIIIIIII?II?IIIIIII8.    
                                                IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIOIII?IIIIIIIIIII    
                                                IIIIIIIIINCLETUSD7IIIIIIIIIIIIIIIIIIIIIIIIIIIIM    
                                                IIIIIIMMMMMMM      O   IMM7IIIIIIIIIIIIIIIIID      
                                                IIIIMMMMMMMMMMM::MIM     N   .OMMZIIIIIDN,         
                                               :IIIMMMMMMMMMMMMIIIIM.,      M     ..             
                                               NIIINZDOZZMMMMMDIII~             MN~               
                                               IIIIMZZZZZZMMMIIIM                                 
                                              DIIIIIIMMMMZIIIIIIIIM                                
                                              IIIIIIIIIIIIIIIIIIIII,                               
                                             OIIIIIIIIIIIIIIIIIIIIN                                
                                             IIIIIIIIIIIIIIIIID..                                  
                                            MIIIIIIIIIIIIIIII,                                     
                                          ,IIIIIIIIIIIIIIIII?                                      
                                        DIIMIIIIIIIIIIIIIIIN                                       
                                    DOZZ7IIMIIIIIIIIIIIIIIO                                        
                                :M7MZZZLOLI7IIIIIIIIIIIIII~                                        
                             OYIIIIIZZZZMIIIIZIIIIIIIIIIIMN                                        
                         ,MIIIIIIIIIMZZZZMIIIIIMIIIIIIIIIIIM                                       
                       MIIIIIIIIIIIIIZZZZMMIIIIII7MMNMMIIIIZM                                      
                    MIIIIIIIIIIIIIIIINZZZOIIMIIIIIIIIIIIII7ZZIO                                    
                 .MIIIIIIIIIIIIIIIIIIMZZZZIIIINOIIIIIIIIIMZZZIIIM                                  
                   
        """
        print(cletus_ascii)
