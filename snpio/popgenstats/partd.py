from typing import TYPE_CHECKING, Tuple

import numpy as np

from snpio import GenotypeEncoder
from snpio.popgenstats.dstats_base import DStatsBase, DStatsConfig, DStatsResults
from snpio.popgenstats.numba_helpers import _compute_partd, _execute_bootstrap_partd
from snpio.utils.logging import LoggerManager

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


class PartitionedDStats(DStatsBase):
    """5-taxon Partitioned-D via weighted pattern sums + bootstrapping.

    This class computes the Partitioned-D statistic for a 5-taxon model, which is used to detect introgression between populations.
    """

    def __init__(self, genotype_data: "GenotypeData", *, verbose=False, debug=False):
        """Initialize the PartitionedDStats class.

        Args:
            genotype_data (GenotypeData): The genotype data to analyze.
            verbose (bool): If True, enables verbose logging.
            debug (bool): If True, enables debug logging.
        """
        super().__init__(genotype_data, verbose=verbose, debug=debug)
        self.logger = LoggerManager(
            __name__, prefix=genotype_data.prefix, verbose=verbose, debug=debug
        ).get_logger()

    def calculate(
        self,
        pop1: np.ndarray,
        pop2: np.ndarray,
        pop3a: np.ndarray,
        pop3b: np.ndarray,
        outgroup: np.ndarray,
        n_boot: int,
        seed: int | None = None,
        use_jackknife: bool = False,
        block_size: int = 500,
    ) -> Tuple[dict, np.ndarray]:
        """Calculate Partitioned-D statistics from 5 populations.

        This method performs the necessary computations to obtain the Partitioned-D statistics and their associated bootstrap or jackknife estimates.

        Args:
            pop1 (np.ndarray): Genotype data for the first population.
            pop2 (np.ndarray): Genotype data for the second population.
            pop3a (np.ndarray): Genotype data for the third population (first part).
            pop3b (np.ndarray): Genotype data for the third population (second part).
            outgroup (np.ndarray): Genotype data for the outgroup population.
            n_boot (int): Number of bootstrap replicates.
            seed (int | None): Random seed for reproducibility. If None, a random seed will be used.
            use_jackknife (bool): If True, uses jackknife resampling instead of bootstrap.
            block_size (int): Block size for jackknife resampling.

        Returns:
            Tuple[dict, np.ndarray]: A tuple containing the results of the Partitioned-D statistic calculation and the bootstrap results.
        """
        # — 1) load & subset —
        ge = GenotypeEncoder(self.genotype_data)
        geno012 = ge.genotypes_012.astype(int, copy=False)  # (N, S)

        config = DStatsConfig(
            geno012=geno012,
            pop1=pop1,
            pop2=pop2,
            pop3=pop3a,
            pop4=pop3b,
            outgroup=outgroup,
            n_boot=n_boot,
            seed=seed,
            MISSING=-9,
            method="partitioned",
        )

        # Subset the genotypes to only relevant populations.
        geno_sub, pops_mapped = self._map_geno_to_pops(geno012, config)

        # Get derived and ancestral population frequencies.
        arr = self._extract_pop_freqs(config, geno_sub, pops_mapped)

        # Compute observed Partitioned-D statistics
        dparts = _compute_partd(arr)

        if use_jackknife:
            # Jackknife resampling
            boots = self.jackknife_indices(arr.shape[1], block_size=block_size)
        else:
            # Bootstrap the Partitioned-D statistics
            boots = self.bootstrap_indices(
                arr.shape[1], config.n_boot, seed=config.seed
            )

        boot_res = _execute_bootstrap_partd(
            boots.shape[0], arr, boots, n_dstats=config.n_dstats
        )

        zs, ps = self.zscore(dparts, boot_res)
        x2s, p_x2s = self.chisq(dparts, boot_res)

        if self.verbose:
            self.log_statistics((dparts[0], dparts[1], dparts[2]), boot_res)

        # — return the results —
        self.results = DStatsResults(
            D1=dparts[0],
            D2=dparts[1],
            D12=dparts[2],
            Z_D1=zs[0],
            Z_D2=zs[1],
            Z_D12=zs[2],
            P_D1=ps[0],
            P_D2=ps[1],
            P_D12=ps[2],
            X2_D1=x2s[0],
            X2_D2=x2s[1],
            X2_D12=x2s[2],
            P_X2_D1=p_x2s[0],
            P_X2_D2=p_x2s[1],
            P_X2_D12=p_x2s[2],
            n_boot=n_boot,
            seed=seed,
            method="partitioned",
        )

        if self.verbose:
            self.results.log_results(self.logger)

        return self.results.to_dict(), boot_res
