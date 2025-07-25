from typing import TYPE_CHECKING, Tuple

import numpy as np

from snpio import GenotypeEncoder
from snpio.popgenstats.dstats_base import DStatsBase, DStatsConfig, DStatsResults
from snpio.popgenstats.numba_helpers import _compute_dfoil, _execute_bootstrap_dfoil
from snpio.utils.logging import LoggerManager

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


class DfoilStats(DStatsBase):
    """5-taxon DFOIL via weighted pattern sums + bootstrapping.

    This class computes the DFOIL statistic for a 5-taxon model, which is used to detect introgression between populations.
    """

    def __init__(self, genotype_data: "GenotypeData", *, verbose=False, debug=False):
        """Initialize the DfoilStats class.

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
        population1: np.ndarray,
        population2: np.ndarray,
        population3: np.ndarray,
        population4: np.ndarray,
        outgroup: np.ndarray,
        n_boot: int,
        seed: int | None = None,
        use_jackknife: bool = False,
        block_size: int = 500,
    ) -> Tuple[dict, np.ndarray]:
        """Calculate DFOIL statistics from 5 populations and an outgroup.

        This method performs the necessary computations to obtain the DFOIL statistics and their associated bootstrap or jackknife estimates.

        Args:
            population1 (np.ndarray): Genotype data for the first population.
            population2 (np.ndarray): Genotype data for the second population.
            population3 (np.ndarray): Genotype data for the third population.
            population4 (np.ndarray): Genotype data for the fourth population.
            outgroup (np.ndarray): Genotype data for the outgroup population.
            n_boot (int): Number of bootstrap replicates.
            seed (int | None): Random seed for reproducibility. If None, a random seed will be used.
            use_jackknife (bool): If True, uses jackknife resampling instead of bootstrap.
            block_size (int): Block size for jackknife resampling.

        Returns:
            Tuple[dict, np.ndarray]: A tuple containing the results of the DFOIL statistics calculation and the bootstrap results.
        """
        ge = GenotypeEncoder(self.genotype_data)
        geno012 = ge.genotypes_012.astype(int, copy=False)

        config = DStatsConfig(
            geno012=geno012,
            pop1=population1,
            pop2=population2,
            pop3=population3,
            pop4=population4,
            outgroup=outgroup,
            n_boot=n_boot,
            seed=seed,
            MISSING=-9,
            method="dfoil",
        )

        self.logger.debug(f"Config: {config.to_dict()}")

        # Subset the genotypes to only relevant populations.
        geno_sub, pops_mapped = self._map_geno_to_pops(geno012, config)

        # Get derived and ancestral population frequencies.
        arr = self._extract_pop_freqs(config, geno_sub, pops_mapped)

        # Compute observed DFOIL statistics
        dfoil = _compute_dfoil(arr)

        if use_jackknife:
            # Jackknife resampling
            boots = self.jackknife_indices(arr.shape[1], block_size=block_size)
        else:
            # Bootstrap the DFOIL statistics
            boots = self.bootstrap_indices(
                arr.shape[1], config.n_boot, seed=config.seed
            )

        boot_res = _execute_bootstrap_dfoil(
            boots.shape[0], arr, boots, n_dstats=config.n_dstats
        )

        zs, ps = self.zscore(dfoil, boot_res)
        x2s, p_x2s = self.chisq(dfoil, boot_res)

        self.results = DStatsResults(
            DFO=dfoil[0],
            DFI=dfoil[1],
            DOL=dfoil[2],
            DIL=dfoil[3],
            Z_DFO=zs[0],
            Z_DFI=zs[1],
            Z_DOL=zs[2],
            Z_DIL=zs[3],
            P_DFO=ps[0],
            P_DFI=ps[1],
            P_DOL=ps[2],
            P_DIL=ps[3],
            X2_DFO=x2s[0],
            X2_DFI=x2s[1],
            X2_DOL=x2s[2],
            X2_DIL=x2s[3],
            P_X2_DFO=p_x2s[0],
            P_X2_DFI=p_x2s[1],
            P_X2_DOL=p_x2s[2],
            P_X2_DIL=p_x2s[3],
            n_boot=n_boot,
            seed=seed,
            method="dfoil",
        )

        if self.verbose:
            self.results.log_results(self.logger)

        return self.results.to_dict(), boot_res
