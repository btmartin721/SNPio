from typing import TYPE_CHECKING, Tuple

import numpy as np

from snpio.popgenstats.dstats_base import DStatsBase, DStatsConfig, DStatsResults
from snpio.popgenstats.numba_helpers import _compute_d, _execute_bootstrap_d
from snpio.utils.logging import LoggerManager

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


class PattersonDStats(DStatsBase):
    """4-taxon Patterson's D-statistic via weighted pattern sums + faithful bootstrapping.

    This class computes the D-statistic for a 4-taxon model, which is used to detect introgression between populations.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        geno012: np.ndarray,
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initialize the PattersonDStats class.

        Args:
            genotype_data (GenotypeData): The genotype data to analyze.
            geno012 (np.ndarray): Genotype matrix in 0/1/2 format.
            verbose (bool): If True, enables verbose logging.
            debug (bool): If True, enables debug logging.
        """
        super().__init__(genotype_data, verbose=verbose, debug=debug)
        self.logger = LoggerManager(
            __name__, prefix=genotype_data.prefix, verbose=verbose, debug=debug
        ).get_logger()

        self.geno012 = geno012

    def calculate(
        self,
        population1: str,
        population2: str,
        population3: str,
        outgroup: str,
        n_boot: int,
        seed: int | None = None,
        use_jackknife: bool = False,
        block_size: int = 500,
    ) -> Tuple[dict, np.ndarray]:
        """Calculate the Patterson's D-statistic for the specified populations.

        This method performs the necessary computations to obtain the D-statistic and its associated bootstrap or jackknife estimates.

        Args:
            population1 (str): Name of the first population.
            population2 (str): Name of the second population.
            population3 (str): Name of the third population.
            outgroup (str): Name of the outgroup population.
            n_boot (int): Number of bootstrap replicates.
            seed (int | None): Random seed for reproducibility. If None, a random seed will be used.
            use_jackknife (bool): If True, uses jackknife resampling instead of bootstrap.
            block_size (int): Block size for jackknife resampling.

        Returns:
            Tuple[dict, np.ndarray]: A tuple containing the results of the D-statistic calculation and the bootstrap results.
        """
        # 1. Create config
        config = DStatsConfig(
            geno012=self.geno012,
            pop1=population1,
            pop2=population2,
            pop3=population3,
            pop4=None,
            outgroup=outgroup,
            n_boot=n_boot,
            seed=seed,
            method="patterson",
            MISSING=-9,
        )

        self.logger.debug(f"Config: {config.to_dict()}")

        # Subset the genotypes to only relevant populations.
        geno_sub, pops_mapped = self._map_geno_to_pops(self.geno012, config)

        # Get derived and ancestral population frequencies.
        arr = self._extract_pop_freqs(config, geno_sub, pops_mapped)

        # Compute observed DFOIL statistics
        dobs = _compute_d(arr)

        if use_jackknife:
            # Jackknife resampling
            boots = self.jackknife_indices(arr.shape[1], block_size=block_size)
        else:
            # Bootstrap the DFOIL statistics
            boots = self.bootstrap_indices(
                arr.shape[1], config.n_boot, seed=config.seed
            )

        boot_res = _execute_bootstrap_d(
            boots.shape[0], arr, boots, n_dstats=config.n_dstats
        )

        observed_sequence = (dobs,)
        z_scores, p_values = self.zscore(observed_sequence, boot_res)
        chi_squares, p_chi_squares = self.chisq(observed_sequence, boot_res)

        # 12. Wrap results
        self.results = DStatsResults(
            n_boot=config.n_boot,
            seed=config.seed,
            D=dobs,
            Z=z_scores[0],
            P=p_values[0],
            X2=chi_squares[0],
            P_X2=p_chi_squares[0],
            method="patterson",
        )

        if self.verbose:
            self.results.log_results(self.logger)

        return self.results.to_dict(), boot_res
