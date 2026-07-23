import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from snpio.utils.logging import LoggerManager
from snpio.utils.multiqc_reporter import SNPioMultiQC
from snpio.utils.output_paths import OutputPaths

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData
    from snpio.plotting.plotting import Plotting


class GeneticDistance:
    """Class for computing pairwise Nei's genetic distance between populations with optional permutation and bootstrap inference."""

    def __init__(
        self,
        genotype_data: "GenotypeData",
        plotter: "Plotting",
        verbose: bool = False,
        debug: bool = False,
        seed: int | None = None,
    ) -> None:
        self.genotype_data = genotype_data
        self.plotter = plotter
        self.seed = seed

        self.outdir = OutputPaths.from_genotype_data(genotype_data).reports(
            "nei_distance"
        )

        self.outdir.mkdir(parents=True, exist_ok=True)

        logman = LoggerManager(
            __name__, self.genotype_data.prefix, debug=debug, verbose=verbose
        )
        self.logger = logman.get_logger()

        self.iupac_decoder = self.genotype_data.reverse_iupac_mapping
        self.snpio_mqc = SNPioMultiQC

    @staticmethod
    def _clean_inds(inds: list[str]) -> list[str]:
        """Filter out missing genotype codes.

        Args:
            inds (list[str]): Genotype codes.

        Returns:
            list[str]: Non-missing genotype codes.
        """
        missing_codes = {
            "",
            ".",
            "./.",
            ".|.",
            "N",
            "N/N",
            "N|N",
            "-",
            "?",
            "NA",
            "NAN",
            "NONE",
        }

        clean: list[str] = []

        for ind in inds:
            if ind is None:
                continue

            code = str(ind).strip().upper()

            if code in missing_codes:
                continue

            clean.append(code)

        return clean

    @staticmethod
    def _get_alleles(
        iupac_decoder: dict[str, tuple[str, ...]], iupac_list: list[str]
    ) -> list[str]:
        """Decode IUPAC genotype codes into diploid allele calls.

        Args:
            iupac_decoder: Mapping from IUPAC genotype codes to allele tuples.
            iupac_list: List of IUPAC genotype codes.

        Returns:
            Flat list of alleles.

        Notes:
            Unambiguous bases are treated as diploid homozygotes. For example,
            "A" is decoded as ("A", "A"), not ("A",).
        """
        alleles: list[str] = []
        homozygous_bases = {"A", "C", "G", "T"}

        for iupac_code in iupac_list:
            code = str(iupac_code).strip().upper()

            if code in homozygous_bases:
                alleles.extend((code, code))
                continue

            decoded = iupac_decoder.get(code)

            if decoded is None:
                if "/" in code or "|" in code:
                    sep = "/" if "/" in code else "|"
                    parts = tuple(part.strip().upper() for part in code.split(sep))

                    if len(parts) == 2 and all(
                        part in homozygous_bases for part in parts
                    ):
                        alleles.extend(parts)

                continue

            decoded_tuple = tuple(str(allele).strip().upper() for allele in decoded)

            if len(decoded_tuple) == 1 and decoded_tuple[0] in homozygous_bases:
                alleles.extend((decoded_tuple[0], decoded_tuple[0]))
            elif len(decoded_tuple) == 2:
                alleles.extend(decoded_tuple)

        return alleles

    @staticmethod
    def _calculate_allele_freqs(
        matrix: np.ndarray,
        indices: list[int],
        iupac_decoder: dict[str, tuple[str, ...]],
    ) -> dict[int, dict[str, float]]:
        """Compute per-locus allele frequencies for a subset of individuals.

        Args:
            matrix: SNP matrix (individuals x loci) as string IUPAC codes.
            indices: Selected individual indices.
            iupac_decoder: Mapping like {'R': ('A','G'), ...}.

        Returns:
            Dict: locus_index -> {allele: frequency}.
        """
        n_loci = matrix.shape[1]
        allele_freqs: dict[int, dict[str, float]] = {}

        for locus in range(n_loci):
            genos = [matrix[i, locus] for i in indices]
            # Reuse class-level missing filter (doesn't capture self)
            clean = GeneticDistance._clean_inds(genos)
            if not clean:
                allele_freqs[locus] = {}
                continue

            alleles = GeneticDistance._get_alleles(iupac_decoder, clean)
            total = len(alleles)
            if total == 0:
                allele_freqs[locus] = {}
                continue
            # Use numpy for speed; small dict at the end is fine
            vals, counts = np.unique(np.asarray(alleles), return_counts=True)
            freqs = {a: c / total for a, c in zip(vals.tolist(), counts.tolist())}
            allele_freqs[locus] = freqs

        return allele_freqs

    @staticmethod
    def _neis_distance(
        f1: dict[int, dict[str, float]], f2: dict[int, dict[str, float]]
    ) -> float:
        """Calculate Nei's genetic distance using multilocus genetic identity.

        Args:
            f1: Locus-indexed allele frequencies for population 1.
            f2: Locus-indexed allele frequencies for population 2.

        Returns:
            Nei's genetic distance, calculated as -log(I), where I is Nei's
            normalized genetic identity.

        Notes:
            Loci where both populations have valid allele frequencies are included.
            Loci with no shared alleles contribute zero to the numerator but still
            contribute to the denominator. This is essential; skipping no-shared-
            allele loci artificially deflates genetic distance.
        """
        total_num = 0.0
        total_denom1 = 0.0
        total_denom2 = 0.0
        n_valid_loci = 0

        for locus in sorted(set(f1) & set(f2)):
            p1 = f1.get(locus, {})
            p2 = f2.get(locus, {})

            if not p1 or not p2:
                continue

            alleles = set(p1) | set(p2)

            locus_num = sum(
                p1.get(allele, 0.0) * p2.get(allele, 0.0) for allele in alleles
            )
            locus_denom1 = sum(p1.get(allele, 0.0) ** 2 for allele in alleles)
            locus_denom2 = sum(p2.get(allele, 0.0) ** 2 for allele in alleles)

            if locus_denom1 <= 0.0 or locus_denom2 <= 0.0:
                continue

            total_num += locus_num
            total_denom1 += locus_denom1
            total_denom2 += locus_denom2
            n_valid_loci += 1

        if n_valid_loci == 0 or total_denom1 <= 0.0 or total_denom2 <= 0.0:
            return np.nan

        identity = total_num / np.sqrt(total_denom1 * total_denom2)

        if not np.isfinite(identity) or identity <= 0.0:
            return np.inf

        identity = min(identity, 1.0)

        return float(-np.log(identity))

    @staticmethod
    def _nei_permutation_pvalue(
        pop1_inds: np.ndarray,
        pop2_inds: np.ndarray,
        full_matrix: np.ndarray,
        iupac_decoder: dict[str, tuple[str, ...]],
        n_permutations: int,
        seed: int,
    ) -> tuple[float, float, np.ndarray]:
        """Calculates a p-value for Nei distance using a permutation test (pure/static)."""
        f1_obs = GeneticDistance._calculate_allele_freqs(
            full_matrix, list(pop1_inds), iupac_decoder
        )
        f2_obs = GeneticDistance._calculate_allele_freqs(
            full_matrix, list(pop2_inds), iupac_decoder
        )
        obs_nei = GeneticDistance._neis_distance(f1_obs, f2_obs)

        pooled_indices = np.concatenate((pop1_inds, pop2_inds))
        n1_size = len(pop1_inds)
        perm_distances = np.full(n_permutations, np.nan)
        rng = np.random.default_rng(seed)

        for i in range(n_permutations):
            rng.shuffle(pooled_indices)
            perm_inds1 = pooled_indices[:n1_size]
            perm_inds2 = pooled_indices[n1_size:]
            f1_perm = GeneticDistance._calculate_allele_freqs(
                full_matrix, list(perm_inds1), iupac_decoder
            )
            f2_perm = GeneticDistance._calculate_allele_freqs(
                full_matrix, list(perm_inds2), iupac_decoder
            )
            perm_distances[i] = GeneticDistance._neis_distance(f1_perm, f2_perm)

        perm_distances = perm_distances[~np.isnan(perm_distances)]
        p_value = (
            (np.sum(perm_distances >= obs_nei) + 1) / (len(perm_distances) + 1)
            if len(perm_distances) > 0
            else np.nan
        )

        if not isinstance(p_value, float) and ~np.isnan(p_value):
            try:
                pval = float(p_value)
            except Exception:
                pval = 1.0
        elif np.isnan(p_value):
            pval = 1.0
        elif isinstance(p_value, float):
            pval = p_value
        else:
            try:
                pval = float(p_value)
            except Exception:
                pval = 1.0

        return obs_nei, pval, perm_distances

    @staticmethod
    def _permutation_worker(
        pop_pair_keys: tuple[str, str],
        pop_indices: dict[str, np.ndarray],
        full_matrix: np.ndarray,
        iupac_decoder: dict[str, tuple[str, ...]],
        n_reps: int,
        seed: int,
    ):
        """Top-level/staticmethod worker: runs Nei permutation for one pair with its own seed."""
        p1_key, p2_key = pop_pair_keys
        i1 = pop_indices[p1_key]
        i2 = pop_indices[p2_key]

        obs, pval, dist = GeneticDistance._nei_permutation_pvalue(
            i1, i2, full_matrix, iupac_decoder, n_permutations=n_reps, seed=seed
        )
        return (p1_key, p2_key), {"nei": obs, "pvalue": pval, "perm_dist": dist}

    @staticmethod
    def _bootstrap_replicate(
        seed: int,
        n_loci: int,
        full_matrix: np.ndarray,
        iupac_decoder: dict[str, tuple[str, ...]],
        pop_pairs: list[tuple[str, str]],
        pop_indices: dict[str, np.ndarray],
    ) -> dict[tuple[str, str], float]:
        """Top-level/staticmethod worker: one bootstrap replicate over loci (pure/static)."""
        rng = np.random.default_rng(seed)
        resampled_loci = rng.choice(n_loci, size=n_loci, replace=True)
        resampled_matrix = full_matrix[:, resampled_loci]

        replicate: dict[tuple[str, str], float] = {}
        for p1_key, p2_key in pop_pairs:
            i1 = pop_indices[p1_key]
            i2 = pop_indices[p2_key]
            f1 = GeneticDistance._calculate_allele_freqs(
                resampled_matrix, list(i1), iupac_decoder
            )
            f2 = GeneticDistance._calculate_allele_freqs(
                resampled_matrix, list(i2), iupac_decoder
            )
            replicate[(p1_key, p2_key)] = GeneticDistance._neis_distance(f1, f2)

        return replicate

    def nei_distance(
        self,
        method: Literal["observed", "permutation", "bootstrap"] = "observed",
        n_reps: int = 1000,
        n_jobs: int = 1,
    ):
        """Calculate pairwise Nei's genetic distance with optional statistical tests.

        Args:
            method (Literal["observed", "permutation", "bootstrap"]): The method to use for calculating Nei's distance.
            n_reps (int): The number of replicates to use for permutation and bootstrap methods.
            n_jobs (int): The number of parallel jobs to use. -1 uses all available cores.

        Returns:
            dict: A dictionary containing the results of the Nei distance calculation.
            If method is "observed":
                pd.DataFrame: A symmetric matrix of Nei distances between populations.

            If method is "permutation":
                dict: A dictionary where keys are (pop1, pop2) tuples and values are dicts with:
                    - "nei": observed Nei distance (float)
                    - "pvalue": p-value from permutation test (float)
                    - "perm_dist": array of Nei distances from permutations (np.ndarray)

            If method is "bootstrap":
                dict: A dictionary where keys are (pop1, pop2) tuples and values are dicts with:
                    - "nei": observed Nei distance (float)
                    - "boot_dist": array of Nei distances from bootstrap replicates (np.ndarray)
        """

        self.logger.info(
            f"Calculating Nei's distance using method='{method}' with {n_reps} replicates and {n_jobs} jobs."
        )

        if n_jobs == 0 or n_jobs < -1:
            msg = f"n_jobs must be -1 or a positive integer, but got: {n_jobs}"
            self.logger.error(msg)
            raise ValueError(msg)

        rng = np.random.default_rng(self.seed)

        popmap = self.genotype_data.popmap_inverse

        if popmap is None:
            msg = "Population map is required for Nei distance calculation but was not found."
            self.logger.error(msg)
            raise AttributeError(msg)

        sample_to_idx = {s: i for i, s in enumerate(self.genotype_data.samples)}
        pop_indices = {
            pop: np.array([sample_to_idx[s] for s in popmap[pop]]) for pop in popmap
        }

        pop_keys = sorted(popmap.keys())
        full_matrix = self.genotype_data.snp_data.astype(str)
        n_loci = full_matrix.shape[1]
        num_pops = len(pop_keys)
        iupac_decoder = self.genotype_data.reverse_iupac_mapping

        if method == "observed":
            nei_mat = np.full((num_pops, num_pops), np.nan)
            for ia, ib in itertools.combinations(range(num_pops), 2):
                p1_key, p2_key = pop_keys[ia], pop_keys[ib]
                i1, i2 = pop_indices[p1_key], pop_indices[p2_key]
                f1 = GeneticDistance._calculate_allele_freqs(
                    full_matrix, list(i1), iupac_decoder
                )
                f2 = GeneticDistance._calculate_allele_freqs(
                    full_matrix, list(i2), iupac_decoder
                )
                dist = GeneticDistance._neis_distance(f1, f2)
                nei_mat[ia, ib] = nei_mat[ib, ia] = dist

            np.fill_diagonal(nei_mat, 0.0)
            df = pd.DataFrame(nei_mat, index=pop_keys, columns=pop_keys)
            self.logger.info("Nei distance calculation complete!")
            return df

        elif method == "permutation":
            perm_result: dict[tuple[str, str], dict[str, object]] = {}
            pop_pairs = list(itertools.combinations(pop_keys, 2))

            # Per-pair independent seeds.
            # If self.seed is None, SeedSequence uses nondeterministic entropy.
            ss = np.random.SeedSequence(self.seed)
            children = ss.spawn(len(pop_pairs))

            pair_seeds = {
                pair: int(child.generate_state(1, dtype=np.uint32)[0]) & 0x7FFFFFFF
                for pair, child in zip(pop_pairs, children)
            }

            from concurrent.futures import as_completed

            max_workers = mp.cpu_count() if n_jobs == -1 else n_jobs

            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        GeneticDistance._permutation_worker,
                        pair,
                        pop_indices,
                        full_matrix,
                        iupac_decoder,
                        n_reps,
                        pair_seeds[pair],
                    )
                    for pair in pop_pairs
                ]

                for fut in tqdm(
                    as_completed(futures),
                    desc=f"Nei population pairs ({n_reps} permutations each)",
                    total=len(futures),
                ):
                    pop_pair, res_dict = fut.result()
                    perm_result[pop_pair] = res_dict

            for (p1_key, p2_key), res_dict in perm_result.items():
                obs_value = res_dict["nei"]

                if isinstance(obs_value, (int, float, str)):
                    try:
                        obs = float(obs_value)
                    except (ValueError, TypeError):
                        obs = np.nan
                elif isinstance(obs_value, np.generic):
                    try:
                        obs = float(obs_value.item())
                    except (ValueError, TypeError):
                        obs = np.nan
                else:
                    obs = np.nan

                if not isinstance(res_dict["perm_dist"], np.ndarray):
                    try:
                        res_dict["perm_dist"] = np.array(
                            res_dict["perm_dist"], dtype=float
                        )
                    except (ValueError, TypeError):
                        res_dict["perm_dist"] = np.array([], dtype=float)

                self.plotter.plot_permutation_dist(
                    obs,
                    res_dict["perm_dist"],
                    p1_key,
                    p2_key,
                    dist_type="nei",
                )

            self.logger.info("Nei distance permutation test complete!")
            return perm_result

        elif method == "bootstrap":
            pop_pairs = list(itertools.combinations(pop_keys, 2))

            bootstrap_result: dict[tuple[str | int, str | int], dict[str, object]] = {}

            for p1_key, p2_key in pop_pairs:
                i1 = pop_indices[p1_key]
                i2 = pop_indices[p2_key]

                f1_obs = GeneticDistance._calculate_allele_freqs(
                    full_matrix,
                    list(i1),
                    iupac_decoder,
                )
                f2_obs = GeneticDistance._calculate_allele_freqs(
                    full_matrix,
                    list(i2),
                    iupac_decoder,
                )

                obs_nei = GeneticDistance._neis_distance(f1_obs, f2_obs)

                bootstrap_result[(p1_key, p2_key)] = {
                    "nei": obs_nei,
                    "boot_dist": np.zeros(n_reps, dtype=float),
                }

            worker_func = partial(
                GeneticDistance._bootstrap_replicate,
                n_loci=n_loci,
                full_matrix=full_matrix,
                iupac_decoder=iupac_decoder,
                pop_pairs=pop_pairs,
                pop_indices=pop_indices,
            )

            seeds = rng.integers(0, 1_000_000_000, size=n_reps)

            with ProcessPoolExecutor(
                max_workers=(mp.cpu_count() if n_jobs == -1 else n_jobs)
            ) as pool:
                reps = pool.map(worker_func, seeds)

                for i, rep_data in tqdm(
                    enumerate(reps),
                    desc="Nei bootstrapping",
                    total=n_reps,
                ):
                    for pair, dist_val in rep_data.items():
                        boot_dist = bootstrap_result[pair]["boot_dist"]

                        if isinstance(boot_dist, np.ndarray):
                            boot_dist[i] = dist_val

            self.logger.info("Nei distance bootstrap complete!")
            return bootstrap_result

        else:
            msg = f"Unknown method '{method}'. Choose from 'observed', 'permutation', or 'bootstrap'."
            self.logger.error(msg)
            raise ValueError(msg)

    def parse_nei_result(
        self,
        result_dict: dict | pd.DataFrame,
        alpha: float = 0.05,
    ) -> tuple[
        pd.DataFrame | None,
        pd.DataFrame | None,
        pd.DataFrame | None,
        pd.DataFrame | None,
    ]:
        """Convert the output of `nei_distance()` into pairwise result DataFrames.

        Args:
            result_dict (dict | pd.DataFrame): Structure returned by `nei_distance()`.
            alpha (float): Significance level for interval bounds. Default 0.05 gives 95%
                intervals.

        Returns:
            tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]: ``(df_observed, df_lower, df_upper, df_pval)``.

            For ``method="observed"``:
                - df_observed: Observed Nei's distance matrix.
                - df_lower: None.
                - df_upper: None.
                - df_pval: None.

            For ``method="bootstrap"`` with the preferred structure
            ``{(pop1, pop2): {"nei": observed, "boot_dist": array}}``:
                - df_observed: Observed Nei's distance matrix.
                - df_lower: Lower bootstrap confidence bound.
                - df_upper: Upper bootstrap confidence bound.
                - df_pval: None.

            For ``method="permutation"``:
                - df_observed: Observed Nei's distance matrix.
                - df_lower: Lower permutation/null interval, if ``perm_dist`` exists.
                - df_upper: Upper permutation/null interval, if ``perm_dist`` exists.
                - df_pval: Empirical permutation p-value matrix.

        Raises:
            ValueError: If the result structure is empty or unrecognized.
        """
        if isinstance(result_dict, pd.DataFrame):
            return result_dict, None, None, None

        if not isinstance(result_dict, dict) or not result_dict:
            msg = f"Nei distance result must be a non-empty dictionary or a pandas DataFrame. Got: {type(result_dict)}"
            self.logger.error(msg)
            raise ValueError(msg)

        first_val = next(iter(result_dict.values()))

        def all_populations_from_keys(dict_keys):
            """Return sorted unique population labels from tuple pairwise keys."""
            pop_set = set()
            for key in dict_keys:
                pop_set.update(key)
            return sorted(pop_set)

        def init_pairwise_frame(pops: list[str]) -> pd.DataFrame:
            """Create an empty square pairwise DataFrame."""
            return pd.DataFrame(np.nan, index=pops, columns=pops, dtype=float)

        def fill_symmetric(
            df: pd.DataFrame,
            pop1: str,
            pop2: str,
            value: float | int | np.number | None,
        ) -> None:
            """Fill symmetric pairwise cells."""
            if value is None:
                return

            try:
                val = float(value)
            except (TypeError, ValueError):
                val = np.nan

            df.loc[pop1, pop2] = val
            df.loc[pop2, pop1] = val

        def percentile_interval(values: np.ndarray) -> tuple[float, float]:
            """Return lower and upper percentile interval bounds."""
            values = np.asarray(values, dtype=float)
            values = values[~np.isnan(values)]

            if values.size == 0:
                return np.nan, np.nan

            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))

            return float(lower), float(upper)

        pop_pairs = list(result_dict.keys())
        pops = all_populations_from_keys(pop_pairs)

        # ------------------------------------------------------------------
        # Preferred bootstrap result:
        # {(pop1, pop2): {"nei": observed, "boot_dist": np.array([...])}}
        # ------------------------------------------------------------------
        if (
            isinstance(first_val, dict)
            and "nei" in first_val
            and "boot_dist" in first_val
        ):
            df_obs = init_pairwise_frame(pops)
            df_lower = init_pairwise_frame(pops)
            df_upper = init_pairwise_frame(pops)

            for (pop1, pop2), subdict in result_dict.items():
                obs_val = subdict.get("nei")
                boot_dist = np.asarray(subdict.get("boot_dist", []), dtype=float)

                lower_val, upper_val = percentile_interval(boot_dist)

                fill_symmetric(df_obs, pop1, pop2, obs_val)
                fill_symmetric(df_lower, pop1, pop2, lower_val)
                fill_symmetric(df_upper, pop1, pop2, upper_val)

            np.fill_diagonal(df_obs.values, 0.0)
            np.fill_diagonal(df_lower.values, 0.0)
            np.fill_diagonal(df_upper.values, 0.0)

            return df_obs, df_lower, df_upper, None

        # ------------------------------------------------------------------
        # Legacy bootstrap result:
        # {(pop1, pop2): np.array([...])}
        #
        # This structure does not contain the observed Nei estimate. The only
        # possible summary is the bootstrap mean, which is not ideal for the
        # manuscript table. Keep as fallback, but warn loudly.
        # ------------------------------------------------------------------
        if isinstance(first_val, np.ndarray):
            self.logger.warning(
                "Parsing legacy Nei bootstrap output containing only replicate arrays. "
                "Observed Nei distances are unavailable, so df_observed will contain "
                "bootstrap means. Prefer returning {'nei': observed, 'boot_dist': array} "
                "from nei_distance(method='bootstrap')."
            )

            df_mean = init_pairwise_frame(pops)
            df_lower = init_pairwise_frame(pops)
            df_upper = init_pairwise_frame(pops)

            for (pop1, pop2), arr in result_dict.items():
                arr = np.asarray(arr, dtype=float)
                arr_nonan = arr[~np.isnan(arr)]

                if arr_nonan.size == 0:
                    mean_val = np.nan
                    lower_val = np.nan
                    upper_val = np.nan
                else:
                    mean_val = float(np.mean(arr_nonan))
                    lower_val, upper_val = percentile_interval(arr_nonan)

                fill_symmetric(df_mean, pop1, pop2, mean_val)
                fill_symmetric(df_lower, pop1, pop2, lower_val)
                fill_symmetric(df_upper, pop1, pop2, upper_val)

            np.fill_diagonal(df_mean.values, 0.0)
            np.fill_diagonal(df_lower.values, 0.0)
            np.fill_diagonal(df_upper.values, 0.0)

            return df_mean, df_lower, df_upper, None

        # ------------------------------------------------------------------
        # Permutation result:
        # {(pop1, pop2): {"nei": observed, "pvalue": p, "perm_dist": array}}
        # ------------------------------------------------------------------
        if isinstance(first_val, dict) and "nei" in first_val and "pvalue" in first_val:
            df_obs = init_pairwise_frame(pops)
            df_pval = init_pairwise_frame(pops)
            df_lower = init_pairwise_frame(pops)
            df_upper = init_pairwise_frame(pops)

            has_perm_dist = False

            for (pop1, pop2), subdict in result_dict.items():
                obs_val = subdict.get("nei")
                p_value = subdict.get("pvalue")
                perm_dist = subdict.get("perm_dist", None)

                fill_symmetric(df_obs, pop1, pop2, obs_val)
                fill_symmetric(df_pval, pop1, pop2, p_value)

                if perm_dist is not None:
                    perm_dist = np.asarray(perm_dist, dtype=float)
                    perm_dist = perm_dist[~np.isnan(perm_dist)]

                    if perm_dist.size > 0:
                        lower_val, upper_val = percentile_interval(perm_dist)
                        fill_symmetric(df_lower, pop1, pop2, lower_val)
                        fill_symmetric(df_upper, pop1, pop2, upper_val)
                        has_perm_dist = True

            np.fill_diagonal(df_obs.values, 0.0)
            np.fill_diagonal(df_pval.values, 1.0)

            if has_perm_dist:
                np.fill_diagonal(df_lower.values, 0.0)
                np.fill_diagonal(df_upper.values, 0.0)
            else:
                df_lower = None
                df_upper = None

            self.logger.info(f"Nei distance files saved to {self.outdir}")

            return df_obs, df_lower, df_upper, df_pval

        msg = "Unrecognized Nei result structure. Expected one of: DataFrame; {(pop1, pop2): {'nei': float, 'boot_dist': array}}; {(pop1, pop2): {'nei': float, 'pvalue': float, 'perm_dist': array}}; or legacy {(pop1, pop2): np.ndarray}."
        self.logger.error(msg)
        raise ValueError(msg)
