import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from snpio.utils.logging import LoggerManager
from snpio.utils.misc import IUPAC
from snpio.utils.multiqc_reporter import SNPioMultiQC

if TYPE_CHECKING:
    from snpio.plotting.plotting import Plotting
    from snpio.read_input.genotype_data import GenotypeData

phased_encoding = IUPAC().get_phased_encoding()


class FstDistance:
    """Class for calculating Fst distance between populations using Weir and Cockerham's method.

    This class implements Weir and Cockerham's (1984) Fst calculation for two populations, s1 and s2. It supports:

    - Observed Fst calculation
    - Permutation tests for statistical significance
    - Bootstrap resampling for confidence intervals

    References:
        ..[1] Weir, B. S., & Cockerham, C. C. (1984). Estimating F-statistics for the analysis of population structure. Evolution, 38(6), 1358-1370.
    """

    def __init__(
        self,
        genotype_data: "GenotypeData",
        plotter: "Plotting",
        seed: int | None = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.genotype_data = genotype_data
        self.plotter = plotter
        self.verbose = verbose
        self.seed = seed

        if self.genotype_data.was_filtered:
            self.outdir = Path(
                f"{genotype_data.prefix}_output", "nremover", "analysis", "fst"
            )
        else:
            self.outdir = Path(f"{genotype_data.prefix}_output", "analysis", "fst")
        self.outdir.mkdir(parents=True, exist_ok=True)

        logman = LoggerManager(
            __name__, self.genotype_data.prefix, debug=debug, verbose=verbose
        )
        self.logger = logman.get_logger()
        self.iupac = IUPAC(logger=self.logger)
        self.snpio_mqc = SNPioMultiQC

    @staticmethod
    def _get_alleles(phased_list: list[str]) -> list[str]:
        """Flattens a list of 'A/T' strings into a list of single alleles."""
        return [allele for genotype in phased_list for allele in genotype.split("/")]

    @staticmethod
    def _get_het_from_phased(allele: str, phased_list: list[str], count: bool = False):
        """Calculates observed heterozygosity for an allele from phased genotypes."""
        hets = 0.0
        for genotype in phased_list:
            a1, a2 = genotype.split("/")
            if (a1 == allele and a2 != allele) or (a2 == allele and a1 != allele):
                hets += 1.0

        if count:
            return hets
        else:
            # Correctly divide by number of individuals (N), not 2N
            return hets / len(phased_list) if phased_list else 0.0

    @staticmethod
    def _decode_and_clean_phased_genotypes(genotypes) -> list[str]:
        """Decode IUPAC or phased diploid genotypes and remove missing calls.

        Args:
            genotypes: Iterable of genotype strings. Values may be IUPAC codes,
                already-phased diploid strings, VCF-style diploid strings, or
                missing values.

        Returns:
            list[str]: Clean phased diploid genotypes using "/" as the separator.
        """
        missing_alleles = {
            "",
            ".",
            "-",
            "?",
            "N",
            "NA",
            "NAN",
            "NONE",
            "NULL",
            "MISSING",
        }

        iupac_diploid = {
            "A": "A/A",
            "C": "C/C",
            "G": "G/G",
            "T": "T/T",
            "R": "A/G",
            "Y": "C/T",
            "S": "G/C",
            "W": "A/T",
            "K": "G/T",
            "M": "A/C",
        }

        clean: list[str] = []

        for gt in genotypes:
            if gt is None:
                continue

            gt_str = str(gt).strip().upper()

            if gt_str in missing_alleles:
                continue

            phased = iupac_diploid.get(gt_str)

            if phased is None:
                phased = phased_encoding.get(
                    gt_str,
                    phased_encoding.get(gt_str.upper(), gt_str),
                )

            phased = str(phased).strip().upper().replace("|", "/")

            if "/" not in phased:
                continue

            alleles = [allele.strip().upper() for allele in phased.split("/")]

            if len(alleles) != 2:
                continue

            a1, a2 = alleles

            if a1 in missing_alleles or a2 in missing_alleles:
                continue

            clean.append(f"{a1}/{a2}")

        return clean

    @staticmethod
    def _clean_inds(inds: list[str]) -> list[str]:
        """Remove genotypes containing missing alleles.

        Args:
            inds (list[str]): Genotype strings.

        Returns:
            list[str]: Genotype strings without missing alleles.
        """
        missing_codes = {
            "",
            ".",
            "-",
            "?",
            "N",
            "NA",
            "NAN",
            "NONE",
            "NULL",
            "MISSING",
        }

        clean: list[str] = []

        for ind in inds:
            if ind is None:
                continue

            gt = str(ind).strip().replace("|", "/")

            if gt.upper() in missing_codes:
                continue

            if "/" in gt:
                alleles = [allele.strip().upper() for allele in gt.split("/")]

                if len(alleles) != 2:
                    continue

                if any(allele in missing_codes for allele in alleles):
                    continue

                clean.append(f"{alleles[0]}/{alleles[1]}")
            else:
                clean.append(gt.upper())

        return clean

    @staticmethod
    def _permutation_worker(
        pop_pair_keys: tuple[str, str],
        pop_indices: dict[str, np.ndarray],
        full_matrix: np.ndarray,
        n_reps: int,
        seed: int,
    ):
        """Worker function for a single Fst permutation test between two populations.

        Args:
            pop_pair_keys (tuple[str, str]): Tuple of population keys (pop1, pop2) to compare.
            pop_indices (dict[str, np.ndarray]): Dictionary mapping population keys to individual indices in the genotype matrix.
            full_matrix (np.ndarray): Full genotype matrix with shape (num_individuals, num_loci).
            n_reps (int): Number of permutations to perform for p-value estimation.
            seed (int): Random seed for reproducibility in permutations.

        Returns:
            tuple: ((pop1, pop2), {"fst": float, "pvalue": float, "perm_dist": np.ndarray})
        """
        p1_key, p2_key = pop_pair_keys
        i1 = pop_indices[p1_key]
        i2 = pop_indices[p2_key]

        obs, pval, dist = FstDistance._fst_permutation_pvalue(
            i1, i2, full_matrix, n_permutations=n_reps, seed=seed
        )

        # Return the key with the result to reassemble the final dictionary
        return (p1_key, p2_key), {"fst": obs, "pvalue": pval, "perm_dist": dist}

    @staticmethod
    def _fst_permutation_pvalue(
        pop1_inds: np.ndarray,
        pop2_inds: np.ndarray,
        full_matrix: np.ndarray,
        n_permutations: int,
        seed: int,
    ):
        """Calculates a p-value for Fst using a permutation test.

        Args:
            pop1_inds (np.ndarray): Indices of individuals in population
            pop2_inds (np.ndarray): Indices of individuals in population 2
            full_matrix (np.ndarray): Full genotype matrix
            n_permutations (int): Number of permutations to perform
            seed (int): Random seed for reproducibility

        Returns:
            Tuple[float, float, np.ndarray]: (observed Fst, p-value, permutation distribution array)
        """
        obs_fst = FstDistance._compute_multilocus_fst(pop1_inds, pop2_inds, full_matrix)

        pooled_indices = np.concatenate((pop1_inds, pop2_inds))
        n1_size = len(pop1_inds)
        perm_dist = np.full(n_permutations, np.nan)
        rng = np.random.default_rng(seed)

        for i in range(n_permutations):
            rng.shuffle(pooled_indices)
            perm_inds1 = pooled_indices[:n1_size]
            perm_inds2 = pooled_indices[n1_size:]
            perm_dist[i] = FstDistance._compute_multilocus_fst(
                perm_inds1, perm_inds2, full_matrix
            )

        perm_dist = perm_dist[~np.isnan(perm_dist)]
        p_value = (
            (np.sum(perm_dist >= obs_fst) + 1) / (len(perm_dist) + 1)
            if len(perm_dist) > 0
            else np.nan
        )
        return obs_fst, p_value, perm_dist

    @staticmethod
    def audit_iupac_decoding(
        full_matrix: np.ndarray,
        max_loci: int | None = 1000,
    ) -> dict[str, object]:
        """Audit IUPAC genotype decoding for SNPio Fst calculations.

        Args:
            full_matrix (np.ndarray): Genotype matrix with individuals as rows and loci as columns.
            max_loci (int | None): Maximum number of loci to inspect.

        Returns:
            dict[str, object]: Summary of raw genotype tokens and decoding behavior.
        """
        n_loci = full_matrix.shape[1]
        loci_to_check = n_loci if max_loci is None else min(max_loci, n_loci)

        raw_counts: dict[str, int] = {}
        decoded_counts: dict[str, int] = {}
        skipped_counts: dict[str, int] = {}

        for loc in range(loci_to_check):
            for raw_gt in full_matrix[:, loc]:
                raw = str(raw_gt).strip().upper() if raw_gt is not None else "NONE"
                raw_counts[raw] = raw_counts.get(raw, 0) + 1

                decoded = FstDistance._decode_and_clean_phased_genotypes([raw_gt])

                if decoded:
                    key = decoded[0]
                    decoded_counts[key] = decoded_counts.get(key, 0) + 1
                else:
                    skipped_counts[raw] = skipped_counts.get(raw, 0) + 1

        return {
            "n_loci_checked": loci_to_check,
            "raw_counts": raw_counts,
            "decoded_counts": decoded_counts,
            "skipped_counts": skipped_counts,
            "warning": (
                "Check skipped_counts. N should dominate skipped genotypes. "
                "A/C/G/T/R/Y/S/W/K/M should not appear in skipped_counts."
            ),
        }

    @staticmethod
    def _two_pop_weir_cockerham_fst_locus(
        s1: list[str],
        s2: list[str],
    ) -> tuple[float, float, float]:
        """Compute HierFstat-style Weir-Cockerham per-locus components.

        This directly aligns with `hierfstat::wc()` by correctly handling degrees
        of freedom per allele, enforcing exact Mean Square denominators, and
        accounting for monomorphic limits.

        Args:
            s1 (list[str]): Clean phased diploid genotypes for population 1.
            s2 (list[str]): Clean phased diploid genotypes for population 2.

        Returns:
            tuple[float, float, float]: Per-locus ``siga``, ``sigb``, and ``sigw``.
        """
        n1 = float(len(s1))
        n2 = float(len(s2))

        # If either population is missing at this locus, variance cannot be calculated
        if n1 == 0.0 or n2 == 0.0:
            return np.nan, np.nan, np.nan

        n_t = n1 + n2
        r = 2.0  # strictly pairwise

        # Hierfstat requires N_T > r for MSI degrees of freedom (N - r > 0)
        if n_t <= r:
            return np.nan, np.nan, np.nan

        # nc formula for r=2 populations
        n_c = (n_t - ((n1**2 + n2**2) / n_t)) / (r - 1.0)

        if not np.isfinite(n_c) or n_c <= 0.0:
            return np.nan, np.nan, np.nan

        def summarize_population(
            phased_genotypes: list[str],
        ) -> tuple[dict[str, int], dict[str, int]]:
            allele_counts: dict[str, int] = {}
            het_counts: dict[str, int] = {}

            for genotype in phased_genotypes:
                a1, a2 = genotype.split("/")
                allele_counts[a1] = allele_counts.get(a1, 0) + 1
                allele_counts[a2] = allele_counts.get(a2, 0) + 1

                if a1 != a2:
                    het_counts[a1] = het_counts.get(a1, 0) + 1
                    het_counts[a2] = het_counts.get(a2, 0) + 1

            return allele_counts, het_counts

        allele_counts1, het_counts1 = summarize_population(s1)
        allele_counts2, het_counts2 = summarize_population(s2)

        unique_alleles = set(allele_counts1) | set(allele_counts2)

        # Monomorphic loci contribute 0 variance globally; returning 0s explicitly
        # prevents floating point/NaN injection into the multi-locus sums.
        if len(unique_alleles) < 2:
            return 0.0, 0.0, 0.0

        siga_total = 0.0
        sigb_total = 0.0
        sigw_total = 0.0

        for allele in unique_alleles:
            ac1 = float(allele_counts1.get(allele, 0))
            ac2 = float(allele_counts2.get(allele, 0))

            p1 = ac1 / (2.0 * n1)
            p2 = ac2 / (2.0 * n2)

            # Global allele frequency weighted by sample size (matches hierfstat `pb`)
            p_bar = (ac1 + ac2) / (2.0 * n_t)

            h1 = float(het_counts1.get(allele, 0))
            h2 = float(het_counts2.get(allele, 0))

            mhom1 = (ac1 - h1) / 2.0
            mhom2 = (ac2 - h2) / 2.0

            # HierFstat Sums of Squares (SSG, SSi, SSP)
            ssg = ((n1 * p1) - mhom1) + ((n2 * p2) - mhom2)
            ssi = n1 * (p1 - 2.0 * p1**2) + mhom1 + n2 * (p2 - 2.0 * p2**2) + mhom2
            ssp = 2.0 * (n1 * (p1 - p_bar) ** 2 + n2 * (p2 - p_bar) ** 2)

            # HierFstat Mean Squares (MSG, MSP, MSI)
            msg = ssg / n_t
            msp = ssp / (r - 1.0)
            msi = ssi / (n_t - r)

            # Variance Components
            sigw = msg
            sigb = 0.5 * (msi - msg)
            siga = (msp - msi) / (2.0 * n_c)

            if np.isfinite(siga):
                siga_total += siga
            if np.isfinite(sigb):
                sigb_total += sigb
            if np.isfinite(sigw):
                sigw_total += sigw

        return siga_total, sigb_total, sigw_total

    @staticmethod
    def _compute_multilocus_fst(
        pop1_inds: np.ndarray,
        pop2_inds: np.ndarray,
        full_matrix: np.ndarray,
    ) -> float:
        """Compute multilocus Weir-Cockerham Fst from summed components."""
        a_vals, b_vals, c_vals = FstDistance._fst_variance_components_per_locus(
            pop1_inds,
            pop2_inds,
            full_matrix,
        )

        # Align with R's na.rm=TRUE logic
        valid_loci = np.isfinite(a_vals) & np.isfinite(b_vals) & np.isfinite(c_vals)

        if not np.any(valid_loci):
            return np.nan

        total_a = float(np.sum(a_vals[valid_loci]))
        total_b = float(np.sum(b_vals[valid_loci]))
        total_c = float(np.sum(c_vals[valid_loci]))

        total_denom = total_a + total_b + total_c

        # Handle identical populations natively avoiding NaN assignment limits
        if not np.isfinite(total_denom) or total_denom == 0.0:
            return 0.0

        return total_a / total_denom

    @staticmethod
    def _bootstrap_replicate(
        seed: int,
        pair_components: dict[
            tuple[str, str],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ],
        pop_pairs: list[tuple[str, str]],
        n_loci: int,
    ) -> dict[tuple[str, str], float]:
        """Worker function for a single locus-bootstrap replicate."""
        rng = np.random.default_rng(seed)
        resampled_loci = rng.choice(n_loci, size=n_loci, replace=True)

        replicate: dict[tuple[str, str], float] = {}

        for pair in pop_pairs:
            a_vals, b_vals, c_vals = pair_components[pair]

            a_resampled = a_vals[resampled_loci]
            b_resampled = b_vals[resampled_loci]
            c_resampled = c_vals[resampled_loci]

            valid_loci = (
                np.isfinite(a_resampled)
                & np.isfinite(b_resampled)
                & np.isfinite(c_resampled)
            )

            if not np.any(valid_loci):
                replicate[pair] = np.nan
                continue

            total_a = float(np.sum(a_resampled[valid_loci]))
            total_b = float(np.sum(b_resampled[valid_loci]))
            total_c = float(np.sum(c_resampled[valid_loci]))

            total_denom = total_a + total_b + total_c

            if np.isfinite(total_denom) and total_denom != 0.0:
                replicate[pair] = total_a / total_denom
            else:
                replicate[pair] = 0.0

        return replicate

    def weir_cockerham_fst(
        self,
        method: Literal["observed", "permutation", "bootstrap"] = "observed",
        n_reps: int = 1000,
        n_jobs: int = 1,
    ):
        """Calculates pairwise Fst with optional statistical tests.

        Args:
            method (Literal["observed", "permutation", "bootstrap"]): The method to use for Fst calculation.
            n_reps (int): The number of replicates to use for permutation/bootstrap methods.
            n_jobs (int): The number of parallel jobs to run.

        Returns:
            pd.DataFrame: A DataFrame containing pairwise Fst values.
        """
        self.logger.info(
            f"Calculating Fst using method='{method}' with {n_reps} replicates."
        )

        rng = np.random.default_rng(self.seed)

        if n_jobs == 0 or n_jobs < -1:
            msg = f"n_jobs must be -1 or a positive integer, but got: {n_jobs}"
            self.logger.error(msg)
            raise ValueError(msg)

        popmap = self.genotype_data.popmap_inverse

        if popmap is None:
            msg = "Population map is not available in genotype data, cannot calculate Fst."
            self.logger.error(msg)
            raise TypeError(msg)

        sample_to_idx = {s: i for i, s in enumerate(self.genotype_data.samples)}
        pop_indices = {
            pop: np.array([sample_to_idx[s] for s in samples if s in sample_to_idx])
            for pop, samples in popmap.items()
        }

        pop_keys = sorted(pop_indices.keys())

        full_matrix = self.genotype_data.snp_data
        n_loci = full_matrix.shape[1]
        num_pops = len(pop_keys)

        iupac_audit = FstDistance.audit_iupac_decoding(full_matrix, max_loci=1000)

        self.logger.info(
            "IUPAC decoding audit: "
            f"n_loci_checked={iupac_audit['n_loci_checked']}, "
            f"raw_counts={iupac_audit['raw_counts']}, "
            f"decoded_counts={iupac_audit['decoded_counts']}, "
            f"skipped_counts={iupac_audit['skipped_counts']}, "
            f"warning={iupac_audit['warning']}"
        )

        n_loci = full_matrix.shape[1]
        num_pops = len(pop_keys)

        if method == "observed":
            fst_mat = np.full((num_pops, num_pops), np.nan)
            for ia, ib in itertools.combinations(range(num_pops), 2):
                p1_key, p2_key = pop_keys[ia], pop_keys[ib]
                fst_val = self._compute_multilocus_fst(
                    pop_indices[p1_key], pop_indices[p2_key], full_matrix
                )
                fst_mat[ia, ib] = fst_mat[ib, ia] = fst_val

            np.fill_diagonal(fst_mat, 0.0)
            df = pd.DataFrame(fst_mat, index=pop_keys, columns=pop_keys)
            self.logger.info("Fst calculation complete!")
            return df

        elif method == "permutation":
            result = {}
            pop_pairs = list(itertools.combinations(pop_keys, 2))

            seed_base = self.seed if self.seed is not None else None
            ss = np.random.SeedSequence(seed_base)
            children = ss.spawn(len(pop_pairs))

            pair_seeds = {
                pair: int(child.generate_state(1, dtype=np.uint32)[0])
                for pair, child in zip(pop_pairs, children)
            }

            max_workers = mp.cpu_count() if n_jobs == -1 else n_jobs

            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        FstDistance._permutation_worker,
                        pair,
                        pop_indices,
                        full_matrix,
                        n_reps,
                        pair_seeds[pair],
                    )
                    for pair in pop_pairs
                ]

                for fut in tqdm(
                    as_completed(futures),
                    desc=f"Fst population pairs ({n_reps} permutations each)",
                    total=len(futures),
                ):
                    pop_pair, res_dict = fut.result()
                    result[pop_pair] = res_dict

            for (p1_key, p2_key), res_dict in result.items():
                self.plotter.plot_permutation_dist(
                    res_dict["fst"],
                    res_dict["perm_dist"],
                    p1_key,
                    p2_key,
                    dist_type="Fst",
                )

            self.logger.info("Fst permutation testing complete!")
            return result

        elif method == "bootstrap":
            pop_pairs = list(itertools.combinations(pop_keys, 2))
            result = {pair: np.zeros(n_reps, dtype=float) for pair in pop_pairs}

            pair_components = {
                pair: FstDistance._fst_variance_components_per_locus(
                    pop_indices[pair[0]],
                    pop_indices[pair[1]],
                    full_matrix,
                )
                for pair in pop_pairs
            }

            worker_func = partial(
                FstDistance._bootstrap_replicate,
                pair_components=pair_components,
                pop_pairs=pop_pairs,
                n_loci=n_loci,
            )

            seeds = rng.integers(0, 1_000_000_000, size=n_reps)

            with ProcessPoolExecutor(
                max_workers=(mp.cpu_count() if n_jobs == -1 else n_jobs)
            ) as pool:
                reps = pool.map(worker_func, seeds)
                for i, rep_data in tqdm(
                    enumerate(reps), desc="Fst bootstraps", total=n_reps
                ):
                    for pair, fst_val in rep_data.items():
                        result[pair][i] = fst_val

            self.logger.info("Fst bootstrapping complete!")
            return result

        else:
            msg = f"Unknown method '{method}'. Choose from 'observed', 'permutation', or 'bootstrap'."
            self.logger.error(msg)
            raise ValueError(msg)

    def parse_wc_fst(self, result_dict, alpha: float = 0.05):
        """Convert the output of `weir_cockerham_fst()` into summary DataFrames.

        Args:
            result_dict (dict or pd.DataFrame): The output from
                ``weir_cockerham_fst()``.
            alpha (float): Significance level for confidence intervals. Defaults
                to ``0.05`` for 95% CIs.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: ``(df_mean, df_lower, df_upper, df_pval)``. For bootstrap results, ``df_mean`` is the observed multilocus Fst matrix, not the mean of bootstrap replicates.
        """
        if isinstance(result_dict, pd.DataFrame):
            self.snpio_mqc.queue_heatmap(
                df=result_dict,
                panel_id="wc_fst_observed",
                section="genetic_differentiation",
                title="SNPio: Pairwise Weir & Cockerham Fst (Observed)",
                description=(
                    "Observed pairwise Weir & Cockerham (1984) Fst for all "
                    "population pairs. No resampling was performed."
                ),
                index_label="Population",
                pconfig={
                    "title": "SNPio: Pairwise Weir & Cockerham Fst (Observed)",
                    "id": "wc_fst_observed",
                    "xlab": "Population",
                    "ylab": "Population",
                    "zlab": "Observed Fst",
                    "tt_decimals": 3,
                    "min": 0.0,
                    "max": 1.0,
                },
            )
            return result_dict, None, None, None

        first_val = next(iter(result_dict.values()))

        def all_populations_from_keys(dict_keys):
            pop_set = set()
            for key in dict_keys:
                pop_set.update(key)
            return sorted(pop_set)

        def get_pop_indices() -> dict[str, np.ndarray]:
            popmap = self.genotype_data.popmap_inverse
            sample_to_idx: dict[str, int] = {
                s: i for i, s in enumerate(self.genotype_data.samples)
            }

            if popmap is None:
                msg = "Population map is not available in genotype data, cannot get population indices."
                self.logger.error(msg)
                raise TypeError(msg)

            return {
                str(pop): np.array(
                    [sample_to_idx[s] for s in samples if s in sample_to_idx],
                    dtype=int,
                )
                for pop, samples in popmap.items()
            }

        df_pval = None

        if isinstance(first_val, np.ndarray):
            pop_pairs = list(result_dict.keys())
            pops = all_populations_from_keys(pop_pairs)
            pop_indices = get_pop_indices()
            full_matrix = self.genotype_data.snp_data

            df_obs = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_lower = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_upper = pd.DataFrame(np.nan, index=pops, columns=pops)

            for (p1, p2), arr in result_dict.items():
                obs_val = self._compute_multilocus_fst(
                    pop_indices[p1],
                    pop_indices[p2],
                    full_matrix,
                )

                arr_nonan = arr[~np.isnan(arr)]

                if len(arr_nonan) == 0:
                    lower_val = np.nan
                    upper_val = np.nan
                else:
                    lower_val = np.percentile(arr_nonan, 100 * alpha / 2)
                    upper_val = np.percentile(arr_nonan, 100 * (1 - alpha / 2))

                df_obs.loc[p1, p2] = df_obs.loc[p2, p1] = obs_val
                df_lower.loc[p1, p2] = df_lower.loc[p2, p1] = lower_val
                df_upper.loc[p1, p2] = df_upper.loc[p2, p1] = upper_val

            np.fill_diagonal(df_obs.values, 0.0)
            np.fill_diagonal(df_lower.values, 0.0)
            np.fill_diagonal(df_upper.values, 0.0)

            df_ul_combined = self._combine_upper_lower_ci(
                df_upper,
                df_lower,
                diagonal="zero",
            )

            self.snpio_mqc.queue_heatmap(
                df=df_obs,
                panel_id="wc_fst_bootstrap_observed",
                section="genetic_differentiation",
                title="SNPio: Pairwise Weir & Cockerham Fst (Observed)",
                description=(
                    "Observed pairwise Weir & Cockerham (1984) Fst. Confidence "
                    "intervals were estimated by bootstrap resampling loci with "
                    "replacement."
                ),
                index_label="Population",
                pconfig={
                    "title": "WC Fst (Observed)",
                    "id": "wc_fst_bootstrap_observed",
                    "xlab": "Population",
                    "ylab": "Population",
                    "zlab": "Observed Fst",
                    "tt_decimals": 3,
                    "min": 0.0,
                    "max": 1.0,
                    "reverse_colors": False,
                },
            )

            self.snpio_mqc.queue_heatmap(
                df=df_ul_combined,
                panel_id="wc_fst_bootstrap_ci95",
                section="genetic_differentiation",
                title="SNPio: Pairwise Weir & Cockerham Fst - 95% CIs",
                description=(
                    "Bootstrap confidence intervals from locus resampling. Upper "
                    "triangle shows upper CI; lower triangle shows lower CI."
                ),
                index_label="Population",
                pconfig={
                    "title": "WC Fst 95% CIs",
                    "id": "wc_fst_bootstrap_ci95",
                    "xlab": "Population",
                    "ylab": "Population",
                    "zlab": "Fst 95% CI",
                    "tt_decimals": 3,
                    "min": 0.0,
                    "max": 1.0,
                    "reverse_colors": False,
                },
            )

            return df_obs, df_lower, df_upper, df_pval

        if isinstance(first_val, dict) and "fst" in first_val and "pvalue" in first_val:
            pop_pairs = list(result_dict.keys())
            pops = all_populations_from_keys(pop_pairs)

            df_obs = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_pval = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_mean = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_lower = pd.DataFrame(np.nan, index=pops, columns=pops)
            df_upper = pd.DataFrame(np.nan, index=pops, columns=pops)

            for (p1, p2), subdict in result_dict.items():
                obs_val = subdict["fst"]
                pv_raw = subdict["pvalue"]

                try:
                    p_value = float(pv_raw) if pv_raw is not None else np.nan
                except (TypeError, ValueError):
                    p_value = np.nan

                dist = subdict.get("perm_dist", None)

                df_obs.loc[p1, p2] = df_obs.loc[p2, p1] = obs_val
                df_pval.loc[p1, p2] = df_pval.loc[p2, p1] = p_value

                if dist is not None and len(dist) > 0:
                    dist_nonan = dist[~np.isnan(dist)]

                    if len(dist_nonan) == 0:
                        mean_val = np.nan
                        lower_val = np.nan
                        upper_val = np.nan
                    else:
                        mean_val = np.mean(dist_nonan)
                        lower_val = np.percentile(dist_nonan, 100 * alpha / 2)
                        upper_val = np.percentile(dist_nonan, 100 * (1 - alpha / 2))

                    df_mean.loc[p1, p2] = df_mean.loc[p2, p1] = mean_val
                    df_lower.loc[p1, p2] = df_lower.loc[p2, p1] = lower_val
                    df_upper.loc[p1, p2] = df_upper.loc[p2, p1] = upper_val

            np.fill_diagonal(df_obs.values, 0.0)
            np.fill_diagonal(df_pval.values, 1.0)
            np.fill_diagonal(df_mean.values, 0.0)
            np.fill_diagonal(df_lower.values, 0.0)
            np.fill_diagonal(df_upper.values, 0.0)

            self.snpio_mqc.queue_heatmap(
                df=df_obs,
                panel_id="wc_fst_permutation_observed",
                section="genetic_differentiation",
                title="SNPio: Pairwise Weir & Cockerham Fst (Observed; Permutation Test)",
                description=(
                    "Observed pairwise Weir & Cockerham (1984) Fst. Significance "
                    "is assessed via a permutation test: p = Pr(Fst_perm >= Fst_obs)."
                ),
                index_label="Population",
                pconfig={
                    "title": "WC Fst (Observed; Permutation Test)",
                    "id": "wc_fst_permutation_observed",
                    "xlab": "Population",
                    "ylab": "Population",
                    "zlab": "Observed Fst",
                    "reverse_colors": False,
                    "min": 0.0,
                    "max": 1.0,
                    "tt_decimals": 3,
                },
            )

            self.snpio_mqc.queue_heatmap(
                df=df_pval,
                panel_id="wc_fst_permutation_pvalues",
                section="genetic_differentiation",
                title="SNPio: P-values for Pairwise Weir & Cockerham Fst",
                description=(
                    "One-tailed permutation p-values: probability that a permuted "
                    "Fst is greater than or equal to the observed Fst."
                ),
                index_label="Population",
                pconfig={
                    "title": "WC Fst Permutation p-values",
                    "id": "wc_fst_permutation_pvalues",
                    "xlab": "Population",
                    "ylab": "Population",
                    "zlab": "Permutation p-value",
                    "reverse_colors": True,
                    "min": 0.0,
                    "max": 1.0,
                    "tt_decimals": 3,
                },
            )

            return df_obs, df_lower, df_upper, df_pval

        msg = (
            "Unrecognized structure in result_dict when estimating Fst. Expected "
            "either a DataFrame, a bootstrap dictionary of arrays, or a permutation "
            "dictionary with 'fst' and 'pvalue' keys."
        )
        self.logger.error(msg)
        raise ValueError(msg)

    def _combine_upper_lower_ci(
        self, df_upper: pd.DataFrame, df_lower: pd.DataFrame, diagonal="zero"
    ) -> pd.DataFrame:
        """Combines two square DataFrames into one, using the upper triangle from one and the lower from the other.

        Args:
            df_upper (pd.DataFrame): DataFrame to provide the upper triangle (including diagonal if diagonal='upper').
            df_lower (pd.DataFrame): DataFrame to provide the lower triangle (including diagonal if diagonal='lower').
            diagonal (str): Which DataFrame should provide the diagonal values. Options are 'upper', 'lower', "zero", or 'nan'.

        Returns:
            pd.DataFrame: Combined DataFrame with upper/lower triangle values from respective inputs.

        Raises:
            ValueError: If input DataFrames are not square or do not match in shape.
        """
        if df_upper.shape != df_lower.shape:
            msg = "Both DataFrames must have the same shape."
            self.logger.error(msg)
            raise AssertionError(msg)

        if df_upper.shape[0] != df_upper.shape[1]:
            msg = "Input DataFrames must be square."
            self.logger.error(msg)
            raise AssertionError(msg)

        n = df_upper.shape[0]

        upper_mask = np.triu(np.ones((n, n)), k=1)
        lower_mask = np.tril(np.ones((n, n)), k=-1)
        diag_mask = np.eye(n)

        combined = np.full_like(df_upper, np.nan, dtype="float64")

        combined[upper_mask.astype(bool)] = df_upper.values[upper_mask.astype(bool)]
        combined[lower_mask.astype(bool)] = df_lower.values[lower_mask.astype(bool)]

        if diagonal == "upper":
            combined[diag_mask.astype(bool)] = df_upper.values[diag_mask.astype(bool)]
        elif diagonal == "lower":
            combined[diag_mask.astype(bool)] = df_lower.values[diag_mask.astype(bool)]
        elif diagonal == "nan":
            pass  # leave diagonal as NaN
        elif diagonal == "zero":
            np.fill_diagonal(combined, 0.0)
        else:
            msg = (
                "Invalid option for 'diagonal'. Choose from 'upper', 'lower', or 'nan'."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        return pd.DataFrame(combined, index=df_upper.index, columns=df_upper.columns)

    @staticmethod
    def _fst_variance_components_per_locus(
        pop1_inds: np.ndarray,
        pop2_inds: np.ndarray,
        full_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-locus Weir-Cockerham Fst variance components.

        Args:
            pop1_inds (np.ndarray): Population 1 sample indices.
            pop2_inds (np.ndarray): Population 2 sample indices.
            full_matrix (np.ndarray): Genotype matrix with individuals as rows and
                loci as columns.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Per-locus ``a``, ``b``, and
                ``c`` component arrays. These correspond to HierFstat's ``lsiga``,
                ``lsigb``, and ``lsigw`` per locus.
        """
        n_loci = full_matrix.shape[1]

        a_vals = np.full(n_loci, np.nan, dtype=float)
        b_vals = np.full(n_loci, np.nan, dtype=float)
        c_vals = np.full(n_loci, np.nan, dtype=float)

        for loc in range(n_loci):
            s1 = FstDistance._decode_and_clean_phased_genotypes(
                full_matrix[pop1_inds, loc]
            )
            s2 = FstDistance._decode_and_clean_phased_genotypes(
                full_matrix[pop2_inds, loc]
            )

            if not s1 or not s2:
                continue

            a, b, c = FstDistance._two_pop_weir_cockerham_fst_locus(s1, s2)

            a_vals[loc] = a
            b_vals[loc] = b
            c_vals[loc] = c

        return a_vals, b_vals, c_vals

    def fst_variance_components_per_locus(
        self,
        pop1_inds: np.ndarray,
        pop2_inds: np.ndarray,
        full_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-locus numerator and denominator components for pairwise Fst.

        Args:
            pop1_inds (np.ndarray): Population 1 sample indices.
            pop2_inds (np.ndarray): Population 2 sample indices.
            full_matrix (np.ndarray): Genotype matrix with individuals as rows and
                loci as columns.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Per-locus ``a``, ``b``, and
                ``c`` component arrays.
        """
        return FstDistance._fst_variance_components_per_locus(
            pop1_inds,
            pop2_inds,
            full_matrix,
        )
