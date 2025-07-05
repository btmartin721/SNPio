from tqdm import tqdm
from typing import Dict, Literal

import numpy as np
import pandas as pd
from kneed import KneeLocator
from polars import corr
from scipy.stats import ks_2samp
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from statsmodels.stats.multitest import multipletests

from snpio.utils.logging import LoggerManager


class DBSCANOutlierDetector:
    """Estimate p-values for DBSCAN outliers via Monte Carlo, and flag per-pair Fst outliers.

    This class implements a DBSCAN-based outlier detection method for Fst matrices, allowing for the identification of loci that are significantly different from their neighbors in a multivariate feature space.
    """

    def __init__(
        self,
        prefix: str = "snpio",
        eps: Literal["auto"] | float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        n_jobs: int = -1,
        standardize: (
            Literal["standardize", "robust", "power", "quantile"] | None
        ) = None,
        null_model: Literal["shuffle", "uniform"] = "shuffle",
        n_simulations: int = 1000,
        random_state: int | None = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initialize the DBSCANOutlierDetector with parameters for DBSCAN clustering and outlier detection.

        This constructor sets up the DBSCAN parameters, initializes the random number generators for Monte Carlo simulations, and configures the scaler for feature standardization if specified. It also sets up logging for the class.

        Args:
            prefix (str): prefix for logging and output files.
            eps (float | Literal["auto"]): radius for DBSCAN; 'auto' to estimate from data.
            min_samples (int): minimum number of samples in a neighborhood for core points.
            metric (str): distance metric for DBSCAN.
            n_jobs (int): number of parallel jobs for DBSCAN; -1 for all available cores.
            standardize (Literal["standardize", "robust", "power", "quantile"] | None): if not None, which transformer to apply feature-wise.
            null_model (Literal["shuffle", "uniform"]): 'shuffle' or 'uniform'.
            n_simulations (int): number of Monte Carlo replicates.
            random_state (int | None): seed for reproducibility.
            verbose (bool): whether to log info messages.
            debug (bool): whether to log debug messages.

        """
        # … [same validation of args as before] …
        logman = LoggerManager(__name__, prefix=prefix, verbose=verbose, debug=debug)
        self.logger = logman.get_logger()

        # set up scaler
        if standardize == "standardize":
            self._scaler = StandardScaler()
        elif standardize == "robust":
            self._scaler = RobustScaler()
        elif standardize == "power":
            self._scaler = PowerTransformer(method="yeo-johnson")
        elif standardize == "quantile":
            self._scaler = QuantileTransformer(
                output_distribution="normal", random_state=random_state
            )
        else:
            self._scaler = None

        # RNG per replicate
        self._rngs = [
            np.random.default_rng(
                random_state + i if random_state is not None else None
            )
            for i in range(n_simulations)
        ]

        # store basic params
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs
        self.null_model = null_model
        self.n_simulations = n_simulations

    def fit(self, data: pd.DataFrame) -> None:
        """Run DBSCAN on your Fst-matrix (loci x pairs) and store neighbor-counts.

        This method fits the DBSCAN model to the provided Fst matrix, estimates the eps value if set to "auto", and computes the neighbor counts for each locus. The results are stored in the instance variables `labels_` and `neighbor_counts_`.

        Args:
            data (pd.DataFrame): Fst matrix as pandas DataFrame. It should have loci as rows and population pairs as columns.

        Raises:
            TypeError: If the input data is not a pandas DataFrame.
            ValueError: If the input data is empty or has no columns.
            RuntimeError: If the eps value cannot be estimated or is invalid.
        """

        if not isinstance(data, pd.DataFrame):
            msg = f"Input data must be a pandas DataFrame, but got: {type(data)}"
            self.logger.error(msg)
            raise TypeError(msg)

        self._pair_names = data.columns.tolist()

        X = data.to_numpy(copy=True)

        if self._scaler is not None:
            X = self._scaler.fit_transform(X)

        # estimate eps if needed …
        if isinstance(self.eps, str) and self.eps.lower() == "auto":
            self.eps = self._estimate_eps(
                pd.DataFrame(X), self.min_samples, percentile_cap=75
            )

        self.db = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )

        self.labels_ = self.db.fit_predict(X)

        nbrs = NearestNeighbors(radius=self.eps, metric=self.metric)
        nbrs.fit(X)
        self.neighbor_counts_ = np.array(
            [len(neis) - 1 for neis in nbrs.radius_neighbors(X, return_distance=False)],
            dtype=int,
        )

        self._n_samples = X.shape[0]

        # NOTE: Store the (un‐scaled) matrix so p‐value routine sees same shape
        self._last_data = data.to_numpy(copy=True)

    def _generate_null_data(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """One MC replicate: either shuffle each column or draw uniform within its range.

        This method generates a null dataset by either shuffling each column of the input data matrix or drawing uniformly within the range of each feature. The choice of method is determined by the `null_model` parameter.

        Args:
            X (np.ndarray): The input data matrix (loci x features).
            rng (np.random.Generator): Random number generator for reproducibility.

        Returns:
            np.ndarray: A null dataset with the same shape as X, generated by either shuffling each column or drawing uniformly within the range of each feature.
        """
        n, p = X.shape
        if self.null_model == "shuffle":
            Xn = np.empty_like(X)
            for j in range(p):
                Xn[:, j] = rng.permutation(X[:, j])
            return Xn
        else:  # 'uniform'
            mins, maxs = X.min(axis=0), X.max(axis=0)
            return rng.uniform(mins, maxs, size=(n, p))

    def estimate_pvalues(self) -> np.ndarray:
        """Estimate p-values for loci based on their multivariate outlier status using a Monte Carlo approach on the k-Nearest Neighbor (k-NN) distance.

        The k-NN distance serves as a score for "outlierness" that is consistent with the density-based logic of DBSCAN. A locus with a large k-NN distance is in a sparse region of the N-dimensional feature space.

        Returns:
            np.ndarray: A 1D array of p-values, one for each locus.
        """
        X_obs = self._last_data.copy()

        if self._scaler:
            X_obs = self._scaler.transform(X_obs)

        k = self.min_samples

        # 1. Calculate observed test statistic (k-NN distance) for each locus
        self.logger.info(f"Calculating observed k-NN distances (k={k})...")

        nn = NearestNeighbors(n_neighbors=k, metric=self.metric, n_jobs=self.n_jobs)

        nn.fit(X_obs)

        # The distance to the k-th neighbor is the last column of the distances
        # matrix
        observed_knn_dists = nn.kneighbors(X_obs)[0][:, -1]

        # 2. Generate a null distribution of k-NN distances via Monte Carlo
        self.logger.info(
            f"Generating null distribution from {self.n_simulations} simulations..."
        )
        null_knn_dists_list = []
        for rng in tqdm(self._rngs, desc="Simulation", unit="sim"):
            # Generate one null dataset by shuffling columns
            X_null = self._generate_null_data(X_obs, rng)

            # Calculate k-NN distances for this null dataset
            nn_null = NearestNeighbors(
                n_neighbors=k, metric=self.metric, n_jobs=self.n_jobs
            )
            nn_null.fit(X_null)
            null_dists_for_sim = nn_null.kneighbors(X_null)[0][:, -1]
            null_knn_dists_list.append(null_dists_for_sim)

        # 3. Pool all null distances into a single empirical distribution
        null_knn_dists = np.concatenate(null_knn_dists_list)

        # 4. Calculate p-values by comparing each observed distance to the null distribution
        if null_knn_dists.size == 0:
            msg = "No null distances generated. Check the null model settings."
            self.logger.error(msg)
            raise RuntimeError(msg)

        self.logger.info("Calculating final p-values...")

        # For each observed distance, find how many null distances are greater
        # or equal Using broadcasting for an efficient, vectorized computation
        counts = np.sum(null_knn_dists[:, np.newaxis] >= observed_knn_dists, axis=0)
        pvals = (counts + 1) / (len(null_knn_dists) + 1)

        return pvals  # 1D array of shape (n_loci,)

    def identify_outliers(
        self,
        pvals: np.ndarray,
        alpha: float = 0.05,
        correction_method: str | None = "fdr_bh",
    ) -> tuple[pd.Series, pd.Series]:
        """Identify outlier loci from p-values and apply multiple testing correction.

        This method identifies significant outlier loci based on their p-values and applies a multiple testing correction if specified. It returns a boolean Series indicating whether each locus is a significant outlier and a Series of the corrected p-values (i.e., q-values).

        Args:
            pvals (np.ndarray): A 1D array of p-values, one per locus.
            alpha (float): Significance level.
            correction_method (str | None): Method for multiple testing correction. Use None to skip correction.

        Returns:
            tuple[pd.Series, pd.Series]: A boolean Series indicating if a locus is a significant outlier, and a Series of the corrected p-values (q-values).
        """
        if not isinstance(pvals, np.ndarray):
            msg = f"Input p-values must be a 1D numpy array, but got: {type(pvals)}"
            self.logger.error(msg)
            raise TypeError(msg)

        if pvals.ndim != 1:
            msg = f"Input p-values must be a 1D array, but got shape: {pvals.shape}"
            self.logger.error(msg)
            raise ValueError(msg)

        if correction_method is not None:
            # Perform correction on the single vector of p-values
            reject, qvals, _, _ = multipletests(
                pvals, alpha=alpha, method=correction_method
            )
        else:
            # Without correction, outliers are just those below the alpha
            # threshold
            reject = pvals < alpha
            qvals = pvals

        # Return as pandas Series for easy use with original data's index
        outliers = pd.Series(reject, name="is_outlier")
        q_values = pd.Series(qvals, name="q_value")

        return outliers, q_values

    def validate(
        self, data: np.ndarray | pd.DataFrame | list | dict
    ) -> dict[str, float]:
        """Compute silhouette (if possible) and KS test between inlier/outlier neighbor counts.

        This method evaluates the clustering quality by computing the silhouette score and performing a KS test between inlier and outlier neighbor counts.

        Args:
            data: original data for silhouette.
        Returns:
            validation (dict):
                'silhouette_score': silhouette score for clusters (or np.nan)
                'ks_statistic': KS statistic between neighbor-counts of inliers vs. outliers
                'ks_pvalue': corresponding p-value
        """
        if not isinstance(data, (pd.DataFrame, np.ndarray, list, dict)):
            msg = f"Input data must be a pandas DataFrame, numpy.ndarray, list, or dict, but got: {type(data)}"
            self.logger.error(msg)
            raise ValueError(msg)

        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception as e:
                msg = f"Failed to convert input data to DataFrame: {e}"
                self.logger.error(msg)
                raise ValueError(msg)

        # ensure data is in the same format as used in fit
        X = data.to_numpy(copy=True)

        if self._scaler is not None:
            X = self._scaler.transform(X)

        res: Dict[str, float] = {}

        # silhouette only if ≥2 clusters (ignoring outliers)
        labels = self.labels_
        core_mask = labels >= 0
        if not np.any(core_mask):
            res["silhouette_score"] = np.nan
            res["ks_statistic"] = np.nan
            res["ks_pvalue"] = np.nan
            return res

        unique_clusters = np.unique(labels[core_mask])
        if len(unique_clusters) >= 2:
            res["silhouette_score"] = float(
                silhouette_score(X[core_mask], labels[core_mask])
            )

        else:
            res["silhouette_score"] = np.nan

        # KS test on neighbor-count distributions
        in_counts = self.neighbor_counts_[labels >= 0]
        out_counts = self.neighbor_counts_[labels == -1]

        if len(in_counts) > 0 and len(out_counts) > 0:
            ks_stat, ks_p = ks_2samp(in_counts, out_counts)
            res["ks_statistic"] = float(ks_stat)
            res["ks_pvalue"] = float(ks_p)

        else:
            res["ks_statistic"] = np.nan
            res["ks_pvalue"] = np.nan

        return res

    def _estimate_eps(
        self,
        fst_values: pd.DataFrame,
        min_samples: int,
        percentile_cap: int,
        fallbacks: dict = {"eps": 0.1, "percentile": 30},
    ) -> float:
        """Estimate the optimal eps value for DBSCAN clustering.

        This method estimates the optimal eps value for DBSCAN clustering using the k-distance graph method. The optimal eps value is determined based on the knee point of the k-distance graph.

        Args:
            fst_values (pd.DataFrame): Fst values between population pairs.
            min_samples (int): Minimum number of samples required for DBSCAN clustering.
            percentile_cap (int): Maximum percentile cap for the eps value. Defaults to 75, which represents the 75th percentile of the k-distance values.
            fallbacks (dict): Fallback values for eps and percentile if the knee point cannot be determined. Defaults to {"eps": 0.1, "percentile": 30}.

        Returns:
            float: The optimal eps value for DBSCAN clustering.
        """
        if "eps" not in fallbacks or "percentile" not in fallbacks:
            msg = "Fallbacks must contain 'eps' and 'percentile' keys."
            self.logger.error(msg)
            raise ValueError(msg)

        # Step 4: Compute the k-distance graph to find optimal eps
        neighbor = NearestNeighbors(n_neighbors=min_samples)
        nbrs: NearestNeighbors = neighbor.fit(fst_values)
        distances, _ = nbrs.kneighbors(fst_values)

        # Sort the distances to the min_samples-th nearest neighbor
        distances_k = np.sort(distances[:, -1])

        # Check if distances are all zeros
        if np.all(distances_k == 0):
            self.logger.warning(
                "Distances all zeros. Setting eps to a small positive value (0.1)."
            )
            eps = fallbacks["eps"]

        else:
            # Use KneeLocator to find the knee point
            kneedle = KneeLocator(
                range(len(distances_k)),
                distances_k,
                S=1.0,
                curve="convex",
                direction="increasing",
            )

            if kneedle.knee is not None:
                eps = distances_k[kneedle.knee]

            else:
                # Fallback: Use a percentile of the distances
                eps = np.percentile(distances_k, fallbacks["percentile"])
                self.logger.warning(
                    f"Automated eps could not identify a knee. Using eps from {fallbacks['percentile']}th percentile: {eps}"
                )

        if isinstance(eps, (np.float32, np.float64)):
            eps = float(eps)

        eps_cap = np.percentile(distances_k, percentile_cap)
        eps = min(eps, eps_cap)  # Cap eps at percentile of distances
        return eps
