.. _unphased-ld-ne:

Unphased Linkage Disequilibrium and Recent Effective Size
=========================================================

SNPio estimates finite-sample-unbiased linkage-disequilibrium (LD) component
moments from unphased, biallelic diploid SNPs and uses LD between genuinely
unlinked loci to estimate recent effective population size. The implementation
follows :cite:t:`RagsdaleGravel2020` and the unlinked-locus expectations of
:cite:t:`WeirHill1980`.

The production implementation is self-contained. ``moments-popgen``,
``fwdpy11``, ``msprime``, and ``tskit`` are not runtime dependencies. They are
used only in the independent validation infrastructure described in
:doc:`ld_validation`.

What SNPio estimates
--------------------

For two biallelic loci with allele frequencies :math:`p` and :math:`q` and
gametic covariance :math:`D`, SNPio estimates the Hill-Robertson component
moments :cite:p:`HillRobertson1968`

.. math::

   D^2, \qquad
   Dz = D(1 - 2p)(1 - 2q), \qquad
   \pi_2 = p(1-p)q(1-q).

The normalized population statistics are ratios of aggregated component
moments across eligible locus pairs:

.. math::

   r_D^2 = \frac{\sum_{i,j}\widehat{D^2}_{i,j}}
                  {\sum_{i,j}\widehat{\pi_2}_{i,j}},
   \qquad
   r_{Dz} = \frac{\sum_{i,j}\widehat{Dz}_{i,j}}
                   {\sum_{i,j}\widehat{\pi_2}_{i,j}}.

This ratio-of-sums definition is important. SNPio does **not** average the
pairwise ratios ``D2 / Pi2``. The pairwise diagnostic ``r2_star`` remains a
biased ratio and can be negative or greater than one.

For unlinked loci, :math:`c=1/2`, and the equilibrium random-mating
expectation is

.. math::

   E[r_D^2] \simeq \frac{1}{3N_e}.

SNPio therefore reports

.. math::

   \widehat{N}_e = \frac{1}{3\widehat{r_D^2}}

for ``mating_system="random"``. For the monogamous relationship described by
:cite:t:`WeirHill1980`, SNPio uses
:math:`\widehat{N}_e=2/(3\widehat{r_D^2})`.

The :math:`r_{Dz}` statistic is not converted to :math:`N_e`. Its unlinked
expectation is near zero and it is reported as a diagnostic for departures
from the simple population model, including recent migration or population
structure :cite:p:`RagsdaleGravel2020`.

Finite-sample unbiased polynomial algorithm
--------------------------------------------

Each complete diploid at a locus pair belongs to one of nine unphased
two-locus genotype states:

.. code-block:: text

   AABB  AABb  AAbb  AaBB  AaBb  Aabb  aaBB  aaBb  aabb

Let :math:`n_j` be the count of state :math:`j` and
:math:`n=\sum_j n_j`. A two-locus statistic that can be written as a
polynomial in the nine population genotype frequencies,

.. math::

   S = \sum_i a_i \prod_{j=1}^{9} g_j^{k_{ij}},

has the unbiased multinomial estimator

.. math::

   \widehat{S}
   = \sum_i a_i
     \frac{\prod_{j=1}^{9}(n_j)_{k_{ij}}}{(n)_{k_i}},
   \qquad
   k_i=\sum_{j=1}^{9}k_{ij},

where :math:`(x)_k=x(x-1)\cdots(x-k+1)` is a falling factorial. This is the
computational form of the sampling-without-replacement correction derived by
:cite:t:`RagsdaleGravel2020`.

SNPio constructs exact sparse coefficient and exponent tables for ``D``,
``D2``, ``Dz``, and ``Pi2`` in ``snpio/popgenstats/ld_polynomials.py`` and
evaluates them with Numba-compiled kernels. ``D`` is second order; ``D2``,
``Dz``, and ``Pi2`` are fourth order. Consequently, at least four complete
diploid individuals are required for every retained locus pair. Missing data
are handled pairwise, and non-biallelic or monomorphic loci are excluded where
the required two-allele representation is unavailable.

Unbiased component estimates are not constrained estimators. Individual
``D2`` or ``Pi2`` estimates can be slightly negative, just as an unbiased
variance-component estimate can fall outside its population parameter space.
This behavior is not evidence of a coding error.

Selecting genuinely unlinked pairs
-----------------------------------

The :math:`N_e` conversion is valid only for unlinked loci. SNPio uses the
following precedence:

1. ``locus_groups`` supplied by the user. Only pairs from different groups are
   eligible.
2. For VCF-derived data, chromosome or scaffold labels parsed from marker
   names. Within-chromosome or within-scaffold pairs are excluded.
3. ``assume_unlinked=True`` as an explicit assertion for coordinate-free or
   independently pruned data.

``assume_unlinked=True`` does not test linkage and should not be used to
override known chromosome or scaffold relationships. For PHYLIP, STRUCTURE,
or GENEPOP data, provide ``locus_groups`` whenever linkage-group metadata are
available.

Aggregation, pair sampling, and parallel execution
---------------------------------------------------

Eligible pairs are divided into grouped block-pair tasks. SNPio evaluates the
nine-state counts and polynomial moments in vectorized chunks controlled by
``pair_chunk_size``. ``n_jobs`` runs independent block-pair tasks in threads;
``-1`` uses all available CPUs. A fixed ``seed`` makes pair selection,
blocking, and bootstrap resampling reproducible across serial and parallel
runs.

For very large datasets, ``max_pairs`` uniformly subsamples eligible pairs for
the aggregate calculation. Set ``max_pairs=None`` to evaluate every eligible
pair when computationally practical. ``pairwise_sample_size`` controls only
the number of pairwise rows retained for plotting and optional output; it does
not reduce the aggregate calculation below ``max_pairs``.

Bootstrap confidence intervals
------------------------------

By default, SNPio randomly groups loci into ``n_bootstrap_blocks`` and
resamples grouped block-pair summaries. This follows the grouped-locus strategy
used for the unphased LD estimator.

When chromosomes, scaffolds, or another biological grouping are the intended
resampling units, pass one label per locus through ``bootstrap_groups``. SNPio
then performs a node-cluster bootstrap: biological groups are sampled with
replacement, and each group-pair contribution is weighted by the product of
its sampled multiplicities. This preserves dependence among pair summaries
that share a chromosome or scaffold.

Percentile intervals are reported for :math:`r_D^2` and :math:`r_{Dz}`. The
:math:`N_e` interval is obtained by monotonic inversion of the complete
:math:`r_D^2` interval. If the lower :math:`r_D^2` endpoint is nonpositive,
the upper :math:`N_e` endpoint is unbounded. A nonpositive point estimate of
:math:`r_D^2` does not yield a finite :math:`N_e` estimate.

SNPio retains this outcome as a missing :math:`N_e`, emits an explanatory
warning, and labels it as not estimable in the MultiQC summary. Such
populations are omitted from the MultiQC :math:`N_e` bar plot without being
removed from the scientific result table.

Python API
----------

The public entry point is
``PopGenStatistics.calculate_linkage_disequilibrium``.

.. code-block:: python

   from snpio import PopGenStatistics, VCFReader

   genotypes = VCFReader(
       filename="data.vcf.gz",
       popmapfile="popmap.txt",
       force_popmap=True,
       prefix="ld_analysis",
       plot_format="png",
   )

   result = PopGenStatistics(genotypes).calculate_linkage_disequilibrium(
       populations=None,          # Analyze every mapped population.
       include_overall=False,     # Avoid pooling structured populations.
       n_bootstraps=200,
       n_bootstrap_blocks=20,
       n_jobs=-1,
       max_pairs=1_000_000,
       pair_chunk_size=25_000,
       pairwise_sample_size=100_000,
       mating_system="random",
       alpha=0.05,
       seed=42,
       save_pairwise=True,
       save_plots=True,
   )

For coordinate-free data with known linkage groups:

.. code-block:: python

   result = PopGenStatistics(genotypes).calculate_linkage_disequilibrium(
       locus_groups=linkage_group_per_locus,
       bootstrap_groups=linkage_group_per_locus,
       n_jobs=4,
       seed=42,
   )

Use ``assume_unlinked=True`` only when the input has already been restricted
to independently segregating loci and no group labels exist.

Returned result and generated files
-----------------------------------

The returned ``LinkageDisequilibriumResult`` contains:

``summary``
   One row per population with sample, locus, and pair counts; aggregate
   components; ``r2D``; ``rDz``; ``Ne``; and confidence intervals.
   Non-estimable ``Ne`` values remain ``NaN``. The MultiQC representation adds
   a derived ``Ne_Status`` label for readability.

``bootstrap``
   Bootstrap replicate statistics.

``block_summaries``
   Aggregated component moments for each block pair.

``pairwise_sample``
   A bounded diagnostic sample of locus-pair estimates.

``metadata``
   Method, grouping source, pair budgets, bootstrap method, seed, mating
   system, and other reproducibility information.

``files``
   Paths to generated tables and plots.

Reports are written to
``<prefix>_output/reports/linkage_disequilibrium/`` and figures to
``<prefix>_output/plots/linkage_disequilibrium/``. Filtered datasets use
``reports/nremover/linkage_disequilibrium/`` and
``plots/nremover/linkage_disequilibrium/``.

MultiQC panels
--------------

When ``save_plots=True``, the analysis also queues a population summary and
interactive LD/Ne panels for the next ``SNPioMultiQC.build()`` call:

- a summary table retaining sample, locus, pair, ``r2D``, ``rDz``, and ``Ne``
  information, plus a derived ``Ne_Status`` label;
- population bar, heatmap, and categorical line panels for ``r2D`` and
  ``rDz``;
- finite-``Ne`` bar, heatmap, and categorical line panels;
- a population-grouped bootstrap boxplot for ``r2D``, ``rDz``, ``D``, ``Dz``,
  and ``Pi2``; and
- a separate population-grouped bootstrap boxplot for ``Ne``.

Each bootstrap box is built from replicate-level long-form data. Population
identity remains on the categorical x-axis, LD statistic controls color, and
the bootstrap replicate is available in the hover data. This avoids pooling
observations across populations. ``Ne`` is deliberately separated because it
has an incompatible scale and is inversely related to ``r2D``.

Nonfinite values are sanitized only at the reporting boundary. LD statistics
and ``Ne`` are filtered independently, so a non-estimable ``Ne`` never removes
that population's finite LD statistics. The scientific result table retains
the missing value and status; only plots requiring finite ``Ne`` omit it.
Setting ``save_plots=False`` suppresses both the static figures and the LD/Ne
MultiQC panels while leaving the returned result and report tables available.

Plot colors, scales, and diagnostic warnings
--------------------------------------------

The LD figures use a color-blind-safe Okabe--Ito palette with redundant marker
shapes. Blue circles represent ordinary estimates, while orange diamonds mark
a diagnostic warning. Error bars show 95% bootstrap confidence intervals and
gray dashed lines mark zero. In the pairwise-distribution figure, every
population uses the same blue boxes because population identity is already
encoded by the x-axis; the dark center line is the median, the box spans the
interquartile range, and whiskers extend to 1.5 times that range.

An orange diagnostic diamond means that the 95% bootstrap confidence interval
for :math:`r_{Dz}` excludes zero. Under the simple unlinked population model,
:math:`r_{Dz}` is expected to be near zero. A consistently non-zero signal can
therefore indicate a departure from that model, including recent migration or
population structure. The warning does not identify the underlying cause and
does not mean that the calculation failed. It is informative because the
conversion from :math:`r_D^2` to :math:`N_e` relies on the simple population
model; the corresponding :math:`N_e` estimate should consequently be treated
with additional caution.

Signed statistics (``r2D``, ``rDz``, ``D``, ``D2``, ``Dz``, and pairwise
``r2_star``) use linear axes so that sign and distance from zero remain
directly interpretable. SNPio uses a logarithmic axis only for strictly
positive :math:`\pi_2` or :math:`N_e` values spanning at least two orders of
magnitude. Axis labels explicitly identify a log scale and clarify that the
displayed values themselves are not transformed.

Interpretation and limitations
------------------------------

- Analyze populations separately unless a pooled sample is biologically
  panmictic. Pooling structured populations can create LD unrelated to drift
  within one population.
- The estimators assume randomly sampled, non-inbred diploid individuals.
  Inbreeding and nonrandom mating change the relevant LD expectations.
- The reported recent :math:`N_e` is an LD-based effective size under the
  selected mating model, not a census size and not a complete demographic
  history.
- SNP ascertainment, low minor-allele frequencies, small samples, and an
  :math:`r_D^2` estimate near zero can produce wide or unbounded intervals.
- Pair subsampling adds Monte Carlo variability. Increase ``max_pairs`` and
  compare seeded runs when a population estimate is sensitive to the pair
  budget.
- Bootstrap intervals describe uncertainty in the resampled genomic units
  conditional on the sampled individuals. They do not automatically include
  individual-sampling uncertainty.

Validation resources
--------------------

The executable validation framework is maintained in the repository's
`snpio/validation/ directory
<https://github.com/btmartin721/SNPio/tree/master/snpio/validation>`_. The
unphased LD and :math:`N_e` validation implementation is split into
`linkage_disequilibrium.py
<https://github.com/btmartin721/SNPio/blob/master/snpio/validation/linkage_disequilibrium.py>`_,
`ld_simulation.py
<https://github.com/btmartin721/SNPio/blob/master/snpio/validation/ld_simulation.py>`_,
and `ld_plots.py
<https://github.com/btmartin721/SNPio/blob/master/snpio/validation/ld_plots.py>`_.
Frozen external-reference fixtures are under ``snpio/validation/data/``.

Curated, checksummed result evidence is tracked separately in the root
`validation directory
<https://github.com/btmartin721/SNPio/tree/master/validation>`_. See
:doc:`ld_validation` for the validation design, commands, acceptance criteria,
and figures, and :doc:`tutorial` for a worked analysis.
