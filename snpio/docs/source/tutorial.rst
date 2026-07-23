.. _snpio-tutorial:

SNPio Tutorial
==============

This tutorial demonstrates a reproducible unphased LD and recent effective
population-size analysis using the SNPio example VCF and population map. See
:doc:`linkage_disequilibrium` for the algorithm and assumptions and
:doc:`ld_validation` for the validation evidence.

Install SNPio
-------------

Install from PyPI in an isolated environment:

.. code-block:: shell

   python3 -m venv snpio-env
   source snpio-env/bin/activate
   python -m pip install snpio

Conda and Docker installations are documented in :doc:`getting_started` and
:doc:`publication`.

Load the example data
---------------------

Run this example from a repository checkout so the bundled paths are
available:

.. code-block:: python

   from snpio import PopGenStatistics, VCFReader

   vcf = "snpio/example_data/vcf_files/phylogen_subset14K.vcf.gz"
   popmap = "snpio/example_data/popmaps/phylogen_nomx.popmap"

   genotypes = VCFReader(
       filename=vcf,
       popmapfile=popmap,
       force_popmap=True,
       prefix="snpio_ld_tutorial",
       plot_format="png",
       verbose=True,
   )

For a user dataset, replace ``vcf`` and ``popmap`` with your own paths. VCF
chromosome or scaffold labels are used as linkage groups, so the default
analysis excludes within-group pairs.

Run unphased LD and recent :math:`N_e`
--------------------------------------

.. code-block:: python

   statistics = PopGenStatistics(genotypes)

   ld = statistics.calculate_linkage_disequilibrium(
       populations=None,
       include_overall=False,
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

``include_overall=False`` is deliberate: the example contains multiple
populations, and an overall pooled estimate would mix population structure
with within-population LD. ``seed`` makes pair sampling and bootstrapping
reproducible. ``max_pairs`` limits the aggregate pair budget; increase it or
set it to ``None`` when a full-pair analysis is computationally practical.

Inspect the population summary
------------------------------

.. code-block:: python

   columns = [
       "Population",
       "Samples",
       "Loci",
       "Pairs",
       "r2D",
       "r2D_CI_Lower",
       "r2D_CI_Upper",
       "rDz",
       "rDz_CI_Lower",
       "rDz_CI_Upper",
       "Ne",
       "Ne_CI_Lower",
       "Ne_CI_Upper",
   ]
   print(ld.summary[columns].to_string(index=False))

Interpret the columns as follows:

- ``r2D`` is the ratio of aggregate unbiased ``D2`` and ``Pi2`` moments. It is
  the statistic converted to recent :math:`N_e`.
- ``rDz`` should be interpreted as a model diagnostic. A confidence interval
  separated from zero can indicate structure, gene flow, or another departure
  from the simple unlinked random-mating model.
- In the generated figures, an orange diamond marks a population whose 95%
  bootstrap ``rDz`` interval excludes zero. Under the simple unlinked model,
  ``rDz`` is expected near zero, so this persistent non-zero signal warns that
  recent migration, population structure, or another model departure may make
  the corresponding LD-based ``Ne`` estimate less reliable. It is a caution
  about model fit, not a failed calculation or a diagnosis of the cause.
- ``Ne`` is finite only when the aggregate ``r2D`` estimate is positive.
- When ``r2D`` is nonpositive, the MultiQC summary labels ``Ne`` as not
  estimable and omits that population from the ``Ne`` bar plot; the scientific
  result remains missing rather than being coerced to a negative value.
- An infinite ``Ne_CI_Upper`` means the ``r2D`` interval reaches or crosses
  zero; it should not be replaced with an arbitrary finite value.
- ``Pairs`` is the number of complete, eligible unlinked pairs contributing to
  the population estimate.

Inspect reproducibility metadata
--------------------------------

.. code-block:: python

   for key in (
       "method",
       "group_source",
       "bootstrap_method",
       "candidate_pairs",
       "max_pairs",
       "n_jobs",
       "mating_system",
       "seed",
   ):
       print(f"{key}: {ld.metadata[key]}")

The metadata records whether VCF groups, explicit groups, or
``assume_unlinked=True`` defined eligible pairs. Preserve the generated JSON
metadata file with reported results.

Generated outputs
-----------------

Tables are written beneath::

   snpio_ld_tutorial_output/reports/linkage_disequilibrium/

and plots beneath::

   snpio_ld_tutorial_output/plots/linkage_disequilibrium/

The standard figures summarize population LD, recent :math:`N_e`, and sampled
pairwise distributions. With ``save_plots=True``, the analysis also queues a
MultiQC population table; bar, heatmap, and categorical line panels for LD and
finite :math:`N_e`; population-grouped LD bootstrap boxes; and a separate
population-grouped :math:`N_e` bootstrap boxplot.

The boxplots preserve every finite ``Population`` x ``Replicate`` observation.
The LD panel groups ``r2D``, ``rDz``, ``D``, ``Dz``, and ``Pi2`` by statistic,
whereas :math:`N_e` is separate because its scale is not directly comparable.
Replicates with nonpositive or unavailable ``r2D`` do not contribute a finite
:math:`N_e` boxplot point, but the corresponding population and
derived ``Ne_Status`` remain in the MultiQC summary table. In the returned
``ld.summary`` DataFrame, the same non-estimable value remains ``NaN``.

Coordinate-free or pre-pruned input
-----------------------------------

For PHYLIP, STRUCTURE, or GENEPOP data, provide a group label for every locus
when that information exists:

.. code-block:: python

   linkage_groups = ["chr1", "chr1", "chr2", "chr2", "chr3", "chr3"]

   ld = statistics.calculate_linkage_disequilibrium(
       locus_groups=linkage_groups,
       bootstrap_groups=linkage_groups,
       n_bootstraps=200,
       n_jobs=4,
       seed=42,
   )

Only if the data have already been pruned to independently segregating loci
and group labels are unavailable should you assert:

.. code-block:: python

   ld = statistics.calculate_linkage_disequilibrium(
       assume_unlinked=True,
       seed=42,
   )

``assume_unlinked=True`` is not a pruning operation or a statistical test of
linkage.

Sensitivity checks before reporting
-----------------------------------

For a production analysis:

1. Rerun with a larger ``max_pairs`` or several fixed seeds and confirm that
   the population estimates are not driven by pair subsampling.
2. Report the sample size, number of biallelic loci, number of eligible pairs,
   grouping source, mating model, seed, and confidence interval.
3. Inspect ``rDz`` and avoid interpreting a structured pooled population with
   a single-population :math:`N_e` model.
4. Treat wide or unbounded intervals as uncertainty, especially in small
   samples or populations with low diversity.
5. Document SNP ascertainment and any filtering used before the analysis.

Validation and citation
-----------------------

SNPio's executable validation infrastructure is under ``snpio/validation/``;
the LD-specific modules are ``linkage_disequilibrium.py``,
``ld_simulation.py``, and ``ld_plots.py``. The complete protocol and curated
PNG evidence are described in :doc:`ld_validation`.

When publishing results, cite both SNPio :cite:p:`MartinEtAl2026` and the
unphased LD method :cite:p:`RagsdaleGravel2020`. See :doc:`publication` for the
full SNPio citation, BibTeX, OSF reproducibility record, and distribution
links.
