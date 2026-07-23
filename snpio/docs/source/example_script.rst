Command-Line Workflow
=====================

The installed ``snpio`` command runs the opinionated end-to-end workflow in
``snpio.run_snpio``. It loads a VCF and population map, records the arguments,
generates pre-filtering diagnostics, applies ``NRemover2``, runs population-
genetic analyses on the filtered object, and builds a MultiQC report.

Use the Python API when you need a different analysis subset, explicit LD
linkage groups, a non-VCF reader, or custom filter ordering. The command-line
workflow is intended for reproducible full runs rather than as a replacement
for every API option.

Basic invocation
----------------

.. code-block:: console

   snpio \
     --input data/example.vcf.gz \
     --popmap data/popmap.txt \
     --prefix results/example_run \
     --include-pops EA GU TT ON OG \
     --n-jobs 8 \
     --plot-format png \
     --random-seed 42

``--prefix results/example_run`` creates
``results/example_run_output/``. Although the parser accepts an omitted
``--popmap``, the current full population-genetic workflow validates and uses
one, so provide it for command-line runs.

Inspect the installed command for the exact options and defaults in your
version:

.. code-block:: console

   snpio --help
   snpio --version

Input and filtering options
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Option
     - Default
     - Meaning
   * - ``--input PATH``
     - required
     - Input VCF for the current command-line workflow.
   * - ``--popmap PATH``
     - none
     - Two-column sample-to-population map used by population analyses.
   * - ``--prefix PREFIX``
     - required
     - Output prefix; artifacts are rooted at ``<prefix>_output/``.
   * - ``--include-pops POP [POP ...]``
     - all
     - Restrict loading and downstream analyses to named populations.
   * - ``--sample-missing-threshold FLOAT``
     - ``0.8``
     - Maximum sample missingness retained by the filtering workflow.
   * - ``--locus-missing-threshold FLOAT``
     - ``0.75``
     - Maximum overall locus missingness retained.
   * - ``--locus-missing-pop-threshold FLOAT``
     - ``0.75``
     - Within-population locus threshold recorded in run provenance. The
       current bundled workflow applies ``--locus-missing-threshold`` to its
       population-missingness filter as well; use the Python API when the two
       thresholds must differ.
   * - ``--maf-threshold FLOAT``
     - ``0.01``
     - Minor-allele-frequency threshold recorded for the run.
   * - ``--mac-threshold INTEGER``
     - ``2``
     - Minor-allele-count threshold recorded for the run.
   * - ``--exclude-heterozygous``
     - false
     - Exclude heterozygous calls in filters that support this policy.
   * - ``--force-popmap``
     - false
     - Reconcile minor VCF/popmap sample mismatches when loading.
   * - ``--chunk-size INTEGER``
     - ``5000``
     - VCF chunk size; tune this to available memory.

Parallelism, plots, and resampling
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Option
     - Default
     - Meaning
   * - ``--n-jobs INTEGER``
     - ``1``
     - Parallel workers for analyses that support them; ``-1`` uses all CPUs.
   * - ``--plot-format {png,pdf,svg}``
     - ``png``
     - Static plot format.
   * - ``--n-boot-fst INTEGER``
     - ``1000``
     - Fst bootstrap replicates.
   * - ``--n-perm-fst INTEGER``
     - ``1000``
     - Fst permutations, including outlier analysis.
   * - ``--n-boot-neis INTEGER``
     - ``1000``
     - Nei-distance bootstrap replicates.
   * - ``--n-perm-neis INTEGER``
     - ``1000``
     - Nei-distance permutations.
   * - ``--n-boot-dstats INTEGER``
     - ``1000``
     - Bootstrap replicates for each D-statistic method.
   * - ``--n-boot-ld INTEGER``
     - ``1000``
     - Grouped-locus bootstrap replicates for LD and recent :math:`N_e`.

LD and recent effective-size options
------------------------------------

The command runs ``calculate_linkage_disequilibrium`` on the filtered
``GenotypeData`` after the other analyses.

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Option
     - Default
     - Meaning
   * - ``--include-overall``
     - false
     - Add a pooled all-sample estimate to the per-population estimates.
   * - ``--assume-unlinked``
     - false
     - Explicitly assert that all supplied loci are unlinked.

For an ordinary multi-chromosome VCF, leave ``--assume-unlinked`` disabled.
SNPio infers chromosome or scaffold groups from VCF marker names and excludes
within-group pairs. Enabling the flag overrides those labels and emits a
warning. Use the Python API when explicit ``locus_groups`` or
``bootstrap_groups`` are required.

D-statistic options
-------------------

The workflow runs Patterson, partitioned, and DFOIL statistics. Population
labels are provided by ``--population1`` through ``--population4`` and
``--outgroup``; their defaults are ``EA``, ``GU``, ``TT``, ``ON``, and ``OG``.

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Option
     - Default
     - Meaning
   * - ``--individual-selection {random,least_missing,all}``
     - ``random``
     - Choose a random capped subset, the deterministically least-missing
       samples, or all samples in each population.
   * - ``--max-individuals-per-pop INTEGER``
     - ``5``
     - Per-population cap for ``random`` and ``least_missing``. ``all`` ignores
       this cap.

``least_missing`` counts unusable calls in the filtered D-statistic 0/1/2
matrix, ranks within populations, and resolves ties in alignment order. It is
deterministic; ``--random-seed`` controls random selection and resampling but
does not change this ranking.

Fst outlier and run-control options
-----------------------------------

``--use-dbscan`` selects DBSCAN for Fst outlier detection;
``--min-samples-dbscan`` controls its neighborhood size. Permutation-based
multiple testing uses ``--pvalue-correction-method`` with one of
``bonferroni``, ``fdr_bh``, ``holm``, ``hochberg``, ``hommel``, or
``fdr_tsbh``.

``--random-seed`` records and applies a reproducibility seed where supported.
``--overwrite-multiqc`` permits replacement of an existing report.
``--verbose`` and ``--debug`` increase logging detail.

Generated outputs
-----------------

The command uses the shared output layout:

.. code-block:: text

   <prefix>_output/
   ├── data/
   │   ├── vcf/vcf_attributes.h5
   │   └── vcf/nremover/vcf_attributes_filtered_<state>.h5
   ├── logs/
   │   ├── arguments.json
   │   └── <module>.log
   ├── multiqc/
   ├── plots/
   │   ├── <operation>/
   │   └── nremover/<operation>/
   └── reports/
       ├── <operation>/
       └── nremover/<operation>/

All analyses initialized with the resolved filtered object write beneath the
``nremover`` scope. LD reports, for example, use
``reports/nremover/linkage_disequilibrium/`` and static figures use
``plots/nremover/linkage_disequilibrium/``. MultiQC includes the filtered-data
summaries plus population-aware LD and separate recent-:math:`N_e` panels.

The JSON provenance file contains the parsed input, filtering, parallelism,
resampling, LD, D-statistic, plotting, and verbosity settings. Preserve it with
the result bundle when sharing or publishing a run.

Equivalent module invocation
----------------------------

The console entry point calls ``snpio.run_snpio:main``. These invocations are
equivalent after installation:

.. code-block:: console

   snpio --input data/example.vcf.gz --popmap data/popmap.txt --prefix run

   python -m snpio.run_snpio \
     --input data/example.vcf.gz \
     --popmap data/popmap.txt \
     --prefix run

For a modular Python workflow, begin with :doc:`getting_started`; for the full
LD API and assumptions, see :doc:`linkage_disequilibrium`; and for the
executable LD evidence suite, see :doc:`ld_validation`.
