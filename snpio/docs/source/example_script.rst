Example Script
==============

The file ``run_snpio.py`` is a ready-to-use template that walks through a
complete SNPio workflow:

* Load genotype data (VCF, Structure, Phylip, or Genepop + popmap)
* Filter and clean the data
* Run desired population-genetic analyses
* Generate plots and an interactive MultiQC report

Run it using the `snpio` terminal command once SNPio and its
dependencies are installed:

.. code-block:: console

   $ snpio \
      --input   data/example.vcf.gz \
      --popmap  data/popmap.txt \
      --prefix  results/example_run

Command-line options
--------------------

``run_snpio.py`` uses ``argparse`` for a clean CLI:

.. list-table::
   :header-rows: 1
   :widths: 15 50

   * - Option
     - Description
   * - ``--input``
     - Path to a VCF / PHYLIP / STRUCTURE file.
   * - ``--popmap``
     - Two-column file mapping *sample ID → population*.
   * - ``--prefix``
     - Output prefix; results go to ``<prefix>_output/``.
   * - ``--plot-format``
     - Plot format: ``png`` | ``pdf`` | ``svg`` | ``jpeg`` (default: ``png``).
   * - ``--verbose``
     - Verbose logging.
   * - ``--debug``
     - Extra diagnostics for troubleshooting.

Outputs
-------

The script produces:

* **Filtered genotype data** (HDF5 + flat files)  
* **Population-genetic statistics** (CSV / JSON)  
* **Plots** (summary, PCA, distances, outliers, D-stats)  
* **Interactive MultiQC report** in  
  ``<prefix>_output/multiqc/``

Source code
-----------

Below is the full example script bundled with SNPio.  Feel free to copy,
adapt, or mine individual steps for your own pipelines.

.. code-block:: python
   :linenos:

   import argparse
   import sys
   from pathlib import Path

   from snpio import (
      NRemover2,
      PopGenStatistics,
      SNPioMultiQC,
      VCFReader,
   )


   # ---------------------------------------------------------------------
   # Helpers
   # ---------------------------------------------------------------------
   def _validate(path: str, label: str) -> None:
      p = Path(path)
      if not (p.is_file() and p.exists()):
         print(f"[ERROR] {label} file not found: {path}", file=sys.stderr)
         raise FileNotFoundError(path)


   def _parse_args():
      p = argparse.ArgumentParser(
         description="Run an end-to-end SNPio example analysis."
      )
      p.add_argument("--input",   required=True,
                     help="VCF / PHYLIP / STRUCTURE file")
      p.add_argument("--popmap",  required=True,
                     help="Two-column sample→population file")
      p.add_argument("--prefix",  required=True,
                     help="Output prefix (results in <prefix>_output/)")
      p.add_argument("--plot-format", default="png",
                     choices=["png", "pdf", "svg"],
                     help="Plot format (default: png)")
      p.add_argument("--verbose", action="store_true",
                     help="Verbose logging")
      p.add_argument("--debug",   action="store_true",
                     help="Debug mode")
      return p.parse_args()


   # ---------------------------------------------------------------------
   # Main workflow
   # ---------------------------------------------------------------------
   def main():
      args = _parse_args()

      # Validate input paths
      _validate(args.input,  "Input")
      _validate(args.popmap, "Popmap")

      # -----------------------------------------------------------------
      # Load data
      # -----------------------------------------------------------------
      gd = VCFReader(
         filename=args.input,
         popmapfile=args.popmap,
         force_popmap=True,
         prefix=args.prefix,
         plot_format=args.plot_format,
         verbose=args.verbose,
         debug=args.debug,
         chunk_size=5_000,
         include_pops=["EA", "GU", "TT", "ON"],
      )

      gd.missingness_reports(prefix=args.prefix)

      # -----------------------------------------------------------------
      # Filtering
      # -----------------------------------------------------------------
      nrm = NRemover2(gd)
      nrm.search_thresholds(
         thresholds=[0.25, 0.5, 0.75],
         maf_thresholds=[0.01, 0.03],
         mac_thresholds=[2, 3],
         filter_order=[
            "filter_biallelic",
            "filter_missing",
            "filter_missing_pop",
            "filter_singletons",
            "filter_monomorphic",
            "filter_maf",
            "filter_mac",
            "filter_missing_sample",
         ],
      )

      gd_filt = (
         nrm.filter_biallelic(exclude_heterozygous=True)
            .filter_missing(0.75)
            .filter_missing_pop(0.75)
            .filter_singletons(exclude_heterozygous=True)
            .filter_missing_sample(0.80)
            .resolve()
      )

      nrm.plot_sankey_filtering_report()
      gd_filt.missingness_reports(gd_filt.prefix)

      # -----------------------------------------------------------------
      # Population-genetic analyses
      # -----------------------------------------------------------------
      pgs = PopGenStatistics(gd_filt,
                           verbose=args.verbose,
                           debug=args.debug)

      pgs.summary_statistics(n_permutations=100,
                           n_jobs=8,
                           use_pvalues=True)

      pgs.neis_genetic_distance(n_permutations=1_000,
                              n_jobs=8,
                              use_pvalues=True)

      pgs.detect_fst_outliers(n_permutations=100,
                              correction_method="fdr_bh",
                              use_dbscan=False,
                              n_jobs=8,
                              min_samples=5)

      pgs.detect_fst_outliers(n_permutations=1_000,
                              correction_method="fdr_bh",
                              use_dbscan=True,
                              n_jobs=8,
                              min_samples=5)

      # D-statistics (Patterson, Partitioned, D-FOIL)
      inds = gd_filt.popmap_inverse
      pgs.calculate_d_statistics(
         method="patterson",
         population1="EA", population2="GU",
         population3="TT", outgroup="ON",
         individual_selection={k: inds[k] for k in ["EA", "GU", "TT", "ON"]},
         max_individuals_per_pop=5,
         n_jobs=1,
         num_bootstraps=1_000,
      )

      pgs.calculate_d_statistics(
         method="partitioned",
         population1="EA", population2="GU",
         population3="TT", population4="ON",
         outgroup="OG",
         individual_selection=inds,
         max_individuals_per_pop=5,
         n_jobs=1,
         num_bootstraps=1_000,
      )

      pgs.calculate_d_statistics(
         method="dfoil",
         population1="EA", population2="GU",
         population3="TT", population4="ON",
         outgroup="OG",
         individual_selection=inds,
         max_individuals_per_pop=5,
         n_jobs=1,
         num_bootstraps=1_000,
      )

      # PCA
      pgs.pca()

      # -----------------------------------------------------------------
      # Build MultiQC report
      # -----------------------------------------------------------------
      print("📊 Building MultiQC report …")
      SNPioMultiQC.build(
         prefix="Example Report",
         output_dir=f"{args.prefix}_output/multiqc",
         overwrite=True,
      )


   if __name__ == "__main__":
         main()

.. tip::

   For large analyses, adjust the ``chunk_size`` and ``n_jobs`` parameters to match your compute resources, and consider running the filtering and statistics steps in a workflow manager (e.g. Snakemake or Nextflow).