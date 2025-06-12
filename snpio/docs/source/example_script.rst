Example script
===============

The ``run_snpio.py`` script provides a template you can use to get started. It demonstrates how to use the main classes in the package. The script reads a VCF file, a popmap file, and a tree file, and then runs various analyses on the data. The script also demonstrates how to encode genotypes into different formats, filter the data, and parse a tree file. The script is located in the ``snpio`` directory. To run the script, run the following command from the project root directory, assuming you have installed the package and its dependencies:

.. code-block:: shell

   python3 snpio/run_snpio.py

It will then run the example data.

Below is the code for the example script:

.. code-block:: python

   import pandas as pd

   from snpio import (
      GenotypeEncoder,
      NRemover2,
      Plotting,
      PopGenStatistics,
      TreeParser,
      VCFReader,
      GenePopReader,
   )

   def main():
      # Read the alignment, popmap, and tree files.
      gd = VCFReader(
         filename="snpio/example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz",
         popmapfile="snpio/example_data/popmaps/phylogen_nomx.popmap",
         force_popmap=True,  # Remove samples not in the popmap, or vice versa.
         chunk_size=5000,
         exclude_pops=["OG"],
         plot_format="pdf",
      )

      # Run PCA and make missingness report plots.
      plotting = Plotting(genotype_data=gd)
      gd_components, gd_pca = plotting.run_pca()
      gd.missingness_reports()

      nrm = NRemover2(gd)

      # Run NRemover2 across multiple thresholds to optimize filtering.
      nrm.search_thresholds(
         thresholds=[0.25, 0.5, 0.75, 1.0],
         maf_thresholds=[0.0],
         mac_thresholds=[2, 5],
      )

      # Filter the genotype data using NRemover2 with a series of filters.
      # The filters are applied in the order they are called.
      gd_filt = (
         nrm.filter_missing_sample(0.75)
         .filter_missing(0.75)
         .filter_missing_pop(0.75)
         .filter_mac(2)
         .filter_monomorphic(exclude_heterozygous=False)
         .filter_singletons(exclude_heterozygous=False)
         .filter_biallelic(exclude_heterozygous=False)
         .resolve()
      )

      # Make a Sankey plot of the filtering process.
      # This will show how many samples and variants were kept and removed at each step.
      nrm.plot_sankey_filtering_report()

      # # Make missingness report plots again after filtering.
      # This will show the missingness of samples and variants after filtering.
      plotting2 = Plotting(genotype_data=gd_filt)
      filt_components, filt_pca = plotting2.run_pca()
      gd_filt.missingness_reports(prefix="filtered")

      # Write the filtered data to a new VCF file.
      gd_filt.write_vcf("snpio/example_data/vcf_files/nremover_test.vcf")

      # Initialize the PopGenStatistics class with the genotype data object.
      pgs = PopGenStatistics(gd_filt, verbose=True)

      # Estimates observed and expected heterozygosity, nucleotide diversity, and Weir and Cockerham's Fst.
      summary_stats = pgs.summary_statistics(       
         n_permutations=100,
         n_jobs=1,
         save_plots=True,
         use_pvalues=True
      )

      # Detect Fst outliers using different methods.
      df_fst_outliers_boot, df_fst_outlier_pvalues_boot = pgs.detect_fst_outliers(
            correction_method="fdr",
            alpha=0.05,
            use_dbscan=False,
            n_permutations=1000,
            n_jobs=1,
            alternative="upper",
            seed=42,
            kde_bandwidth="scott"
      )

      # Detect Fst outliers using DBSCAN.
      # This method is much faster but may not be as accurate as the bootstrapping method.
      df_fst_outliers_dbscan, df_fst_outlier_pvalues_dbscan = pgs.detect_fst_outliers(
         correction_method="fdr",
         alpha=0.05,    
         use_dbscan=True,
         n_jobs=1,
         seed=42,
         alternative="upper",
      )

      # Calculate Nei's genetic distance and p-values.
      # This will return two dataframes: one with the distances and one with the p-values.
      nei_dist_df, nei_pvals_df = pgs.neis_genetic_distance(
         n_permutations=100,
         n_jobs=1,
         use_pvalues=True,
         palette="magma",
         supress_plot=False,
      )

      # Calculate D statistics using Patterson's D statistic method.
      # This will return a dataframe with the D statistics and a dictionary with the overall results.
      dstats_df, overall_results = pgs.calculate_d_statistics(
         method="patterson",
         population1="EA",
         population2="GU",
         population3="TT",
         outgroup="ON",
         num_bootstraps=10,
         n_jobs=1,
         max_individuals_per_pop=6,
      )

      # # Encode the genotypes into 012, one-hot, and integer formats.
      ge = GenotypeEncoder(gd_filt)
      gt_012 = ge.genotypes_012
      gt_onehot = ge.genotypes_onehot
      gt_int = ge.genotypes_int

      df012 = pd.DataFrame(gt_012)
      dfint = pd.DataFrame(gt_int)

      tp = TreeParser(
         genotype_data=gd_filt,
         treefile="snpio/example_data/trees/test.tre",
         qmatrix="snpio/example_data/trees/test.iqtree",
         siterates="snpio/example_data/trees/test14K.rate",
         verbose=True,
         debug=False,
      )

      # # Get a toytree object by reading the tree file.
      tree = tp.read_tree()

      # # Reroot the tree at any nodes containing the string 'EA' in the sampleID.
      tp.reroot_tree("~EA")

      # # Get a subtree with only the samples containing 'EA' in the sampleID.
      subtree = tp.get_subtree("~EA")

      # # Prune the tree to remove samples containing 'ON' in the sampleID.
      pruned_tree = tp.prune_tree("~ON")


   if __name__ == "__main__":
      main()



