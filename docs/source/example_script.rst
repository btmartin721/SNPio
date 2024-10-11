Example script
===============

The ``run_snpio.py`` script provides a template you can use to get started.

Just type:

.. code-block:: shell

   python3 run_snpio.py

and it will run the example data.

Below is the code for the script:

.. code-block:: python

   from snpio import NRemover2, Plotting, VCFReader

   def main():
      # Read the alignment, popmap, and tree files.
      vcf = "example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz"
      pm = "example_data/popmaps/phylogen_nomx.popmap"
      cs = 5000
      fp = True
      gd = VCFReader(filename=vcf, popmapfile=pm, force_popmap=fp, chunk_size=cs)

      # Run PCA and make missingness report plots.
      plotting = Plotting(genotype_data=gd)

      # Run a Principal Component Analysis (PCA) on the data.
      gd_components, gd_pca = plotting.run_pca()

      # Generate missingness plots and reports.
      gd.missingness_reports()

      # Initialize NRemover2.
      nrm = NRemover2(gd)

      # Search and plot removed and kept loci and samples for various 
      # thresholds.
      nrm.search_thresholds(
         thresholds=[0.25, 0.5, 0.75, 1.0],
         maf_thresholds=[0.0, 0.01, 0.025, 0.05],
         mac_thresholds=[2, 5],
      )

      # Run filtering.
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

      # Generate a Sankey Diagram plot that shows the filtering process.
      nrm.plot_sankey_filtering_report()

      # Run PCA on the filtered data.
      plotting2 = Plotting(genotype_data=gd_filt)
      filt_components, filt_pca = plotting2.run_pca()
      
      # Make missingness plots again after filtering process.
      gd_filt.missingness_reports(prefix="filtered")

      # Write the filtered VCF file to disk.
      gd_filt.write_vcf("example_data/vcf_files/nremover_test.vcf")

   if __name__ == "__main__":
      main()

