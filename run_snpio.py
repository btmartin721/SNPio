import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from snpio import GenotypeData, NRemover2, Plotting


def main():
    # Read the alignment, popmap, and tree files
    gd = GenotypeData(
        filename="example_data/vcf_files/phylogen_subset14K.vcf",
        popmapfile="example_data/popmaps/phylogen_nomx.popmap",
        force_popmap=True,
        filetype="auto",
        qmatrix_iqtree="example_data/trees/test.qmat",
        siterates_iqtree="example_data/trees/test.rate",
        guidetree="example_data/trees/test.tre",
        chunk_size=5000,
    )

    # Make missingness report plots.
    gd.missingness_reports(file_prefix="unfiltered")

    # Run a PCA and make a scatterplot on the unfiltered data.
    Plotting.run_pca(gd, file_prefix="unfiltered")

    # Run the NRemover class to filter out missing data.
    nrm = NRemover2(gd)
    gd_filtered = nrm.nremover(
        max_missing_global=0.5,
        max_missing_pop=0.5,
        max_missing_sample=0.8,
        singletons=True,
        biallelic=True,
        unlinked_only=True,
        monomorphic=True,
        min_maf=0.01,
        search_thresholds=True,
    )

    # Make another missingness report plot for filtered data.
    gd_filtered.missingness_reports(file_prefix="filtered")

    # Run a PCA on the filtered data and make a scatterplot.
    Plotting.run_pca(gd_filtered, file_prefix="filtered")

    gd_filtered.write_vcf("example_data/vcf_files/nremover_test.vcf")
    gd_filtered.write_phylip("example_data/phylip_files/nremover_test.phy")
    gd_filtered.write_structure(
        "example_data/structure_files/nremover_test.str"
    )

    # print(gd_filtered.alignment)
    # print(gd_filtered.tree)

    # Locus-wise summary stats 
    hsht = gd_filtered.locstats()
    print(hsht)

    # compute Gst over range and save all as dataframe 
    mac_range = range(21)  # MAC from 0 to 20
    miss_prop_range = np.arange(0.0, 1.0, 0.05)  # Missing proportion from 0.0 to 0.95
    gst_results = gst_range(hsht, mac_range, miss_prop_range)
    print(gst_results)

    # heatmap plot of Gst over param windows
    plot_gst_heatmap(gst_results)


def gst_range(hsht, mac_range, miss_prop_range):
    results = []

    for mac_threshold in mac_range:
        for miss_prop_threshold in miss_prop_range:
            # Filter data according to current thresholds
            filtered_data = hsht[(hsht['MAC'] >= mac_threshold) & (hsht['MissProp'] <= miss_prop_threshold)]
            
            if len(filtered_data) > 1:
                Ht_global = filtered_data['Ht_corrected'].mean()
                Hs_global = filtered_data['Hs_corrected'].mean()
                
                # Compute Hedrick's Gst
                if Ht_global > 0 and (2.0 * Ht_global - Hs_global) > 0:
                    Ghedrick = ((2.0 * (Ht_global - Hs_global)) /
                                ((2.0 * Ht_global - Hs_global) * (1.0 - Hs_global)))
                else:
                    Ghedrick = np.nan
            else:
                Ghedrick = np.nan
            
            results.append({
                'MAC Threshold': mac_threshold,
                'Miss Prop Threshold': miss_prop_threshold,
                'Gst_prime': Ghedrick
            })

    results_df = pd.DataFrame(results)
    return results_df


def plot_gst_heatmap(gst_df):
    pivot_table = gst_df.pivot(index='Miss Prop Threshold', columns='MAC Threshold', values='Gst_prime')
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pivot_table, annot=False, cmap="viridis", cbar_kws={'label': 'Gst_prime'})
    ax.set_title('Heatmap of Gst by MAC and Missing Proportion')
    ax.set_xlabel('Minor Allele Count Threshold')
    ax.set_ylabel('Missing Proportion Threshold')
    ax.set_yticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_yticklabels()])
    plt.show()

if __name__ == "__main__":
    main()
