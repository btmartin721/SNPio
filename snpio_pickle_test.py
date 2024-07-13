import pickle
import multiprocessing
from snpio import GenotypeData, NRemover2, Plotting


def main():
    # Initialize the GenotypeData object
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
    print(gd.alignment)

    # Test pickling and unpickling
    test_pickling(gd)

    # test parallel process
    parallel_processing_test(gd)


def process_data(gd):
    print(f"Processing data in process with id: {multiprocessing.current_process().pid}")
    return gd.num_snps


def parallel_processing_test(gd):
    with multiprocessing.Pool(2) as pool:
        results = pool.map(process_data, [gd, gd])
    print("Results:", results)


def test_pickling(gd):
    # Serialize the object to a byte stream
    pickled_data = pickle.dumps(gd)

    # Deserialize the byte stream to an object
    unpickled_gd = pickle.loads(pickled_data)

    # Optionally, test if the original and unpickled object have the same attributes
    assert gd.filename == unpickled_gd.filename, "Filename attribute mismatch"
    # Add more assertions as needed for critical attributes

    print("Pickling and unpickling successful!")


if __name__ == '__main__':
    main()
