import csv
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from snpio import NRemover2, VCFReader
from snpio.utils.custom_exceptions import AlignmentFormatError


class TestNRemover2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_vcf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".vcf")
        cls.tmp_popmap_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".popmap"
        )
        cls.tmp_output_vcf_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".vcf"
        )
        cls.tmp_output_popmap_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".popmap"
        )
        cls.test_popmap_content = ["Sample1\tpop1", "Sample2\tpop1", "Sample3\tpop2"]

        with open(cls.tmp_vcf_file.name, "w", newline="") as vcf:
            writer = csv.writer(vcf, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["##fileformat=VCFv4.2"])
            writer.writerow(
                ["##FORMAT=<ID=GT,Number=1,Type=String,Description='Genotype'>"]
            )
            writer.writerow(
                ["##FORMAT=<ID=AD,Number=R,Type=Integer,Description='Allele depths'>"]
            )
            writer.writerow(
                [
                    "#CHROM",
                    "POS",
                    "ID",
                    "REF",
                    "ALT",
                    "QUAL",
                    "FILTER",
                    "INFO",
                    "FORMAT",
                    "Sample1",
                    "Sample2",
                    "Sample3",
                ]
            )
            writer.writerow(
                [
                    "NW123.1",
                    100,
                    ".",
                    "A",
                    "T",
                    ".",
                    "PASS",
                    ".",
                    "GT:AD",
                    "0/0:10,0",
                    "0/1:5,5",
                    "1/1:0,10",
                ]
            )
            writer.writerow(
                [
                    "NW123.1",
                    200,
                    ".",
                    "G",
                    "A",
                    ".",
                    "PASS",
                    ".",
                    "GT:AD",
                    "0/1:3,2",
                    "./.:.",
                    "0/1:4,4",
                ]
            )
            writer.writerow(
                [
                    "XM123.1",
                    100,
                    ".",
                    "C",
                    ".",
                    ".",
                    "PASS",
                    ".",
                    "GT:AD",
                    "0/0:8",
                    "0/0:9",
                    "0/0:10",
                ]
            )
            writer.writerow(
                [
                    "XM123.1",
                    201,
                    ".",
                    "T",
                    "G",
                    ".",
                    "PASS",
                    ".",
                    "GT:AD",
                    "1/1:0,9",
                    "1/1:0,8",
                    "0/1:4,4",
                ]
            )
            writer.writerow(
                [
                    "XM123.1",
                    305,
                    ".",
                    "C",
                    "G,A",
                    ".",
                    "PASS",
                    ".",
                    "GT:AD",
                    "0/2:5,0,5",
                    "0/1:3,3,0",
                    "./.:.",
                ]
            )
            writer.writerow(
                [
                    "XM123.1",
                    400,
                    ".",
                    "A",
                    "C",
                    ".",
                    "PASS",
                    ".",
                    "GT:AD",
                    "0/0:10,0",
                    "0/0:5,5",
                    "0/1:0,10",
                ]
            )

        with open(cls.tmp_popmap_file.name, "w") as popmap:
            for line in cls.test_popmap_content:
                popmap.write(line + "\n")

    def setUp(self):
        self.vcf_reader = VCFReader(
            filename=self.tmp_vcf_file.name,
            popmapfile=self.tmp_popmap_file.name,
            chunk_size=100,
            prefix="test_read_vcf",
            verbose=False,
            store_format_fields=True,
        )
        self.nrm = NRemover2(self.vcf_reader)

    def test_filter_missing_sample(self):
        filtered_data = self.nrm.filter_missing_sample(0.15).resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.sample_indices) if keep
        ]
        expected_indices = [0]  # Only Sample1 has enough data
        self.assertEqual(retained_indices, expected_indices)

    def test_filter_missing(self):
        filtered_data = self.nrm.filter_missing(0.3).resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]
        expected_indices = [0, 2, 3, 5]  # Loci with less than 30% missing data
        self.assertEqual(retained_indices, expected_indices)

    def test_filter_mac(self):
        filtered_data = self.nrm.filter_mac(2, exclude_heterozygous=True).resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]
        self.assertEqual(retained_indices, [0])

    def test_filter_monomorphic(self):
        filtered_data = self.nrm.filter_monomorphic().resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]

        expected_indices = [0, 1, 3, 4, 5]
        self.assertEqual(retained_indices, expected_indices)

    def test_filter_singletons(self):
        filtered_data = self.nrm.filter_singletons().resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]

        # The third and fifth loci are filtered out (singletons)
        # The first, second, and fourth loci are retained
        expected_indices = [0, 1, 2, 4]
        self.assertEqual(retained_indices, expected_indices)

    def test_filter_biallelic(self):
        filtered_data = self.nrm.filter_biallelic().resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]
        self.assertEqual(retained_indices, [0, 1, 3, 5])

    def test_filter_linked(self):
        np.random.seed(0)
        filtered_data = self.nrm.filter_linked().resolve()
        self.assertEqual(filtered_data.num_snps, 2)
        self.assertEqual(np.count_nonzero(filtered_data.loci_indices), 2)

    def test_filter_allele_depth(self):
        # Test filter_allele_depth with minimum total AD = 26
        filtered_data = self.nrm.filter_allele_depth(min_total_depth=26).resolve()
        retained_indices = [
            i for i, keep in enumerate(filtered_data.loci_indices) if keep
        ]

        # Based on AD values in test VCF:
        # all loci have at least one sample with AD sum ≥ 26
        expected_indices = [0, 2, 5]
        self.assertEqual(retained_indices, expected_indices)

    def test_filtered_state_is_internally_aligned(self):
        """Filtering must refresh caches and every sample/locus metadata view."""

        source_hdf = Path(self.vcf_reader.vcf_attributes_fn)
        with h5py.File(source_hdf, "r") as h5:
            original_core = {
                key: h5[key][:]
                for key in (
                    "chrom",
                    "pos",
                    "id",
                    "ref",
                    "alt",
                    "qual",
                    "filt",
                    "fmt",
                )
            }
            original_ad = h5["fmt_metadata/AD"][:]

        # Populate every data-dependent cached_property before copying.
        for attr in (
            "missing_mask",
            "valid_mask",
            "het_mask",
            "per_locus_het_rate",
            "per_individual_het_rate",
            "is_missing_locus",
            "observed_iupac_per_locus",
            "biallelic_mask",
            "nbytes",
        ):
            getattr(self.vcf_reader, attr)
        self.vcf_reader.all_missing_idx = [1, 4]

        filtered = (
            self.nrm.filter_missing_sample(0.15)
            .random_subset_loci(3, seed=7)
            .resolve()
        )

        self.assertEqual(filtered.snp_data.shape, (1, 3))
        self.assertEqual(filtered.samples, ["Sample1"])
        self.assertEqual(filtered.populations, ["pop1"])
        self.assertEqual(filtered.popmap, {"Sample1": "pop1"})
        self.assertEqual(filtered.popmap_inverse, {"pop1": ["Sample1"]})
        self.assertEqual(filtered.num_pops, 1)
        self.assertEqual(filtered.pop_state.num_pops, 1)
        self.assertEqual(filtered.all_missing_idx, [])
        self.assertTrue(filtered.was_filtered)
        self.assertEqual(filtered.num_records, 3)

        self.assertEqual(filtered.missing_mask.shape, (1, 3))
        self.assertEqual(filtered.valid_mask.shape, (1, 3))
        self.assertEqual(filtered.het_mask.shape, (1, 3))
        self.assertEqual(filtered.per_locus_het_rate.shape, (3,))
        self.assertEqual(filtered.per_individual_het_rate.shape, (1,))
        self.assertEqual(filtered.is_missing_locus.shape, (3,))
        self.assertEqual(len(filtered.observed_iupac_per_locus), 3)
        self.assertEqual(filtered.biallelic_mask.shape, (3,))

        self.assertEqual(len(filtered.marker_names), 3)
        self.assertEqual(len(filtered.ref), 3)
        self.assertEqual(len(filtered.alt), 3)

        locus_mask = filtered.loci_indices
        sample_mask = filtered.sample_indices
        self.assertEqual(locus_mask.shape, (6,))
        self.assertEqual(sample_mask.shape, (3,))

        filtered_hdf = Path(filtered.vcf_attributes_fn)
        self.assertNotEqual(filtered_hdf, source_hdf)
        self.assertEqual(filtered_hdf.parent.name, "nremover")
        self.assertEqual(filtered_hdf.parent.parent.name, "vcf")
        with h5py.File(filtered_hdf, "r") as h5:
            for key, values in original_core.items():
                np.testing.assert_array_equal(h5[key][:], values[locus_mask])
            np.testing.assert_array_equal(
                h5["fmt_metadata/AD"][:],
                original_ad[locus_mask][:, sample_mask],
            )

    def test_repeated_and_branched_filtering_keep_independent_hdf5_state(self):
        """Derived VCF states must never overwrite their source or siblings."""

        source_hdf = Path(self.vcf_reader.vcf_attributes_fn)
        branch_a = NRemover2(self.vcf_reader).random_subset_loci(2, seed=1).resolve()
        branch_b = NRemover2(self.vcf_reader).random_subset_loci(3, seed=2).resolve()

        path_a = Path(branch_a.vcf_attributes_fn)
        path_b = Path(branch_b.vcf_attributes_fn)
        self.assertNotEqual(path_a, source_hdf)
        self.assertNotEqual(path_b, source_hdf)
        self.assertNotEqual(path_a, path_b)

        with h5py.File(source_hdf, "r") as h5:
            self.assertEqual(h5["chrom"].shape, (6,))
        with h5py.File(path_a, "r") as h5:
            self.assertEqual(h5["chrom"].shape, (2,))
        with h5py.File(path_b, "r") as h5:
            self.assertEqual(h5["chrom"].shape, (3,))

        repeated = NRemover2(branch_a).random_subset_loci(1, seed=3).resolve()
        repeated_path = Path(repeated.vcf_attributes_fn)
        self.assertNotEqual(repeated_path, path_a)
        self.assertTrue(repeated.was_filtered)
        self.assertEqual(repeated.num_records, 1)
        with h5py.File(path_a, "r") as h5:
            self.assertEqual(h5["chrom"].shape, (2,))
        with h5py.File(repeated_path, "r") as h5:
            self.assertEqual(h5["chrom"].shape, (1,))

        # A no-op pass over an already filtered object retains its history.
        no_op = NRemover2(repeated).random_subset_loci(1, seed=4).resolve()
        self.assertTrue(no_op.was_filtered)
        self.assertEqual(no_op.snp_data.shape, (3, 1))
        with h5py.File(no_op.vcf_attributes_fn, "r") as h5:
            self.assertEqual(h5["chrom"].shape, (1,))

    def test_set_alignment_rejects_inconsistent_mask_counts(self):
        """Invalid masks must fail before mutating the copied object."""

        copied = self.vcf_reader.copy()
        original = copied.snp_data.copy()
        bad_locus_mask = np.array([True, True, True, False, False, False])

        with self.assertRaisesRegex(
            AlignmentFormatError,
            "shape does not match the retained-mask counts",
        ):
            copied.set_alignment(
                snp_data=original[:, :2],
                samples=copied.samples,
                sample_indices=np.ones(copied.num_inds, dtype=bool),
                loci_indices=bad_locus_mask,
                reset_attributes=True,
            )

        np.testing.assert_array_equal(copied.snp_data, original)

    def test_failed_vcf_metadata_write_is_transactional(self):
        """A failed atomic replace must preserve state and remove its temp file."""

        copied = self.vcf_reader.copy()
        source_hdf = Path(copied.vcf_attributes_fn)
        original_data = copied.snp_data.copy()
        original_samples = copied.samples.copy()
        temp_pattern = ".vcf_attributes_filtered_*.tmp.h5"
        temp_files_before = set(source_hdf.parent.glob(temp_pattern))

        locus_mask = np.array([True, False, True, False, False, True])
        sample_mask = np.ones(copied.num_inds, dtype=bool)

        with patch("snpio.io.vcf_reader.os.replace", side_effect=OSError("boom")):
            with self.assertRaisesRegex(OSError, "boom"):
                copied.set_alignment(
                    snp_data=original_data[:, locus_mask],
                    samples=original_samples,
                    sample_indices=sample_mask,
                    loci_indices=locus_mask,
                    reset_attributes=True,
                )

        np.testing.assert_array_equal(copied.snp_data, original_data)
        self.assertEqual(copied.samples, original_samples)
        self.assertEqual(Path(copied.vcf_attributes_fn), source_hdf)
        self.assertEqual(
            set(source_hdf.parent.glob(temp_pattern)),
            temp_files_before,
        )

    @classmethod
    def tearDownClass(cls):
        for path in [
            cls.tmp_vcf_file.name,
            cls.tmp_popmap_file.name,
            cls.tmp_output_vcf_file.name,
            cls.tmp_output_popmap_file.name,
            cls.tmp_vcf_file.name + ".tbi",
        ]:
            Path(path).unlink(missing_ok=True)

        shutil.rmtree(Path("test_read_vcf_output"), ignore_errors=True)

        for pattern in ["tmp*.vcf.gz", "tmp*.vcf.gz.tbi", "tmp*.popmap"]:
            for f in Path(".").glob(pattern):
                f.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
