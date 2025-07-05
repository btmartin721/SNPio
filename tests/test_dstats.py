import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
from snpio import VCFReader
from snpio.popgenstats.d_statistics import DStatistics

logger = logging.getLogger("test")
logger.setLevel(logging.WARNING)


class TestDStatistics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.tmpdir = tempfile.TemporaryDirectory()

        # create a simple popmap with 4 samples per population
        self.popmap = Path(self.tmpdir.name) / "popmap.txt"
        self.num_per_pop = 4
        self.pops = {
            pop: [f"{pop}_{i}" for i in range(self.num_per_pop)]
            for pop in ("P1", "P2", "P3", "P4", "O")
        }
        self.samples = []
        with open(self.popmap, "w") as f:
            for pop, ids in self.pops.items():
                for sid in ids:
                    f.write(f"{sid}\t{pop}\n")
                    self.samples.append(sid)

    def tearDown(self):
        self.tmpdir.cleanup()

    def write_multi_vcf(self, records):
        header = (
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(self.samples)
            + "\n"
        )
        lines = [header]
        for rec in records:
            fields = []
            for sid in self.samples:
                gt = rec["genotypes"].get(sid, (None, None))
                fields.append("./." if None in gt else f"{gt[0]}/{gt[1]}")
            lines.append(
                f"{rec['chrom']}\t{rec['pos']}\t.\t{rec['ref']}\t{rec['alt']}"
                + "\t.\t.\t.\tGT\t"
                + "\t".join(fields)
                + "\n"
            )
        p = Path(self.tmpdir.name) / "test.vcf"
        p.write_text("".join(lines))
        return p

    def load_alignment(self, vcf_path):
        vcf = VCFReader(filename=str(vcf_path), popmapfile=str(self.popmap))
        return vcf.snp_data, vcf.samples

    def run_dstat(self, records, method, num_bootstraps=200):
        # load data and set up DStatistics
        alignment, samples = self.load_alignment(self.write_multi_vcf(records))
        idx = {s: i for i, s in enumerate(samples)}
        ds = DStatistics(alignment=alignment, sample_ids=samples, logger=logger)

        # select individuals
        d1 = [idx[s] for s in self.pops["P1"]]
        d2 = [idx[s] for s in self.pops["P2"]]
        d3 = [idx[s] for s in self.pops["P3"]]
        out = [idx[s] for s in self.pops["O"]]
        if method in ("partitioned", "dfoil"):
            d4 = [idx[s] for s in self.pops["P4"]]

        # encode and bootstrap indices
        geno_enc = ds._encode_alleles(alignment)
        n_snps = geno_enc.shape[1]
        rng = np.random.default_rng(42)
        snp_idx = rng.choice(n_snps, size=(num_bootstraps, n_snps), replace=True)

        # dispatch
        if method == "patterson":
            boots = ds._patterson_d_bootstrap(
                geno_enc,
                np.array(d1),
                np.array(d2),
                np.array(d3),
                np.array(out),
                snp_idx,
            )
            return ds._dstat_z_and_p(boots)

        elif method == "partitioned":
            boots = ds._partitioned_d_bootstrap(
                geno_enc,
                np.array(d1),
                np.array(d2),
                np.array(d3),
                np.array(d4),
                np.array(out),
                snp_idx,
            )
            return ds._dstat_z_and_p(boots)

        elif method == "dfoil":
            boots = ds._dfoil_bootstrap(
                geno_enc,
                np.array(d1),
                np.array(d2),
                np.array(d3),
                np.array(d4),
                np.array(out),
                snp_idx,
            )
            stats = ds._dfoil_z_and_p(boots)
            means, zs, ps = zip(*stats)
            return list(means), list(zs), list(ps)

        else:
            raise ValueError(f"Unknown method: {method}")

    # record generators
    def _make_record(self, pos, pattern):
        geno = {}
        P1, P2, P3, P4, O = (self.pops[k] for k in ("P1", "P2", "P3", "P4", "O"))
        for sid in O:
            geno[sid] = (0, 0)
        if pattern == "ABBA":
            for sid in P1 + P4:
                geno[sid] = (0, 0)
            for sid in P2 + P3:
                geno[sid] = (0, 1)
        elif pattern == "BABA":
            for sid in P2 + P4:
                geno[sid] = (0, 0)
            for sid in P1 + P3:
                geno[sid] = (0, 1)
        else:
            raise ValueError(pattern)
        return {"chrom": "1", "pos": pos, "ref": "A", "alt": "T", "genotypes": geno}

    def _make_null(self, pos):
        geno = {}
        for pop in ("P1", "P2", "P3", "P4", "O"):
            for sid in self.pops[pop]:
                r = np.random.rand()
                geno[sid] = (0, 1) if r < 0.2 else (1, 1) if r < 0.4 else (0, 0)
        return {"chrom": "1", "pos": pos, "ref": "A", "alt": "T", "genotypes": geno}

    def _make_dfoil_record_null(self, pos):
        """Generate a “null” D-FOIL record at the given position:
        randomly assigns each individual either homozygous ref (0/0),
        homozygous alt (1/1), or het (0/1) at rates 40%, 40%, 20%.
        """
        geno = {}
        # populations in order P1, P2, P3, P4, O
        for pop in ("P1", "P2", "P3", "P4", "O"):
            for sid in self.pops[pop]:
                r = np.random.rand()
                if r < 0.2:
                    geno[sid] = (0, 1)  # heterozygote
                elif r < 0.6:
                    geno[sid] = (0, 0)  # homozygous reference
                else:
                    geno[sid] = (1, 1)  # homozygous alternate

        return {
            "chrom": "1",
            "pos": pos,
            "ref": "A",
            "alt": "T",
            "genotypes": geno,
        }

    # actual tests
    def test_patterson_significant(self):
        recs = [self._make_record(1000 + i, "ABBA") for i in range(45)]
        recs += [self._make_record(2000 + i, "BABA") for i in range(5)]
        d_obs, z, p = self.run_dstat(recs, "patterson")
        self.assertGreater(abs(z), 2.0)
        self.assertLess(p, 0.05)

    def test_patterson_null(self):
        recs = [self._make_null(3000 + i) for i in range(1000)]
        d_obs, z, p = self.run_dstat(recs, "patterson")
        self.assertLess(abs(z), 3.0)
        self.assertGreater(p, 0.01)

    def test_partitioned_significant(self):
        recs = [self._make_record(5000 + i, "ABBA") for i in range(45)]
        recs += [self._make_record(6000 + i, "BABA") for i in range(5)]
        recs += [self._make_null(7000 + i) for i in range(5)]
        d_obs, z, p = self.run_dstat(recs, "partitioned")
        self.assertTrue(abs(z) > 2.0)
        self.assertLess(p, 0.05)

    def test_partitioned_null(self):
        recs = [self._make_null(8000 + i) for i in range(2000)]
        d_obs, z, p = self.run_dstat(recs, "partitioned")
        self.assertLess(abs(z), 3.0)
        self.assertGreater(p, 0.01)

    def test_dfoil_significant(self):
        # simple signal for DFO
        recs = [self._make_null(20000 + i) for i in range(200)]  # null baseline
        # spike a few ABBA-like patterns in P1/P2 to boost DFO
        recs += [self._make_record(30000 + i, "ABBA") for i in range(50)]
        means, zs, ps = self.run_dstat(recs, "dfoil")
        self.assertTrue(any(m > 0.1 for m in np.abs(means)))

    def test_dfoil_null(self):
        # Use the D-FOIL null record generator for a more realistic null distribution
        recs = [self._make_dfoil_record_null(40000 + i) for i in range(3000)]
        means, zs, ps = self.run_dstat(recs, "dfoil")
        for idx, (m, z, p) in enumerate(zip(means, zs, ps)):
            self.assertLess(abs(m), 0.2, f"DFOIL mean[{idx}] too large under null: {m}")
            self.assertLess(abs(z), 2.0, f"DFOIL Z[{idx}] too extreme under null: {z}")
            self.assertGreater(
                p, 0.01, f"DFOIL P[{idx}] unexpectedly small under null: {p}"
            )


if __name__ == "__main__":
    unittest.main()
