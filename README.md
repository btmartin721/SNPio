# SNPio: A Python API for Population Genomic Data I/O, Filtering, Analysis, and Encoding

![SNPio Logo](snpio/img/snpio_logo.png)

**SNPio** is a Python package designed to streamline the process of reading, filtering, encoding, and analyzing genotype alignments. It supports VCF, PHYLIP, STRUCTURE, and GENEPOP file formats, and provides high-level tools for visualization, downstream machine learning analysis, and population genetic inference.

SNPio Includes:

- File I/O (VCFReader, PhylipReader, StructureReader, GenePopReader)
- Genotype filtering (NRemover2)
- Genotype encoding for AI & machine learning applications (GenotypeEncoder)
- Population genetic statistics & Principal Component Analysis (PopGenStatistics)
- Finite-sample-unbiased unphased LD and LD-based recent effective population
  size (Ne), with grouped-locus bootstrap intervals and validation evidence
- Patterson, partitioned, and DFOIL statistics with random, deterministic
  least-missing, all-sample, or explicit individual selection
- Artifact-aware output organization and interactive MultiQC reporting
- Experimental: Phylogenetic tree parsing (TreeParser)

---

## 📖 Full Documentation

Detailed API usage, tutorials, and examples are available in the [Documentation](https://snpio.readthedocs.io/en/latest/)

---

## 🔧 Installation

You can install SNPio using one of the following methods:

### ✅ Pip Installation

```bash
python3 -m venv snpio-env
source snpio-env/bin/activate
pip install snpio
```

### ✅ Conda Installation

```bash
conda create -n snpio-env python=3.12
conda activate snpio-env
conda install -c btmartin721 snpio
```

### 🐳 Docker

To run the Docker image interactively in a terminal, run the following commands:

```bash
docker pull btmartin721/snpio:latest
docker run -it btmartin721/snpio:latest
```

If you'd like to run SNPio in a jupyter notebook, instructions to do so in the docker container will be printed to the terminal.  

> **Note:** All three installation versions (pip, conda, docker) are actively maintained and kept up-to-date with CI/CD routines.

> **Note:** SNPio supports Unix-based systems. Windows users should install via WSL.

---

## 🚀 Getting Started

### Import Modules

```python
from snpio import (
    NRemover2, VCFReader, PhylipReader, StructureReader,
    GenePopReader, GenotypeEncoder, PopGenStatistics
)
```

### Load Genotype Data (VCF Example)

```python
vcf = "snpio/example_data/vcf_files/phylogen_subset14K_sorted.vcf.gz"
popmap = "snpio/example_data/popmaps/phylogen_nomx.popmap"

gd = VCFReader(
    filename=vcf,
    popmapfile=popmap,
    force_popmap=True,
    verbose=True,
    plot_format="png",
    prefix="snpio_example"
)
```

You can also specify `include_pops` and `exclude_pops` to control population-level filtering.

### LD and Recent Effective Population Size

```python
from snpio import PopGenStatistics

ld = PopGenStatistics(gd).calculate_linkage_disequilibrium(
    n_bootstraps=200,
    n_jobs=-1,
    max_pairs=1_000_000,
    seed=42,
)

print(ld.summary[["Population", "r2D", "rDz", "Ne"]])
```

The returned scientific table represents non-estimable `Ne` values as `NaN`.
The MultiQC population summary adds a human-readable `Ne_Status` column.

For ordinary VCF input, SNPio infers chromosome or scaffold groups and uses
only between-group locus pairs. Coordinate-free data must provide explicit
`locus_groups` or deliberately set `assume_unlinked=True`. See the
[LD method guide](https://snpio.readthedocs.io/en/latest/linkage_disequilibrium.html)
and [validation protocol](https://snpio.readthedocs.io/en/latest/ld_validation.html).

### Output Layout

SNPio separates generated data, logs, MultiQC bundles, plots, and tabular
reports under `<prefix>_output/`. Results derived from an `NRemover2` object
are placed under `plots/nremover/<operation>/` and
`reports/nremover/<operation>/`; VCF metadata caches use `data/vcf/` with
independent filtered states under `data/vcf/nremover/`.

---

## 🧪 Development Notes

To run the unit tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/
```

The optional forward-time LD calibration dependencies are isolated from the
runtime installation:

```bash
python -m pip install -e '.[dev,ld-validation]'
```

---

## 🧾 License and Citation

SNPio is licensed under the [GPL-3.0 License](https://github.com/btmartin721/SNPio/blob/master/LICENSE).  

Please cite:

> Martin, B. T., Monaco, D. R., Sharabi, N., Mussmann, S. M., and Chafin,
> T. K. (2026). SNPio: a Python interface for population genomic data
> processing. *BMC Bioinformatics*.
> https://doi.org/10.1186/s12859-026-06546-5

When reporting unphased LD or LD-based recent Ne, also cite Ragsdale and
Gravel (2020), *Molecular Biology and Evolution*, 37(3), 923–932.

---

## 🤝 Contributing

We welcome community contributions!

- Report bugs or request features on [GitHub Issues](https://github.com/btmartin721/snpio/issues)
- Submit a pull request
- See [CONTRIBUTING.md](https://github.com/btmartin721/SNPio/blob/master/CONTRIBUTING.md) for contributing guidelines.

---

## 🙏 Acknowledgments

Thanks for using SNPio. We hope it facilitates your population genomic research. Feel free to reach out with questions or feedback!
