# SNPio: A Python API for Population Genomic Data I/O, Filtering, Analysis, and Encoding

![SNPio Logo](snpio/img/snpio_logo.png)

**SNPio** is a Python package designed to streamline the process of reading, filtering, encoding, and analyzing genotype alignments. It supports VCF, PHYLIP, STRUCTURE, and GENEPOP file formats, and provides high-level tools for visualization, downstream machine learning analysis, and population genetic inference.

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

---

## 📖 Full Documentation

Detailed API usage, tutorials, and examples are available at:

🔗 [https://snpio.readthedocs.io/latest](https://snpio.readthedocs.io/latest)

Includes:

- File readers (VCF, PHYLIP, STRUCTURE, GENEPOP)
- Genotype filtering (NRemover2)
- PCA and missingness plots (PopGenStatistics)
- Genotype encoding (GenotypeEncoder)
- Population statistics (PopGenStatistics)
- Experimental: Phylogenetic tree parsing (TreeParser)

---

## 🧪 Development Notes

See [CONTRIBUTING.md](https://github.com/btmartin721/SNPio/blob/master/CONTRIBUTING.md) for more information on contributing to the SNPio code.

To run unit tests:

```bash
pip install snpio[dev]
pytest tests/
```

This will run all unit tests in the `tests` directory.

---

## 🧾 License and Citation

SNPio is licensed under the [GPL-3.0 License](https://github.com/btmartin721/SNPio/blob/master/LICENSE).  

Please cite any publication(s) when using SNPio in your research. A manuscript is currently in development, and this section will be updated upon acceptance.

---

## 🤝 Contributing

We welcome community contributions!

- Report bugs or request features on [GitHub Issues](https://github.com/btmartin721/snpio/issues)
- Submit a pull request
- See [CONTRIBUTING.md](https://github.com/btmartin721/SNPio/blob/master/CONTRIBUTING.md) for contributing guidelines.

---

## 🙏 Acknowledgments

Thanks for using SNPio. We hope it facilitates your population genomic research. Feel free to reach out with questions or feedback!
