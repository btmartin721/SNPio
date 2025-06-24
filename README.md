# SNPio: A Python API for Population Genomic Data I/O, Filtering, and Analysis

![SNPio Logo](snpio/img/snpio_logo.png)

**SNPio** is a Python package designed to streamline the process of reading, filtering, encoding, and analyzing genotype data. It supports VCF, PHYLIP, STRUCTURE, and GENEPOP file formats, and provides high-level tools for visualization, downstream machine learning analysis, and population genetic inference.

---

## 🔧 Installation

You can install SNPio using one of the following methods:

### ✅ Pip Installation (Recommended)

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

```bash
docker pull btmartin721/snpio:latest
docker run -it btmartin721/snpio:latest
```

> **Note:** SNPio supports Unix-based systems. Windows users should install via WSL.

---

## 🚀 Getting Started

### Import Modules

```python
from snpio import (
    NRemover2, VCFReader, PhylipReader, StructureReader,
    GenePopReader, Plotting, GenotypeEncoder, PopGenStatistics
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
- PCA and missingness plots (Plotting)
- Genotype encoding (GenotypeEncoder)
- Population statistics (PopGenStatistics)
- Experimental: Tree parsing (TreeParser)

---

## 🧪 Development Notes

To run tests:

```bash
pip install snpio[dev]
pytest tests/
```

---

## 🧾 License and Citation

SNPio is licensed under the **GPL-3.0 License**. Please cite any publication(s) when using SNPio in your research.

---

## 🤝 Contributing

We welcome community contributions!

- Report bugs or request features on [GitHub Issues](https://github.com/btmartin721/snpio/issues)
- Submit a pull request
- Visit the [documentation](https://snpio.readthedocs.io/latest) for contributing guidelines

---

## 🙏 Acknowledgments

Thanks for using SNPio. We hope it facilitates your population genomic research. Reach out with questions or feedback via GitHub!
