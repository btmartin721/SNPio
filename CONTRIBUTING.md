# Contributing to SNPio

Thank you for your interest in contributing to SNPio! 🎉  
Whether it's fixing bugs, improving documentation, adding tests, or implementing new features, your help is appreciated.

Please follow these guidelines to ensure a smooth and productive collaboration.

---

## 📦 Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/btmartin721/SNPio.git
   cd SNPio
   ```

2. **Set up a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Run tests locally:**

   ```bash
   pytest
   ```

---

## 🧠 Code Style & Linting

- Follow [PEP8](https://www.python.org/dev/peps/pep-0008/) and Google-style docstrings.
- Use [`black`](https://github.com/psf/black) for formatting:

   ```bash
   black snpio/
   ```

---

## 🔁 Branching Model

- Use feature branches based on `master`.
- Use descriptive names: `feature/vcf-compression`, `fix/genepop-parser`, etc.
- Submit a **pull request** to `master`.

---

## ✍️ Commit Message Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/) to automate versioning with GitVersion.

Use this structure:

```text
<type>[!]: <short summary>

[optional body]

[optional BREAKING CHANGE footer]
```

### Types

- `feat`: new feature → MINOR bump
- `fix`: bug fix → PATCH bump
- `feat!` or `BREAKING CHANGE:` → MAJOR bump
- `docs`, `chore`, `refactor`, `test`, etc. → no version change

### Examples

```bash
feat: add PhylipReader support for diploid data
fix: correct bug in allele filtering logic
feat!: change output directory structure

BREAKING CHANGE: Output files are now grouped by population.
```

---

## 🧪 Running Tests

```bash
pytest
```

To test a specific file:

```bash
pytest tests/test_vcf_reader.py
```

---

## 📝 Changelog

Version numbers are automatically calculated using [GitVersion](https://gitversion.net) based on commit history. Do **not** manually change version numbers in `pyproject.toml` or `meta.yaml` — this is handled by CI.

For details, see [docs/dev/versioning.md](docs/dev/versioning.md)

---

## 🧪 Need Help?

Feel free to open a GitHub Discussion or Issue.

Thank you for contributing! 🙌
