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

The release workflow can calculate versions using
[GitVersion](https://gitversion.net), while the repository release script
accepts an explicit `MAJOR.MINOR.PATCH` version for predictable releases. Do
**not** manually change version numbers in `pyproject.toml`, `meta.yaml`, or the
Sphinx configuration; CI updates them atomically.

## 🚀 Creating a Release

After the release changes and changelog are merged into `master`, run:

```bash
./scripts/release.zsh 1.7.5
```

The script validates the version, checks for an existing tag, blocks overlapping
release/publisher runs, asks for confirmation, and triggers the guarded GitHub
Actions workflow from `origin/master`. By default, it monitors the release plus
the PyPI/Docker and Conda publishers through completion.

Use `--dry-run` to perform only the preflight checks, `--no-wait` to exit after
dispatching the workflow, or `--yes` to skip confirmation. Run
`./scripts/release.zsh --help` for the complete interface.

Do **not** create or push the tag manually. The workflow creates the version
commit, `vMAJOR.MINOR.PATCH` tag, and GitHub Release, then dispatches the package
publishers.

---

## 🧪 Need Help?

Feel free to open a GitHub Discussion or Issue.

Thank you for contributing! 🙌
