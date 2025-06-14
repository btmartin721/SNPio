# SNPio Versioning and Release Model

SNPio uses **Semantic Versioning** (`MAJOR.MINOR.PATCH`) and is powered by [GitVersion](https://gitversion.net/) with `Mainline` mode enabled.

This allows automatic version bumps based on commit history and PR merges.

---

## 🔢 Semantic Versioning

| Version Part | Triggered By |
|--------------|--------------|
| MAJOR        | Breaking change (`feat!:` or `BREAKING CHANGE:`) |
| MINOR        | New feature (`feat:`) |
| PATCH        | Bug fix or maintenance (`fix:`, `refactor:`) |

The version is calculated **dynamically** at CI time based on Git history and commit messages.

Any version increment will trigger pushes to PyPi, Anaconda (btmartin721 channel), and DockerHub. The Docker image will be built prior to being pushed.

---

## 🔧 GitVersion Configuration

Our `.gitversion.yml` is configured as:

```yaml
mode: Mainline
commit-message-incrementing: Enabled

branches:
  main:
    regex: ^main$
    increment: Minor
    is-release-branch: true
    prevent-increment-of-merged-branch-version: false
    tag: ""
  master:
    regex: ^master$
    increment: Minor
    is-release-branch: true
    prevent-increment-of-merged-branch-version: false
    tag: ""
```

---

## ✍️ Conventional Commit Guide

Use the following format when writing commits:

```text
<type>[!]: short summary

optional detailed explanation

optional footer (e.g., BREAKING CHANGE:)
```

### Supported Types

| Type       | Meaning                          | Version Bump |
|------------|----------------------------------|---------------|
| `feat:`    | New feature                      | Minor         |
| `fix:`     | Bug fix                          | Patch         |
| `feat!:`   | Breaking change in a feature     | Major         |
| `BREAKING CHANGE:` | Major bump trigger      | Major         |
| `refactor:`| Internal improvement             | Patch         |
| `docs:`    | Documentation only               | None          |
| `chore:`   | Build/CI/tooling changes         | None          |
| `test:`    | Add or update tests              | None          |

---

## ✅ Example Commits

```bash
feat: add PCA support for STRUCTURE files
fix: correct typo in logger output
refactor: improve memory usage in chunked reads
feat!: drop Python < 3.10 support

BREAKING CHANGE: The minimum supported Python version is now 3.11.
```

---

## 🛠 Version Tagging & Releases

- Version tags are generated automatically by GitHub Actions on every push to `master`.
- Tags follow the pattern: `v1.2.3`
- A GitHub Release is created from the tag with auto-generated notes.
- Do **not** manually edit version files — `scripts/update_versions.py` handles that based on the CI-computed version.

---

## 🧪 CI Integration

CI uses GitVersion to:

- Compute the next version number
- Update `pyproject.toml`, `recipe/meta.yaml`, and `snpio/docs/conf.py`
- Push a version tag (`vX.Y.Z`)
- Create a GitHub Release with release notes

This will in-turn push the new version to PyPi, Anaconda, and DockerHub.

---

## 🔍 Debugging Version Bumps

Use the following in CI logs:

```bash
git describe --tags
gitversion /showvariable FullSemVer
```

If the version is not incrementing:

- Ensure the commit follows the correct format.
- Ensure the tag `vX.Y.Z` has not already been created.
- Confirm a merge commit was created and squashed properly.

---

## 📜 References

- [GitVersion Docs](https://gitversion.net/docs/)
- [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
- [SemVer](https://semver.org)

---

## 🧩 Contributors

If you're working on a new release or hotfix, follow the commit conventions and let CI handle tagging and publishing. For questions, ask in GitHub Discussions or contact the maintainer.
