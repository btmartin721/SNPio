#!/usr/bin/env python3

"""
Update version strings across pyproject.toml, meta.yaml, and conf.py.

Usage:
    python scripts/update_versions.py <new_version>
"""

import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib

import tomli_w


def update_file(path: str | Path, pattern: str, replacement: str) -> None:
    """Replace all matches of a regex pattern in a file with a new versioned string."""
    path = Path(path)
    if not path.exists():
        print(f"Warning: {path} does not exist.")
        return
    content = path.read_text()
    updated = re.sub(pattern, replacement, content)
    path.write_text(updated)
    print(f"Updated {path}")


def update_pyproject(version: str) -> None:
    """Update the version in pyproject.toml."""
    path = Path("pyproject.toml")
    if not path.exists():
        print("Error: pyproject.toml not found.")
        sys.exit(1)

    with path.open("rb") as f:
        data = tomllib.load(f)

    data["project"]["version"] = version
    path.write_text(tomli_w.dumps(data))
    print(f"Updated pyproject.toml version to {version}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/update_versions.py <new_version>")
        sys.exit(1)

    version = sys.argv[1]

    update_pyproject(version)

    update_file(
        "recipe/meta.yaml", r'(version\s*=\s*)"\d+\.\d+\.\d+"', rf'\1"{version}"'
    )

    update_file(
        "snpio/docs/source/conf.py",
        r'(release\s*=\s*)"\d+\.\d+\.\d+"',
        rf'\1"{version}"',
    )


if __name__ == "__main__":
    main()
