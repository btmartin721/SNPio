#!/usr/bin/env python3
"""Verify that release distributions contain required runtime assets."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

REQUIRED_PATHS = frozenset({"snpio/img/snpio_logo.png"})


class DistributionVerificationError(ValueError):
    """Raised when a distribution is unreadable or omits required files."""


def _normalize_sdist_member(name: str) -> str:
    """Remove the source-distribution root directory from a member path."""
    parts = PurePosixPath(name).parts
    return PurePosixPath(*parts[1:]).as_posix() if len(parts) > 1 else ""


def distribution_members(distribution: Path) -> set[str]:
    """Return normalized archive member paths for a wheel or source archive."""
    distribution = distribution.resolve()
    if not distribution.is_file():
        raise DistributionVerificationError(
            f"Distribution does not exist: {distribution}"
        )

    if distribution.suffix == ".whl":
        with zipfile.ZipFile(distribution) as archive:
            return {PurePosixPath(name).as_posix() for name in archive.namelist()}

    if distribution.name.endswith((".tar.gz", ".tar.bz2")):
        with tarfile.open(distribution, mode="r:*") as archive:
            return {
                _normalize_sdist_member(member.name)
                for member in archive.getmembers()
            }

    raise DistributionVerificationError(
        f"Unsupported distribution format: {distribution.name}"
    )


def verify_distribution(
    distribution: Path,
    required_paths: Iterable[str] = REQUIRED_PATHS,
) -> None:
    """Raise if a distribution omits any required runtime path."""
    members = distribution_members(distribution)
    missing = sorted(set(required_paths).difference(members))
    if missing:
        joined = ", ".join(missing)
        raise DistributionVerificationError(
            f"{distribution.name} is missing required runtime assets: {joined}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "distributions",
        nargs="+",
        type=Path,
        help="Wheel and source-distribution archives to inspect.",
    )
    return parser.parse_args()


def main() -> int:
    """Verify all requested distributions."""
    args = parse_args()
    for distribution in args.distributions:
        verify_distribution(distribution)
        print(f"Verified runtime assets: {distribution}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
