"""Tests for release-distribution runtime-asset verification."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.verify_distributions import (
    DistributionVerificationError,
    verify_distribution,
)

LOGO_PATH = "snpio/img/snpio_logo.png"


def test_verify_distribution_accepts_wheel_with_required_asset(
    tmp_path: Path,
) -> None:
    """A wheel containing every required runtime asset should pass."""
    wheel = tmp_path / "snpio-1.7.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr(LOGO_PATH, b"logo")

    verify_distribution(wheel)


def test_verify_distribution_accepts_sdist_with_required_asset(
    tmp_path: Path,
) -> None:
    """An sdist root prefix should be removed before checking member paths."""
    sdist = tmp_path / "snpio-1.7.1.tar.gz"
    payload = b"logo"
    info = tarfile.TarInfo(f"snpio-1.7.1/{LOGO_PATH}")
    info.size = len(payload)
    with tarfile.open(sdist, mode="w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))

    verify_distribution(sdist)


def test_verify_distribution_rejects_missing_runtime_asset(
    tmp_path: Path,
) -> None:
    """A distribution missing a required runtime asset should fail clearly."""
    wheel = tmp_path / "snpio-1.7.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr("snpio/__init__.py", "")

    with pytest.raises(
        DistributionVerificationError,
        match="snpio/img/snpio_logo.png",
    ):
        verify_distribution(wheel)
