"""Centralized output-directory layout for SNPio artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from snpio.read_input.genotype_data import GenotypeData


ArtifactType = Literal["plots", "reports"]


@dataclass(frozen=True, slots=True)
class OutputPaths:
    """Resolve SNPio output paths by artifact type and filtering state.

    Plot and report artifacts are organized beneath their respective top-level
    directories. Artifacts derived from an ``NRemover2``-filtered dataset are
    nested beneath an ``nremover`` scope within those directories.
    """

    prefix: str | Path
    filtered: bool = False

    @classmethod
    def from_genotype_data(
        cls,
        genotype_data: "GenotypeData",
        *,
        force_filtered: bool = False,
    ) -> "OutputPaths":
        """Create paths from a genotype dataset and its filtering history."""

        return cls(
            prefix=genotype_data.prefix,
            filtered=force_filtered or bool(genotype_data.was_filtered),
        )

    @property
    def root(self) -> Path:
        """Return the run's top-level output directory."""

        return Path(f"{self.prefix}_output")

    @property
    def logs(self) -> Path:
        """Return the directory reserved for true log and run-provenance files."""

        return self.root / "logs"

    @property
    def multiqc(self) -> Path:
        """Return the MultiQC report bundle directory."""

        return self.root / "multiqc"

    @property
    def data(self) -> Path:
        """Return the directory for generated or cached data artifacts."""

        return self.root / "data"

    @property
    def vcf_data(self) -> Path:
        """Return the directory for cached VCF metadata in HDF5 format."""

        return self.data / "vcf"

    @property
    def popmaps(self) -> Path:
        """Return the directory for generated population-map files."""

        return self.data / "popmaps"

    def plots(self, operation: str | None = None) -> Path:
        """Return the plot directory for an optional named operation."""

        return self._artifact_dir("plots", operation)

    def reports(self, operation: str | None = None) -> Path:
        """Return the report directory for an optional named operation."""

        return self._artifact_dir("reports", operation)

    def _artifact_dir(
        self,
        artifact_type: ArtifactType,
        operation: str | None,
    ) -> Path:
        directory = self.root / artifact_type
        if self.filtered:
            directory /= "nremover"
        if operation is not None:
            self._validate_operation(operation)
            directory /= operation
        return directory

    @staticmethod
    def _validate_operation(operation: str) -> None:
        """Reject empty, absolute, or nested operation directory names."""

        operation_path = Path(operation)
        if (
            not operation.strip()
            or operation_path.is_absolute()
            or len(operation_path.parts) != 1
            or operation in {".", ".."}
        ):
            raise ValueError(
                "Output operation must be a single non-empty directory name; "
                f"got {operation!r}."
            )
