from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class PlotConfig:
    """Immutable plotting configuration.

    Attributes:
        plot_format (str): Plot file format (e.g., 'png', 'jpeg', 'jpg', 'pdf').
        plot_fontsize (int): Font size for plot text.
        dpi (int): Dots per inch (DPI) for plot resolution.
        despine (bool): Whether to remove top and right spines from plots.
        show (bool): Whether to display plots interactively.
        verbose (bool): Whether to enable verbose logging.
        debug (bool): Whether to enable debug mode.
    """

    plot_format: Literal["png", "jpeg", "jpg", "pdf"]
    plot_fontsize: int
    dpi: int
    despine: bool
    show: bool
    verbose: bool
    debug: bool

    def __post_init__(self):
        valid_formats = {"png", "jpeg", "jpg", "pdf"}
        if self.plot_format not in valid_formats:
            raise ValueError(
                f"Invalid plot format: {self.plot_format}. Supported formats: {valid_formats}"
            )

        if self.plot_fontsize <= 0:
            raise ValueError("Font size must be a positive integer.")

        if self.dpi <= 0:
            raise ValueError("DPI must be a positive integer.")

        if not isinstance(self.despine, bool):
            raise ValueError("Despine must be a boolean value.")

        if not isinstance(self.show, bool):
            raise ValueError("Show must be a boolean value.")

        if not isinstance(self.verbose, bool):
            raise ValueError("Verbose must be a boolean value.")

        if not isinstance(self.debug, bool):
            raise ValueError("Debug must be a boolean value.")

    def to_dict(self) -> dict:
        """Convert the PlotConfig to a dictionary."""
        return {
            "plot_format": self.plot_format,
            "plot_fontsize": self.plot_fontsize,
            "dpi": self.dpi,
            "despine": self.despine,
            "show": self.show,
            "verbose": self.verbose,
            "debug": self.debug,
        }


@dataclass(frozen=True)
class IOConfig:
    """Immutable IO configuration for GenotypeData."""

    prefix: str
    chunk_size: int
    force_popmap: bool
    include_pops: list[str] | None
    exclude_pops: list[str] | None
    verbose: bool
    debug: bool

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be a positive integer.")

        if not isinstance(self.force_popmap, bool):
            raise ValueError("force_popmap must be a boolean value.")

        if self.include_pops is not None and not isinstance(self.include_pops, list):
            raise ValueError("include_pops must be a list of population names or None.")

        if self.exclude_pops is not None and not isinstance(self.exclude_pops, list):
            raise ValueError("exclude_pops must be a list of population names or None.")

        if not isinstance(self.verbose, bool):
            raise ValueError("Verbose must be a boolean value (True or False).")

        if not isinstance(self.debug, bool):
            raise ValueError("Debug must be a boolean value (True or False).")

    def to_dict(self) -> dict:
        """Convert the IOConfig to a dictionary."""
        return {
            "prefix": self.prefix,
            "chunk_size": self.chunk_size,
            "force_popmap": self.force_popmap,
            "include_pops": self.include_pops,
            "exclude_pops": self.exclude_pops,
            "verbose": self.verbose,
            "debug": self.debug,
        }


@dataclass
class PopState:
    """Mutable population state (updates after filtering).

    Attributes:
        samples (list[str]): List of sample names.
        populations (list[str | int]): List of population identifiers corresponding to samples.
        popmap (dict[str, str | int] | None): Mapping from sample names to
            population identifiers.
        popmap_inverse (dict[str, list[str]] | None): Inverse mapping from population identifiers to lists of sample names.
        num_pops (int): Number of unique populations.
    """

    samples: list[str] = field(default_factory=list)
    populations: list[str | int] = field(default_factory=list)
    popmap: dict[str, str | int] | None = None
    popmap_inverse: dict[str, list[str]] | None = None
    num_pops: int = 0

    def __post_init__(self):
        self.num_pops = len(list(set(self.populations)))

    def to_dict(self) -> dict:
        """Convert the PopState to a dictionary."""
        return {
            "samples": self.samples,
            "populations": self.populations,
            "popmap": self.popmap,
            "popmap_inverse": self.popmap_inverse,
            "num_pops": self.num_pops,
        }
