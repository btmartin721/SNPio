#!/usr/bin/env python3
"""Generate the frozen LD oracle corpus from moments-popgen 1.6.0.

This maintenance script is not used by SNPio at runtime. Install the exact
reference release separately, then run, for example::

    python scripts/validation/generate_ld_golden_fixture.py \
        --moments-source /tmp/snpio-moments-reference
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260715
N_CASES = 1_000
SAMPLE_SIZES = (4, 5, 6, 8, 10, 20, 50, 100, 200)
OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "snpio"
    / "validation"
    / "data"
    / "moments_popgen_1_6_0_golden.csv.gz"
)


def parse_args() -> argparse.Namespace:
    """Parse maintenance-script arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--moments-source",
        type=Path,
        default=None,
        help="Directory containing an isolated moments-popgen 1.6.0 install.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def _reference_module(source: Path | None):
    """Import the reference implementation without making it a dependency."""

    if source is not None:
        sys.path.insert(0, str(source.resolve()))
    try:
        import moments  # type: ignore[import-not-found]
        from moments.LD import stats_from_genotype_counts
    except ImportError as error:
        raise SystemExit(
            "moments-popgen 1.6.0 is required only to regenerate this fixture. "
            "Install it in an isolated directory and pass --moments-source."
        ) from error
    if moments.__version__ != "1.6.0":
        raise SystemExit(
            f"Expected moments-popgen 1.6.0, found {moments.__version__}."
        )
    return stats_from_genotype_counts


def _count_vectors() -> np.ndarray:
    """Create deterministic dense and sparse nine-state count vectors."""

    rng = np.random.default_rng(SEED)
    rows = []
    for case_id in range(N_CASES):
        sample_size = SAMPLE_SIZES[case_id % len(SAMPLE_SIZES)]
        if case_id < 9:
            counts = np.zeros(9, dtype=np.int64)
            counts[case_id] = sample_size
        elif case_id < 45:
            states = rng.choice(9, size=2 + case_id % 4, replace=False)
            probabilities = rng.dirichlet(np.full(states.size, 0.35))
            counts = np.zeros(9, dtype=np.int64)
            counts[states] = rng.multinomial(sample_size, probabilities)
        else:
            concentration = 0.25 if case_id % 3 == 0 else 1.5
            probabilities = rng.dirichlet(np.full(9, concentration))
            counts = rng.multinomial(sample_size, probabilities)
        rows.append(counts)
    return np.asarray(rows, dtype=np.int64)


def main() -> None:
    """Generate compressed fixture data and a provenance sidecar."""

    args = parse_args()
    reference = _reference_module(args.moments_source)
    counts = _count_vectors()
    rows = []
    for case_id, count_vector in enumerate(counts):
        values = count_vector.tolist()
        count_list = [values]
        rows.append(
            {
                "case_id": case_id,
                "sample_size": int(count_vector.sum()),
                **{f"n{index + 1}": value for index, value in enumerate(values)},
                "D": float(reference.Dhat(values)),
                "D2": float(reference.DD(count_list, [0, 0])),
                "Dz": float(reference.Dz(count_list, [0, 0, 0])),
                "pi2": float(reference.pi2(count_list, [0, 0, 0, 0])),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(
        args.output,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.17g",
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    provenance = {
        "generator": str(
            Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[2])
        ),
        "reference_distribution": "moments-popgen",
        "reference_version": "1.6.0",
        "reference_module": "moments.LD.stats_from_genotype_counts",
        "seed": SEED,
        "case_count": N_CASES,
        "sample_sizes": list(SAMPLE_SIZES),
        "sha256": digest,
        "columns": frame.columns.tolist(),
    }
    args.output.with_suffix("").with_suffix(".json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(frame):,} cases to {args.output}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
