"""Shared result-schema helpers for Fst outlier analyses."""

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

FST_OUTLIER_COLUMNS = ("Locus", "Population_Pair", "Fst", "q_value")


def build_fst_outlier_dataframe(
    records: Iterable[Mapping[str, Any]] = (),
) -> pd.DataFrame:
    """Build a long-form Fst outlier result with a stable column schema."""

    return pd.DataFrame.from_records(records, columns=FST_OUTLIER_COLUMNS)
