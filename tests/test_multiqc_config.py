"""Regression tests for the documented MultiQC configuration schema."""

from __future__ import annotations

from pathlib import Path

import yaml
from jsonschema import validate
from multiqc.utils.config_schema import config_to_schema


CONFIG_PATH = Path(__file__).parents[1] / "multiqc_config.yml"
LD_PANEL_IDS = [
    "linkage_disequilibrium_summary",
    "linkage_disequilibrium_barplot",
    "linkage_disequilibrium_boxplot",
    "linkage_disequilibrium_heatmap",
    "linkage_disequilibrium_linegraph",
    "linkage_disequilibrium_barplot_Ne",
    "linkage_disequilibrium_boxplot_Ne",
    "linkage_disequilibrium_heatmap_Ne",
    "linkage_disequilibrium_linegraph_Ne",
]


def test_multiqc_config_uses_documented_section_order_schema() -> None:
    """Section ordering should use report anchors and nested order values."""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    validate(instance=config, schema=config_to_schema())

    legacy_keys = {"parent_ids", "section_names", "section_descriptions"}
    assert legacy_keys.isdisjoint(config)

    section_order = config["report_section_order"]
    assert all(set(rule) == {"order"} for rule in section_order.values())
    assert all(isinstance(rule["order"], int) for rule in section_order.values())

    ld_orders = [section_order[panel_id]["order"] for panel_id in LD_PANEL_IDS]
    assert ld_orders == sorted(ld_orders)
    assert len(ld_orders) == len(set(ld_orders))
