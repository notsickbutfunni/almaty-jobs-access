"""Categorize and clean OSM points of interest.

Raw POIs (one column per OSM tag key, see ``schemas.RAW_OSM_POIS``) are
mapped to a compact category scheme defined in
``config/poi_categories.yaml``: an ordered list of rules where the first
match wins. Unmapped POIs are counted, logged and dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

from almaty_access.config import Config
from almaty_access.io import ensure_built, read_geoparquet, write_geoparquet
from almaty_access.prepare.boundary import BUFFER_FILE

logger = logging.getLogger(__name__)

RAW_FILE = "osm_pois.parquet"
OUTPUT_FILE = "pois.parquet"


@dataclass(frozen=True)
class CategoryRule:
    """One mapping rule: tag key (+ allowed values) -> category."""

    category: str
    key: str
    values: frozenset[str] | None
    """Allowed tag values; None means any non-empty value ("*")."""


def load_category_rules(path: str | Path) -> list[CategoryRule]:
    """Parse ``poi_categories.yaml`` into an ordered, flat rule list."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or not isinstance(raw.get("categories"), list):
        raise ValueError(f"{path}: expected a mapping with a 'categories' list")
    rules: list[CategoryRule] = []
    for block in raw["categories"]:
        if not isinstance(block, dict) or "category" not in block or "rules" not in block:
            raise ValueError(f"{path}: each category block needs 'category' and 'rules'")
        category = str(block["category"])
        for rule in block["rules"]:
            if not isinstance(rule, dict) or "key" not in rule:
                raise ValueError(f"{path}: rule in '{category}' needs a 'key'")
            values = rule.get("values", "*")
            if values == "*":
                parsed = None
            elif isinstance(values, list) and values:
                parsed = frozenset(str(v) for v in values)
            else:
                raise ValueError(
                    f"{path}: 'values' in '{category}' must be a non-empty list or '*'"
                )
            rules.append(CategoryRule(category=category, key=str(rule["key"]), values=parsed))
    if not rules:
        raise ValueError(f"{path}: no rules defined")
    return rules


def categorize(pois: pd.DataFrame, rules: Sequence[CategoryRule]) -> pd.Series:
    """Assign each POI the category of the first matching rule (else <NA>).

    A rule matches when its tag-key column holds a non-empty value and,
    unless the rule is a wildcard, the value is in the rule's set.
    Missing columns are skipped, so the same rule file works for frames
    that carry only a subset of the tag keys.
    """
    category = pd.Series(pd.NA, index=pois.index, dtype="string")
    for rule in rules:
        if rule.key not in pois.columns:
            continue
        column = pois[rule.key].astype("string")
        has_value = column.notna() & (column.str.len() > 0)
        matches = has_value if rule.values is None else has_value & column.isin(rule.values)
        category = category.mask(category.isna() & matches, rule.category)
    return category


def prepare_pois(config: Config, *, force: bool = False) -> Path:
    """Categorize raw POIs, clip to the buffered boundary, deduplicate."""
    raw_path = config.paths.raw / RAW_FILE
    buffer_path = config.paths.interim / BUFFER_FILE
    rules_path = config.pois.categories_file
    output = config.paths.interim / OUTPUT_FILE

    def build() -> None:
        rules = load_category_rules(rules_path)
        pois = read_geoparquet(raw_path)
        boundary = read_geoparquet(buffer_path).geometry.iloc[0]

        pois["category"] = categorize(pois, rules)
        unmapped = pois["category"].isna()
        if unmapped.any():
            logger.info(
                "dropping %d of %d POIs with no category (kept %.1f%%)",
                int(unmapped.sum()),
                len(pois),
                100 * (1 - unmapped.mean()),
            )
        pois = pois.loc[~unmapped]

        pois = pois[pois.geometry.within(boundary)]
        pois = pois.drop_duplicates(subset="osm_id", keep="first")
        # railway/station ride along so Phase 2 transit features can tell
        # metro stations (railway=station + station=subway) from rail ones.
        na = pd.Series(pd.NA, index=pois.index)
        result = gpd.GeoDataFrame(
            {
                "osm_id": pois["osm_id"].astype("string"),
                "name": pois.get("name", na).astype("string"),
                "category": pois["category"],
                "railway": pois.get("railway", na).astype("string"),
                "station": pois.get("station", na).astype("string"),
            },
            geometry=pois.geometry,
            crs=pois.crs,
        )
        logger.info("pois: %d features in %d categories", len(result), result["category"].nunique())
        write_geoparquet(result, output)

    ensure_built(
        output, [raw_path, buffer_path, rules_path, config.source_path], build, force=force
    )
    return output
