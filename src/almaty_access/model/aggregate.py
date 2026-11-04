"""Hex → district aggregation with unit consistency.

The random forest trains on district rows and predicts on hex rows, so
both must express every feature in the SAME units (a model trained on
district-scale GFA sums would see hex sums ~1000x smaller and all its
splits would misfire). Rule per column:

- **mass** (m² of floor space, counts, metres of road, persons,
  k-ring sums): divided by the unit's area → densities per km²,
- **intensity** (shares, means, entropy, distances, densities that
  already are per-km², ratios, k-ring means): plain mean over the
  district's hexes (each hex counts once — see district_design's
  docstring on why border-share weighting is deliberately skipped),
  unchanged at hex level.

Every feature column must match a rule; an unknown column raises, so a
future feature cannot silently enter the model in the wrong units.
"""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd

from almaty_access.config import Config
from almaty_access.schemas import FEATURE_ID_COLUMNS

#: Columns that are per-unit intensities (aggregate as weighted means).
_MEAN_MARKERS = ("share", "density", "entropy", "per_1k")
_MEAN_PREFIXES = (
    "bf_levels",
    "tr_dist",
    "net_dist",
    "cent_dist",
    "ter_",
    "net_betweenness",
)
#: Columns that are masses (aggregate as sums, then per km²).
_MASS_PREFIXES = (
    "bf_",
    "poi_cnt_",
    "net_intersection_cnt",
    "net_len_",
    "tr_stop_cnt",
    "pop_residents",
)


def column_kind(column: str) -> str:
    """'mass' or 'mean' — how a feature aggregates (raises if unknown)."""
    if any(marker in column for marker in _MEAN_MARKERS):
        return "mean"
    if column.startswith(_MEAN_PREFIXES):
        return "mean"
    if column.startswith("ctx_"):
        return "mass" if "_sum_" in column else "mean"
    if column.startswith(_MASS_PREFIXES):
        return "mass"
    raise ValueError(
        f"feature column '{column}' matches no aggregation rule - "
        "extend model/aggregate.py before it can enter the model"
    )


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in features.columns if column not in FEATURE_ID_COLUMNS]


def hex_design(features: pd.DataFrame) -> pd.DataFrame:
    """Per-hex design matrix in model units, indexed by h3 (city hexes).

    Mass columns become densities per km² of hex area so hex rows and
    district rows live on the same scale.
    """
    rows = features.loc[features["in_city"]].set_index("h3")
    resolution = int(rows["res"].iloc[0])
    hex_area_km2 = h3.average_hexagon_area(resolution, unit="km^2")
    design = pd.DataFrame(index=rows.index)
    for column in feature_columns(features):
        values = rows[column].astype(float)
        design[column] = values / hex_area_km2 if column_kind(column) == "mass" else values
    return design


def district_design(features: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Per-district design matrix in model units (+ ``area_km2``).

    Hex contributions are weighted by ``district_share`` would be ideal,
    but the feature table deliberately carries only the winning
    district per hex (D-025); the share-weighted refinement matters at
    district borders only, far below the model's resolution. Masses are
    summed and divided by the district's hex area; intensities are
    plain means over the district's hexes.
    """
    rows = features.loc[features["in_city"]]
    resolution = int(rows["res"].iloc[0])
    hex_area_km2 = h3.average_hexagon_area(resolution, unit="km^2")
    grouped = rows.groupby("district")
    area_km2 = grouped.size().astype(float) * hex_area_km2

    design = pd.DataFrame(index=area_km2.index)
    for column in feature_columns(features):
        if column_kind(column) == "mass":
            design[column] = grouped[column].sum() / area_km2
        else:
            design[column] = grouped[column].mean()
    design["area_km2"] = area_km2
    expected = {district.key for district in config.districts}
    extra = set(design.index) - expected
    if extra:
        raise ValueError(f"feature table contains unknown districts: {sorted(extra)}")
    if design.isna().any().any():
        bad = design.columns[design.isna().any()].tolist()
        raise ValueError(f"district design has NaN in {bad}")
    return design


def apply_log1p(design: pd.DataFrame, prefixes: tuple[str, ...]) -> pd.DataFrame:
    """The configured model-time transform (README: recorded in config)."""
    result = design.copy()
    for column in result.columns:
        if column.startswith(prefixes) and column != "area_km2":
            result[column] = np.log1p(result[column].clip(lower=0))
    return result
