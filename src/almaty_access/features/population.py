"""Population features: residents and the residents-to-workspace ratio.

``pop_per_1k_nonres_gfa`` relates residents to the non-residential
floor area around them (commercial + industrial + public GFA). The
denominator gets a 100 m² floor so hexes with essentially no workspace
produce a large-but-finite ratio instead of dividing by zero — the
feature table must stay NaN- and inf-free for Phase 3.
"""

from __future__ import annotations

import geopandas as gpd
import h3
import pandas as pd

#: Functions whose GFA counts as workspace.
NONRES_FUNCTIONS: tuple[str, ...] = ("commercial", "industrial", "public")

#: Denominator floor, m² — keeps the ratio finite (documented above).
MIN_NONRES_GFA_M2 = 100.0


def population_features(
    population: gpd.GeoDataFrame, built_form: pd.DataFrame, grid: gpd.GeoDataFrame, res: int
) -> pd.DataFrame:
    """Per-hex population block (``pop_`` columns), aligned to the grid."""
    index = pd.Index(grid["h3"], name="h3")
    out = pd.DataFrame(index=index)
    residents = population.set_index("h3")["population"].reindex(index).fillna(0.0).astype(float)
    out["pop_residents"] = residents
    out["pop_density_km2"] = residents / h3.average_hexagon_area(res, unit="km^2")
    nonres = sum(built_form[f"bf_gfa_{function}_m2"] for function in NONRES_FUNCTIONS)
    out["pop_per_1k_nonres_gfa"] = residents / nonres.clip(lower=MIN_NONRES_GFA_M2) * 1000.0
    return out
