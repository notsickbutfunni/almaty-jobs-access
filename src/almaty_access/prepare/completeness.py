"""OSM completeness audit: how much does OSM cover, per district?

Compares the merged building layer's OSM share and POI counts across
districts. A district where most footprints come from Overture has weak
OSM coverage — its model inputs (tags, POIs) are less trustworthy, which
matters when interpreting the job-density model (README §7 Phase 1).
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from almaty_access.config import Config
from almaty_access.io import ensure_built, read_geoparquet
from almaty_access.prepare.boundary import DISTRICTS_FILE
from almaty_access.prepare.buildings import OUTPUT_FILE as BUILDINGS_FILE
from almaty_access.prepare.pois import OUTPUT_FILE as POIS_FILE

logger = logging.getLogger(__name__)

OUTPUT_FILE = "completeness.csv"

#: Flag a district when less than this share of its footprints is from OSM.
WEAK_OSM_SHARE = 0.5


def completeness_table(
    buildings: gpd.GeoDataFrame, pois: gpd.GeoDataFrame, districts: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Per-district OSM/Overture building counts, POI counts and flags."""
    building_points = gpd.GeoDataFrame(
        {"src": buildings["src"].astype(str)},
        geometry=buildings.geometry.representative_point(),
        crs=buildings.crs,
    )
    with_district = gpd.sjoin(
        building_points, districts[["district", "geometry"]], predicate="within"
    )
    counts = with_district.groupby(["district", "src"]).size().unstack(fill_value=0).reset_index()
    for column in ("osm", "overture"):
        if column not in counts.columns:
            counts[column] = 0
    # Every district stays in the table — a district with zero buildings
    # is the strongest possible coverage warning, not a row to drop.
    counts = (
        pd.DataFrame({"district": districts["district"].unique()})
        .merge(counts, on="district", how="left")
        .fillna({"osm": 0, "overture": 0})
        .astype({"osm": int, "overture": int})
    )

    poi_counts = (
        gpd.sjoin(pois[["geometry"]], districts[["district", "geometry"]], predicate="within")
        .groupby("district")
        .size()
        .rename("pois")
    )

    table = counts.merge(poi_counts, on="district", how="left").fillna({"pois": 0})
    table = table.rename(columns={"osm": "buildings_osm", "overture": "buildings_overture"})
    total = table["buildings_osm"] + table["buildings_overture"]
    table["osm_share"] = (table["buildings_osm"] / total.where(total > 0)).round(3)
    table["weak_osm_coverage"] = table["osm_share"].fillna(0) < WEAK_OSM_SHARE
    table["pois"] = table["pois"].astype(int)
    return table[
        [
            "district",
            "buildings_osm",
            "buildings_overture",
            "osm_share",
            "pois",
            "weak_osm_coverage",
        ]
    ].sort_values("district")


def prepare_completeness(config: Config, *, force: bool = False) -> Path:
    """Write the per-district completeness table to data/interim/."""
    buildings_path = config.paths.interim / BUILDINGS_FILE
    pois_path = config.paths.interim / POIS_FILE
    districts_path = config.paths.interim / DISTRICTS_FILE
    output = config.paths.interim / OUTPUT_FILE

    def build() -> None:
        table = completeness_table(
            read_geoparquet(buildings_path),
            read_geoparquet(pois_path),
            read_geoparquet(districts_path),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
        weak = table.loc[table["weak_osm_coverage"], "district"].tolist()
        if weak:
            logger.warning("districts with weak OSM coverage: %s", ", ".join(weak))
        else:
            logger.info("no district flagged for weak OSM coverage")

    ensure_built(
        output, [buildings_path, pois_path, districts_path, config.source_path], build, force=force
    )
    return output
