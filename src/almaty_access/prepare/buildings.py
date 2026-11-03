"""Merge OSM and Overture buildings; estimate floors, function and GFA.

Steps (README §7 Phase 1):

1. **Merge**: Overture footprints that overlap an OSM footprint with
   IoU > threshold are duplicates — the OSM copy wins (richer tags);
   non-overlapping Overture footprints fill OSM's gaps.
2. **Function**: residential / commercial / industrial / public /
   unknown, decided in priority order: the building's own type tag →
   its own amenity/shop/office tags → categorized POIs inside it →
   the land-use polygon under its centroid.
3. **Floors**: ``building:levels`` tag when present, else height / 3 m,
   else a per-function default from the config.
4. **GFA**: footprint area (m2, metric CRS) x floors.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from almaty_access.config import Config
from almaty_access.io import ensure_built, read_geoparquet, write_geoparquet
from almaty_access.prepare.boundary import BUFFER_FILE
from almaty_access.prepare.pois import OUTPUT_FILE as POIS_FILE

logger = logging.getLogger(__name__)

RAW_OSM_FILE = "osm_buildings.parquet"
RAW_OVERTURE_FILE = "overture_buildings.parquet"
RAW_LANDUSE_FILE = "osm_landuse.parquet"
OUTPUT_FILE = "buildings.parquet"

_NUMBER_RE = re.compile(r"(\d+(?:[.,]\d+)?)")

#: building=<tag> (OSM) and subtype/class (Overture) -> function.
TAG_FUNCTION: dict[str, str] = {
    # residential
    "apartments": "residential",
    "house": "residential",
    "residential": "residential",
    "detached": "residential",
    "semidetached_house": "residential",
    "terrace": "residential",
    "dormitory": "residential",
    "bungalow": "residential",
    # commercial
    "commercial": "commercial",
    "retail": "commercial",
    "office": "commercial",
    "supermarket": "commercial",
    "kiosk": "commercial",
    "hotel": "commercial",
    "shop": "commercial",
    "entertainment": "commercial",
    "service": "commercial",
    # industrial
    "industrial": "industrial",
    "warehouse": "industrial",
    "factory": "industrial",
    "manufacture": "industrial",
    "agricultural": "industrial",
    "barn": "industrial",
    "greenhouse": "industrial",
    "hangar": "industrial",
    # public
    "school": "public",
    "kindergarten": "public",
    "university": "public",
    "college": "public",
    "hospital": "public",
    "clinic": "public",
    "civic": "public",
    "public": "public",
    "government": "public",
    "fire_station": "public",
    "train_station": "public",
    "transportation": "public",
    "sports_centre": "public",
    "sports_hall": "public",
    "stadium": "public",
    "church": "public",
    "mosque": "public",
    "cathedral": "public",
    "chapel": "public",
    "synagogue": "public",
    "religious": "public",
    "education": "public",
    "medical": "public",
    "military": "public",
}

#: POI category (config/poi_categories.yaml) -> function group.
POI_CATEGORY_FUNCTION: dict[str, str] = {
    "office": "commercial",
    "retail_other": "commercial",
    "grocery": "commercial",
    "market": "commercial",
    "finance": "commercial",
    "food_service": "commercial",
    "hotel": "commercial",
    "fuel_auto": "commercial",
    "craft": "commercial",
    "pharmacy": "commercial",
    "industrial": "industrial",
    "education_school": "public",
    "education_kindergarten": "public",
    "education_university": "public",
    "health_hospital": "public",
    "health_clinic": "public",
    "public_institution": "public",
    "culture_leisure": "public",
    "religious": "public",
    # transit_stop deliberately ignored: a bus stop in front of a building
    # says nothing about the building's use.
}

#: Structures that are not occupiable floor space: kept out of the layer
#: entirely, so they contribute neither GFA nor footprint features.
NON_OCCUPIABLE_TAGS = frozenset(
    {"roof", "bridge", "construction", "ruins", "proposed", "razed", "demolished", "collapsed"}
)

#: landuse=<value> -> function (used for the centroid fallback).
LANDUSE_FUNCTION: dict[str, str] = {
    "industrial": "industrial",
    "commercial": "commercial",
    "retail": "commercial",
    "residential": "residential",
}


def parse_number(value: object) -> float:
    """First number in a tag string ('3', '12 m', '2,5') -> float, else NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMBER_RE.search(str(value))
    return float(match.group(1).replace(",", ".")) if match else float("nan")


def merge_osm_overture(
    osm: gpd.GeoDataFrame,
    overture: gpd.GeoDataFrame,
    *,
    iou_threshold: float,
    metric_crs: str,
) -> gpd.GeoDataFrame:
    """Union of OSM and Overture footprints with IoU-based de-duplication.

    An Overture footprint whose best IoU against any intersecting OSM
    footprint is >= ``iou_threshold`` is considered the same building and
    dropped (the OSM attributes win, per README §7). Returns a frame in
    the input CRS with normalized columns: bld_id, src, building_tag,
    levels_num, height_num, name, amenity, shop, office, geometry.
    """
    osm = osm.reset_index(drop=True)
    overture = overture.reset_index(drop=True)

    duplicate_mask = np.zeros(len(overture), dtype=bool)
    if len(osm) > 0 and len(overture) > 0:
        osm_metric = osm.to_crs(metric_crs)
        overture_metric = overture.to_crs(metric_crs)
        pairs = gpd.sjoin(
            overture_metric[["geometry"]], osm_metric[["geometry"]], predicate="intersects"
        )
        if len(pairs) > 0:
            left = overture_metric.geometry.values[pairs.index]
            right = osm_metric.geometry.values[pairs["index_right"].to_numpy()]
            inter = shapely.area(shapely.intersection(left, right))
            union = shapely.area(left) + shapely.area(right) - inter
            with np.errstate(invalid="ignore", divide="ignore"):
                iou = np.where(union > 0, inter / union, 0.0)
            best = pd.Series(iou, index=pairs.index).groupby(level=0).max()
            duplicate_mask[best.index[best >= iou_threshold]] = True

    logger.info(
        "building merge: %d OSM + %d of %d Overture (dropped %d duplicates, IoU >= %.2f)",
        len(osm),
        int((~duplicate_mask).sum()),
        len(overture),
        int(duplicate_mask.sum()),
        iou_threshold,
    )

    osm_part = gpd.GeoDataFrame(
        {
            "bld_id": "osm/" + osm["osm_id"].astype(str),
            "src": "osm",
            "building_tag": osm["building"].astype("string"),
            "levels_num": osm["building_levels"].map(parse_number),
            "height_num": osm["height"].map(parse_number),
            "name": osm.get("name", pd.Series(pd.NA, index=osm.index)).astype("string"),
            "amenity": osm.get("amenity", pd.Series(pd.NA, index=osm.index)).astype("string"),
            "shop": osm.get("shop", pd.Series(pd.NA, index=osm.index)).astype("string"),
            "office": osm.get("office", pd.Series(pd.NA, index=osm.index)).astype("string"),
        },
        geometry=osm.geometry,
        crs=osm.crs,
    )

    kept = overture.loc[~duplicate_mask]
    tag = kept["subtype"].astype("string").fillna(kept["class"].astype("string"))
    overture_part = gpd.GeoDataFrame(
        {
            "bld_id": "ovt/" + kept["overture_id"].astype(str),
            "src": "overture",
            "building_tag": tag,
            "levels_num": pd.to_numeric(kept["num_floors"], errors="coerce"),
            "height_num": pd.to_numeric(kept["height"], errors="coerce"),
            "name": pd.Series(pd.NA, index=kept.index, dtype="string"),
            "amenity": pd.Series(pd.NA, index=kept.index, dtype="string"),
            "shop": pd.Series(pd.NA, index=kept.index, dtype="string"),
            "office": pd.Series(pd.NA, index=kept.index, dtype="string"),
        },
        geometry=kept.geometry,
        crs=kept.crs,
    )
    merged = pd.concat([osm_part, overture_part], ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=osm.crs)


def classify_function(
    buildings: gpd.GeoDataFrame,
    pois: gpd.GeoDataFrame,
    landuse: gpd.GeoDataFrame,
) -> pd.Series:
    """Assign a function to each building (see module docstring for rules)."""
    function = pd.Series(pd.NA, index=buildings.index, dtype="string")

    # 1. The building's own type tag (works for OSM values and Overture subtypes).
    tag = buildings["building_tag"].astype("string").str.lower()
    function = function.mask(function.isna(), tag.map(TAG_FUNCTION))

    # 2. The building's own amenity/shop/office tags.
    has_shop_or_office = buildings["shop"].notna() | buildings["office"].notna()
    function = function.mask(function.isna() & has_shop_or_office, "commercial")
    amenity_function = buildings["amenity"].astype("string").str.lower().map(TAG_FUNCTION)
    function = function.mask(function.isna(), amenity_function)

    # 3. Majority of categorized POIs inside the footprint.
    unknown = buildings[function.isna()]
    if len(unknown) > 0 and len(pois) > 0:
        poi_groups = pois[["category", "geometry"]].copy()
        poi_groups["poi_function"] = poi_groups["category"].map(POI_CATEGORY_FUNCTION)
        poi_groups = poi_groups.dropna(subset=["poi_function"])
        joined = gpd.sjoin(
            unknown[["geometry"]], poi_groups[["poi_function", "geometry"]], predicate="contains"
        )
        if len(joined) > 0:
            counts = joined.groupby([joined.index, "poi_function"]).size().unstack(fill_value=0)
            verdict = pd.Series("commercial", index=counts.index, dtype="string")
            if "industrial" in counts.columns:
                verdict = verdict.mask(counts["industrial"] > 0, "industrial")
            if "public" in counts.columns:
                public_wins = (counts["public"] > counts.get("commercial", 0)) & (
                    counts.get("industrial", 0) == 0
                )
                verdict = verdict.mask(public_wins, "public")
            function = function.mask(function.isna(), verdict.reindex(function.index))

    # 4. Land use under the centroid.
    unknown_index = function[function.isna()].index
    if len(unknown_index) > 0 and len(landuse) > 0:
        landuse_polys = landuse[["landuse", "geometry"]].dropna(subset=["landuse"]).copy()
        landuse_polys["lu_function"] = (
            landuse_polys["landuse"].astype("string").str.lower().map(LANDUSE_FUNCTION)
        )
        landuse_polys = landuse_polys.dropna(subset=["lu_function"])
        if len(landuse_polys) > 0:
            centroids = gpd.GeoDataFrame(
                geometry=buildings.geometry.loc[unknown_index].representative_point(),
                crs=buildings.crs,
            )
            joined = gpd.sjoin(
                centroids, landuse_polys[["lu_function", "geometry"]], predicate="within"
            )
            joined = joined[~joined.index.duplicated(keep="first")]
            function.loc[joined.index] = joined["lu_function"].astype("string")

    return function.fillna("unknown")


def estimate_levels(
    buildings: gpd.GeoDataFrame,
    function: pd.Series,
    *,
    m_per_level: float,
    default_levels: dict[str, float],
) -> tuple[pd.Series, pd.Series]:
    """Floors per building and where the estimate came from (tag|height|default)."""
    from_tag = buildings["levels_num"]
    from_height = buildings["height_num"] / m_per_level
    from_default = function.map(default_levels).astype(float)

    levels = from_tag.copy()
    source = pd.Series("tag", index=buildings.index, dtype="string")
    use_height = levels.isna() & from_height.notna()
    levels = levels.mask(use_height, from_height)
    source = source.mask(use_height, "height")
    use_default = levels.isna()
    levels = levels.mask(use_default, from_default)
    source = source.mask(use_default, "default")

    levels = levels.clip(lower=1.0).round(1)
    return levels, source


def prepare_buildings(config: Config, *, force: bool = False) -> Path:
    """Merged building layer with function, floors and GFA."""
    raw_osm = config.paths.raw / RAW_OSM_FILE
    raw_overture = config.paths.raw / RAW_OVERTURE_FILE
    raw_landuse = config.paths.raw / RAW_LANDUSE_FILE
    pois_path = config.paths.interim / POIS_FILE
    buffer_path = config.paths.interim / BUFFER_FILE
    output = config.paths.interim / OUTPUT_FILE

    def build() -> None:
        boundary = read_geoparquet(buffer_path).geometry.iloc[0]
        osm = read_geoparquet(raw_osm)
        overture = read_geoparquet(raw_overture)
        landuse = read_geoparquet(raw_landuse)
        pois = read_geoparquet(pois_path)

        osm = osm[osm.geometry.intersects(boundary)]
        overture = overture[overture.geometry.intersects(boundary)]
        osm = osm.drop_duplicates(subset="osm_id", keep="first")

        merged = merge_osm_overture(
            osm,
            overture,
            iou_threshold=config.buildings.iou_threshold,
            metric_crs=config.crs.metric,
        )
        non_occupiable = (
            merged["building_tag"].astype("string").str.lower().isin(NON_OCCUPIABLE_TAGS)
        )
        if non_occupiable.any():
            logger.info(
                "dropping %d non-occupiable structures (roofs, construction, bridges, ...)",
                int(non_occupiable.sum()),
            )
            merged = merged.loc[~non_occupiable]
        function = classify_function(merged, pois, landuse)
        levels, levels_source = estimate_levels(
            merged,
            function,
            m_per_level=config.buildings.m_per_level,
            default_levels=config.buildings.default_levels,
        )
        footprint_m2 = merged.to_crs(config.crs.metric).geometry.area

        result = gpd.GeoDataFrame(
            {
                "bld_id": merged["bld_id"],
                "src": merged["src"].astype("string"),
                "building_tag": merged["building_tag"],
                "function": function,
                "levels": levels.astype(float),
                "levels_source": levels_source,
                "footprint_m2": footprint_m2.round(1),
                "gfa_m2": (footprint_m2 * levels).round(1),
                "name": merged["name"],
            },
            geometry=merged.geometry,
            crs=merged.crs,
        )
        logger.info(
            "buildings: %d footprints, total GFA %.1f km2, functions: %s",
            len(result),
            result["gfa_m2"].sum() / 1e6,
            result["function"].value_counts().to_dict(),
        )
        write_geoparquet(result, output)

    ensure_built(
        output,
        [raw_osm, raw_overture, raw_landuse, pois_path, buffer_path, config.source_path],
        build,
        force=force,
    )
    return output
