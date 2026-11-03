"""Layer schemas: the column contract between pipeline stages.

Each constant documents (and lets tests assert) what a stage writes and
what the next stage may rely on. Raw layers mirror the source as closely
as practical; interim layers are cleaned, clipped and carry only the
columns later phases need. All stored geometries use the storage CRS
(EPSG:4326); metric quantities (areas, lengths) are computed in the
metric CRS (EPSG:32643) before storage.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Raw layers (data/raw/, written by `almaty-access download`)
# ---------------------------------------------------------------------------

#: Transport network modes downloaded and prepared in Phase 1.
NETWORK_MODES: tuple[str, ...] = ("walk", "bike", "drive")

#: OSM tag keys kept as columns when extracting POIs from the PBF.
POI_TAG_KEYS: tuple[str, ...] = (
    "amenity",
    "shop",
    "office",
    "craft",
    "tourism",
    "leisure",
    "healthcare",
    "public_transport",
    "highway",
    "railway",
    "man_made",
    "industrial",
    "building",
)

#: data/raw/osm_pois.parquet — point geometry (polygon POIs → centroid).
RAW_OSM_POIS = ("osm_id", "name", *POI_TAG_KEYS, "geometry")

#: data/raw/osm_buildings.parquet — polygon geometry.
RAW_OSM_BUILDINGS = (
    "osm_id",
    "building",
    "building_levels",  # raw building:levels tag, string
    "height",  # raw height tag, string
    "name",
    "amenity",
    "shop",
    "office",
    "geometry",
)

#: data/raw/osm_landuse.parquet — polygon geometry (landuse=* and leisure=park).
RAW_OSM_LANDUSE = ("osm_id", "landuse", "leisure", "geometry")

#: data/raw/boundary_osm.parquet — level is 'city' or 'district'.
#: name is OSM's primary (Kazakh); name_ru/name_kk are language variants.
RAW_BOUNDARY = ("name", "name_ru", "name_kk", "admin_level", "level", "geometry")

#: data/raw/overture_buildings.parquet — subset of Overture building theme.
RAW_OVERTURE_BUILDINGS = (
    "overture_id",
    "subtype",
    "class",
    "height",  # meters, float, often missing
    "num_floors",  # int, often missing
    "geometry",
)

#: data/raw/population_r8_raw.parquet — Kontur hexes clipped to the area.
RAW_POPULATION_R8 = ("h3", "population", "geometry")

#: data/raw/network_{mode}_nodes.parquet / _edges.parquet (mode: walk/bike/drive)
RAW_NETWORK_NODES = ("node_id", "x", "y", "geometry")
RAW_NETWORK_EDGES = ("u", "v", "key", "length_m", "highway", "geometry")

# ---------------------------------------------------------------------------
# Interim layers (data/interim/, written by `almaty-access prepare`)
# ---------------------------------------------------------------------------

#: boundary_city.parquet (1 row), boundary_buffer.parquet (1 row)
INTERIM_BOUNDARY = ("name", "geometry")

#: boundary_districts.parquet — exactly 8 rows, one per district.
INTERIM_DISTRICTS = ("district", "name_osm", "area_km2", "geometry")

#: pois.parquet — categorized, deduplicated, clipped to the buffer.
INTERIM_POIS = ("osm_id", "name", "category", "geometry")

#: buildings.parquet — merged OSM+Overture with floors, function and GFA.
INTERIM_BUILDINGS = (
    "bld_id",
    "src",  # 'osm' | 'overture'
    "building_tag",
    "function",  # residential | commercial | industrial | public | unknown
    "levels",
    "levels_source",  # 'tag' | 'height' | 'default'
    "footprint_m2",
    "gfa_m2",
    "name",
    "geometry",
)

#: population_r8.parquet / population_r9.parquet
INTERIM_POPULATION = ("h3", "population", "geometry")

#: network_{mode}_nodes.parquet — adds elevation; _edges.parquet adds grade.
INTERIM_NETWORK_NODES = ("node_id", "x", "y", "elevation_m", "geometry")
INTERIM_NETWORK_EDGES = ("u", "v", "key", "length_m", "highway", "grade", "geometry")

#: Building functions the classifier may assign.
BUILDING_FUNCTIONS = ("residential", "commercial", "industrial", "public", "unknown")
