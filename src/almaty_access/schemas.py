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

#: Tag keys stored as POI columns but NOT used to decide whether an
#: element is a POI (a bare station=* without railway=* is mapping noise).
#: station=subway separates metro stations from heavy-rail ones.
POI_EXTRA_KEYS: tuple[str, ...] = ("station",)

#: data/raw/osm_pois.parquet — point geometry (polygon POIs → representative point).
RAW_OSM_POIS = ("osm_id", "name", *POI_TAG_KEYS, *POI_EXTRA_KEYS, "geometry")

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
#: railway/station ride along for the Phase 2 transit features
#: (railway=station + station=subway identifies metro stations).
INTERIM_POIS = ("osm_id", "name", "category", "railway", "station", "geometry")

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

# ---------------------------------------------------------------------------
# Processed layers (data/processed/, written by `almaty-access features`)
# ---------------------------------------------------------------------------

#: grid_r{res}.parquet — the study grid, one row per hex whose centroid
#: lies inside the buffered boundary (∩ area.bbox when one is set).
#: district is the largest-overlap district (NA in the buffer ring);
#: district_share is that district's share of the hex area.
PROCESSED_GRID = ("h3", "res", "district", "district_share", "in_city", "geometry")

#: features_r{res}.parquet — plain parquet, NO geometry (join grid_r{res}
#: on h3 for maps). First the identifier columns, then only feature
#: columns, each named with a group prefix.
FEATURE_ID_COLUMNS = ("h3", "res", "district", "in_city")
FEATURE_PREFIXES = ("bf_", "poi_", "lu_", "net_", "tr_", "pop_", "cent_", "ter_", "ctx_")

#: subcenters.parquet — POI-cluster job sub-centres (point geometry).
#: in_city=False marks a buffer-zone cluster (satellite town): eligible
#: as a distance target (real jobs for edge hexes) but not an Almaty
#: sub-centre in the administrative sense.
PROCESSED_SUBCENTERS = ("subcenter_id", "n_pois", "in_city", "geometry")

#: jobs_r{res}.parquet — Phase 3 job estimates per hex (plain parquet,
#: join the grid on h3 for maps). Buffer-ring hexes carry 0 jobs:
#: official totals exist only for the 8 districts. jobs_rf is the
#: central estimate (median across trees, district-rescaled); p10/p90
#: bound it from the forest spread.
PROCESSED_JOBS = (
    "h3",
    "res",
    "district",
    "in_city",
    "jobs_baseline",
    "jobs_rf",
    "jobs_rf_p10",
    "jobs_rf_p90",
)
