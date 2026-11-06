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

#: OSM route-relation types the Phase 4 transit fallback understands.
#: Almaty has buses, trolleybuses and one metro line; tram service ended
#: in 2015 (tram relations elsewhere in the country PBF are clipped away).
TRANSIT_MODES: tuple[str, ...] = ("bus", "trolleybus", "subway")

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

#: transit_stops.parquet — distinct stop/platform nodes of kept routes
#: (point geometry, clipped to the study area).
INTERIM_TRANSIT_STOPS = ("stop_id", "name", "geometry")

#: transit_routes.parquet — one row per kept route relation (plain parquet).
#: headway_min comes from the relation's ``interval`` tag when parseable
#: (headway_source = "interval_tag"), otherwise from the assumed per-mode
#: headway in config (headway_source = "assumed").
INTERIM_TRANSIT_ROUTES = (
    "route_id",
    "mode",
    "ref",
    "name",
    "headway_min",
    "headway_source",
    "n_stops",
)

#: transit_legs.parquet — hops between consecutive kept stops of a route
#: (plain parquet). ``seg`` increments where the stop sequence was broken
#: (a stop dropped by clipping); legs never span a break. ``dist_m`` is the
#: straight-line metric distance (circuity is applied at graph-build time).
INTERIM_TRANSIT_LEGS = ("route_id", "seg", "seq", "from_stop", "to_stop", "dist_m")

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

# ---------------------------------------------------------------------------
# Processed layers (data/processed/, written by `almaty-access access`)
# ---------------------------------------------------------------------------

#: tt_{mode}_r{res}.parquet — long-format travel-time matrices (plain
#: parquet): walk at r8 and r9, bike/transit/car at r8 only. Only pairs
#: within max(travel.thresholds_min) minutes are stored; minutes include
#: snapping of both endpoints, walked at the flat walking speed.
PROCESSED_TT = ("origin_h3", "dest_h3", "minutes")

#: access_r{res}.parquet — one row per grid hex (plain parquet, join the
#: grid on h3 for maps). After the identifier columns, every column name
#: starts with one of these prefixes:
#:   acc_     reachable opportunities: acc_{opp}_{mode}_{T} (cumulative,
#:            threshold T minutes) and acc_jobs_{mode}_grav (gravity).
#:            Hex-aggregated: times run centroid-to-centroid, so at r8 a
#:            facility can be "12 min away" (near_) yet its HEX centroid
#:            beyond 15 min — use r9 for short-threshold service claims.
#:   accpct_  acc_jobs_{mode}_{T} as % of all city jobs at that resolution
#:   near_    minutes to the nearest facility POINT, walking:
#:            near_{facility}_walk_min (not hex-aggregated)
#:   sfca_    two-step floating catchment ratio: sfca_{facility}_walk
#:   transit_ transit_routes_cnt (distinct routes stopping in the hex)
ACCESS_ID_COLUMNS = ("h3", "res", "district", "in_city")
ACCESS_PREFIXES = ("acc_", "accpct_", "near_", "sfca_", "transit_")

# ---------------------------------------------------------------------------
# Processed layers (data/processed/, written by `almaty-access equity`)
# ---------------------------------------------------------------------------

#: Access columns where MORE is better; only these get Gini / Lorenz /
#: Palma, LISA, deserts and bivariate classes. Minutes-to-nearest
#: (near_*) run the other way and get threshold-share tables instead.
EQUITY_BENEFIT_PREFIXES = ("acc_", "accpct_", "sfca_")

#: Population weightings every Phase 5 statistic is produced under:
#: the Kontur raster as it is, and the raster rescaled so district
#: totals follow the district shares of employed residents in the
#: statistics file (DECISIONS D-058). The second is absent when no such
#: series exists.
EQUITY_WEIGHTINGS = ("raster", "calibrated")

#: equity_r{res}.parquet (raster weights) and
#: equity_calibrated_r{res}.parquet (calibrated weights) — one row per
#: grid hex, identical layout (plain parquet, join the grid on h3 for
#: maps). ``population`` is the resident count under that weighting;
#: ``inhabited`` marks in-city hexes at or above equity.min_population
#: on the RASTER (one domain for both files); ``point_mass`` flags a
#: fine-resolution hex that received (almost) all residents of its
#: coarse parent although its siblings are built up too — an artifact
#: of the residential-floor-area split, not a real concentration.
#: After the identifier columns every column starts with one of:
#:   modegap_  modegap_transit_car_{T}: transit / car job access ratio
#:             (low = car-dependent); NaN where car access is zero
#:   lisa_     LISA cluster of a headline measure: HH (hot spot), LL
#:             (cold spot), HL, LH, ns (not significant), island (no
#:             inhabited neighbour); NA outside the inhabited domain
#:   lisap_    its tie-aware two-sided pseudo p-value
#:   desert_   True for a dense hex among the worst served: population
#:             above its hex quantile AND access at or below the
#:             population-weighted access quantile (D-055)
#:   biv_      3x3 bivariate class "{population tercile}-{access
#:             tercile}", 1 = low ... 3 = high; "3-1" = a dense hex
#:             among the worst-served third of residents
#: The calibrated file of the FINE resolution additionally carries the
#: 2SFCA values recomputed with calibrated demand (sfca_*), because
#: those differ from the access table's. (2SFCA is analysed at the fine
#: resolution only, so the coarse calibrated file has no such columns.)
EQUITY_ID_COLUMNS = ("h3", "res", "district", "in_city", "population", "inhabited", "point_mass")
EQUITY_PREFIXES = ("modegap_", "lisa_", "lisap_", "desert_", "biv_")
EQUITY_CALIBRATED_VALUE_PREFIXES = ("sfca_",)

#: The long tables below share the leading ``res`` and ``weights``
#: columns; ``weights`` is one of EQUITY_WEIGHTINGS.

#: equity_summary.parquet — population-weighted distribution of the
#: benefit measures over inhabited hexes: job measures at the analysis
#: resolution, everything at the fine one (service measures are
#: granularity noise on coarse hexes, D-049). ``palma`` is inf when the
#: bottom 40% of residents hold nothing, NaN when the total is zero.
PROCESSED_EQUITY_SUMMARY = (
    "res",
    "weights",
    "measure",
    "n_hexes",
    "population",
    "mean",
    "median",
    "p10",
    "p90",
    "gini",
    "palma",
    "pop_share_zero",
)

#: equity_lorenz.parquet — Lorenz curve of each headline measure on a
#: regular population grid (exact: the curve is piecewise linear).
PROCESSED_EQUITY_LORENZ = ("res", "weights", "measure", "pop_share", "access_share")

#: equity_districts.parquet — population-weighted mean / median of each
#: headline measure per district; rank 1 = highest weighted mean.
PROCESSED_EQUITY_DISTRICTS = (
    "res",
    "weights",
    "measure",
    "district",
    "population",
    "mean",
    "median",
    "rank",
)

#: equity_deserts.parquet — access-desert hexes per headline measure,
#: rank 1 = most residents affected under that weighting.
PROCESSED_EQUITY_DESERTS = (
    "res",
    "weights",
    "measure",
    "rank",
    "h3",
    "district",
    "population",
    "value",
    "point_mass",
)

#: equity_nearest.parquet — share of residents within T walking minutes
#: of the nearest facility (unreachable / off-network counts as beyond);
#: ``district`` is "ALL" for the city-wide rows.
PROCESSED_EQUITY_NEAREST = (
    "res",
    "weights",
    "district",
    "facility",
    "threshold_min",
    "pop_share_within",
)

# ---------------------------------------------------------------------------
# Processed layers (data/processed/, written by `almaty-access scenarios`)
# ---------------------------------------------------------------------------

#: Feature kinds a scenario GeoJSON may contain (property ``kind``):
#: ``line`` (a new or extending transit line, LineString), ``stop``
#: (an explicit station of a named line, Point) and ``bike_boost``
#: (a street corridor whose bike edges get a higher flat speed,
#: LineString).
SCENARIO_FEATURE_KINDS = ("line", "stop", "bike_boost")

#: First synthetic identifier for scenario stops and routes: far above
#: any OSM node/relation id (~1e10 in 2026), so merged tables can never
#: collide with real ids.
SCENARIO_ID_BASE = 10**12

#: scenario_{name}_r{res}.parquet — one row per grid hex (plain parquet,
#: join the grid on h3 for maps), at the analysis resolution. After the
#: identifier columns, every affected job-access column of the access
#: table appears three times: ``{col}_base`` (the stored baseline),
#: ``{col}_scen`` (recomputed with the scenario network) and
#: ``{col}_delta`` (scen - base, never negative: a scenario only adds
#: arcs or speeds edges up, and rounding is monotone).
SCENARIO_ID_COLUMNS = ("h3", "res", "district", "in_city")
SCENARIO_VALUE_SUFFIXES = ("_base", "_scen", "_delta")

#: scenarios_summary.parquet — population-weighted distribution of each
#: affected relative measure (accpct_*/grav) over inhabited hexes,
#: before and after, under both population weightings (EQUITY_WEIGHTINGS).
PROCESSED_SCENARIOS_SUMMARY = (
    "scenario",
    "res",
    "weights",
    "measure",
    "population",
    "mean_base",
    "mean_scen",
    "median_base",
    "median_scen",
    "gini_base",
    "gini_scen",
    "gini_delta",
)

#: scenarios_gains.parquet — residents whose absolute job access at the
#: headline threshold grows by MORE than each configured cutoff
#: (README Phase 6: "population gaining > X jobs").
PROCESSED_SCENARIOS_GAINS = (
    "scenario",
    "res",
    "weights",
    "measure",
    "cutoff_jobs",
    "residents_gaining",
    "share_gaining",
)
