# Data directory

| Folder | Contents | In git? |
|---|---|---|
| `raw/` | Original downloads, never modified by hand | No (git-ignored) |
| `interim/` | Cleaned, clipped, reprojected layers (GeoParquet) | No (git-ignored) |
| `processed/` | Hex-level feature tables and model outputs | No (git-ignored) |
| `sample/raw/` | Small REAL data clip for tests, CI and the demo | **Yes** (committed) |
| `sample/interim|processed/` | Generated from the sample by `prepare` | No (git-ignored) |

Only open data enters this project. Internal work data (Visum models,
zone matrices, internal surveys) must never appear here or anywhere else
in the repository (README §13).

## Registered datasets (verified 2026-11-03)

Written by `almaty-access download` into `raw/`:

| File | Source | Version | Licence |
|---|---|---|---|
| `boundary_osm.parquet` | OSM via Nominatim/Overpass (city relation 2465058, districts admin_level=6) | live | ODbL |
| `osm_extract.osm.pbf` | Geofabrik `asia/kazakhstan-latest.osm.pbf` (~215 MB); bbox mode: Overpass export | daily | ODbL |
| `osm_buildings/pois/landuse.parquet` | extracted from the PBF (pyosmium) | = PBF | ODbL |
| `network_{walk,bike,drive}_*.parquet` | OSM via osmnx 2.1 (Overpass) | live | ODbL |
| `overture_buildings.parquet` | Overture Maps buildings theme, release 2026-10-21.0, via `overturemaps` | 2026-10-21.0 | ODbL |
| `population_r8_raw.parquet` | Kontur Population Kazakhstan, H3 r8 ([HDX](https://data.humdata.org/dataset/kontur-population-kazakhstan)) | 2023-11-01 | CC BY (Kontur) |
| `dem.tif` | Copernicus GLO-30 COGs, AWS Open Data (tiles N43_E076, N43_E077) | GLO-30 | free, attribution required |
| `district_stats.csv` | Bureau of National Statistics (Taldau API / stat.gov.kz) | per file | free with attribution to BNS |

## District statistics (`district_stats.csv`)

Long format: `district,year,month,indicator,value,unit,source,notes`.
`almaty-access download --only stats` fetches two verified indicators
from the Taldau API automatically when the file is absent (an existing
file is never overwritten):

- **`payroll_jobs_located`** (Taldau indicator 18935544, quarterly
  2016–2026): payroll employees incl. small enterprises, counted at the
  district where the enterprise reports → this is **jobs located in the
  district**, the Phase 3 training target. Excludes individual
  entrepreneurs and the self-employed.
- **`employed_residents`** (indicator 702840, annual 2019–2025): from
  the labour-force household survey → **residents of the district who
  are employed anywhere**. NOT the same thing: central Almaly has ~250k
  jobs but ~132k employed residents; Alatau ~63k jobs vs ~187k residents.

Manual fallback: stat.gov.kz → region Almaty → «Динамические таблицы» /
«Электронные таблицы» (see DECISIONS.md D-016 for exact tables), or the
Taldau UI (`taldau.stat.gov.kz/ru/NewIndex/GetIndex/18935544`) which has
an Excel export.

## Sample dataset (`sample/raw/`, committed)

Real data for a ~1.7 × 2.2 km bbox in the Almaly core, rebuilt with:

```
uv run almaty-access sample          # needs network
uv run almaty-access prepare -c config/config.sample.yaml
```

Same file names and schemas as the real `raw/` (see
`src/almaty_access/schemas.py`), so tests exercise the actual pipeline
code offline. Exception: `district_stats.csv` in the sample is
**synthetic** (`source=synthetic-for-tests`) — real statistics are never
fabricated. OSM-derived sample files are ODbL, the population clip CC BY
(Kontur), the DEM clip Copernicus (attribution below).

## Processed outputs (Phase 2)

Written by `almaty-access features` into `processed/` (git-ignored;
regenerate with one command). All derived purely from the layers above,
so the same licences apply (ODbL for OSM-derived content):

| File | Contents |
|---|---|
| `grid_r8.parquet`, `grid_r9.parquet` | Study grid: hex polygons, district (largest overlap), `district_share`, `in_city` |
| `features_r8.parquet`, `features_r9.parquet` | One row per hex, identical schema at both resolutions (plain parquet; join the grid on `h3` for maps; column list and count in `features_report.json`) |
| `subcenters.parquet` | Job sub-centres found by clustering business POIs (`in_city=False` marks buffer-zone satellite clusters) |
| `features_report.json` | What was log-transformed, what was pruned as a near-duplicate and why, sub-centre count, seeds |
| `jobs_r8.parquet`, `jobs_r9.parquet` | Phase 3 job estimates per hex: Model A baseline + random-forest P10/P50/P90, district totals conserved (buffer hexes carry 0) |
| `model_report.json` | Targets and years used, LODO comparison of both models, permutation importances, jobs-per-resident ratios, top-20 hexes |

## Transit network (Phase 4, interim)

Written by `almaty-access prepare --only transit` into `interim/` from
the already-downloaded PBF (OSM route relations — **no public GTFS
exists for Almaty**, so this layer is the documented approximation):

| File | Contents |
|---|---|
| `transit_stops.parquet` | Stop/platform nodes of kept routes, clipped to the study area |
| `transit_routes.parquet` | One row per kept `route=bus\|trolleybus\|subway` relation: mode, ref, headway and its provenance (`headway_source`: the OSM `interval` tag where parseable — 2 of 55 city routes — else `assumed` from config) |
| `transit_legs.parquet` | Hops between consecutive kept stops (straight-line metres; circuity and speed are applied at graph-build time) |
| `transit_extract.json` | Extraction honesty counts: relations without usable stops, routes clipped away, legs dropped as data errors |

## Accessibility outputs (Phase 4)

Written by `almaty-access access` into `processed/`. **Transit values
are approximate transit access** (assumed speeds/headways, see
`access.transit` in `config.yaml` and DECISIONS D-044/D-047):

| File | Contents |
|---|---|
| `tt_walk_r8.parquet`, `tt_walk_r9.parquet`, `tt_bike_r8.parquet`, `tt_transit_r8.parquet`, `tt_car_r8.parquet` | Long travel-time matrices (`origin_h3, dest_h3, minutes`), pairs within 60 min only; snapping of both endpoints included as walk time |
| `nearest_r8.parquet`, `nearest_r9.parquet` | Walking minutes to the nearest school / kindergarten / clinic / grocery / park / metro per hex |
| `access_r8.parquet`, `access_r9.parquet` | Per-hex accessibility: cumulative jobs by mode x threshold, gravity access, % of city jobs, service counts on foot, 2SFCA per 1,000 residents, routes-per-hex (column conventions in `schemas.py`; note `near_*` measures to the facility point while `acc_*` is hex-aggregated — use r9 for short-threshold service claims) |
| `tt_stats.json`, `access_report.json` | Snap statistics per mode, assumed-parameter echo, transit network counts, coverage, gravity half-life sensitivity, 10 spot-check OD pairs |

## Attribution

- **OpenStreetMap** (and OSM-derived layers, Overture buildings):
  © OpenStreetMap contributors, ODbL. Overture: "Overture Maps
  Foundation, overturemaps.org".
- **Kontur Population**: © Kontur, CC BY.
- **Copernicus GLO-30 DEM**: © DLR e.V. 2010-2014 and © Airbus Defence
  and Space GmbH 2014-2018 provided under COPERNICUS by the European
  Union and ESA; all rights reserved.
- **Official statistics**: Bureau of National Statistics of the Agency
  for Strategic Planning and Reforms of the Republic of Kazakhstan
  (stat.gov.kz / taldau.stat.gov.kz).
