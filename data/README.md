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

## Equity outputs (Phase 5)

Written by `almaty-access equity` into `processed/`. All statistics
cover **inhabited city hexes** and weight them by residents. Because the
population raster is skewed between districts, everything is produced
under **two weightings**: `raster` (Kontur as prepared in Phase 1) and
`calibrated` (hex residents rescaled so district totals follow the
district shares of employed residents; 2SFCA is recomputed with the
calibrated residents as demand). Long tables carry a `weights` column;
there is one per-hex file per weighting. Transit figures inherit the
approximate lower-bound caveat, and the report states how much of the
"transit" layer is really walking.

| File | Contents |
|---|---|
| `equity_r8.parquet`, `equity_r9.parquet` | Per-hex layers on the raster weights: residents, `inhabited`, the `point_mass` flag (fine hexes that received a whole coarse parent's residents), transit/car mode gap (r8), and for each headline measure the LISA cluster + p-value, desert flag and 3x3 bivariate class (column conventions in `schemas.py`) |
| `equity_calibrated_r8.parquet`, `equity_calibrated_r9.parquet` | The same layers under the calibrated weights; the r9 file also carries the 2SFCA values recomputed with calibrated demand (`sfca_*` — 2SFCA is analysed at r9 only). Both are absent when no residence-based series exists |
| `equity_summary.parquet` | Population-weighted mean, median, 10th / 90th percentile (`p10`, `p90`), Gini, Palma and zero-share per weighting: job measures at r8, every benefit measure at r9 |
| `equity_lorenz.parquet` | Lorenz curves of the headline measures (101 points each) |
| `equity_districts.parquet` | Population-weighted mean / median per district and the district ranking |
| `equity_deserts.parquet` | Access-desert hexes per headline measure and weighting, ranked by residents, with the `point_mass` flag |
| `equity_nearest.parquet` | Share of residents within 10 / 15 / 20 / 30 walking minutes of the nearest facility, city-wide (`district = "ALL"`) and per district |
| `equity_report.json` | Parameters, caveats (generated from the data), the calibration (source quoted verbatim, `synthetic` flag, year, district factors), one result block per weighting and resolution (cutoff shares, Moran / LISA with FDR counts and the `borderline` hexes within Monte Carlo error of alpha, deserts, mode gap, transit coverage, nearest-facility summary, centre gradient), a raster-vs-calibrated `comparison`, and the one-sentence result |

### Scenario outputs (`almaty-access scenarios`, Phase 6)

One scenario per GeoJSON file in `config/scenarios/` (ILLUSTRATIVE
alignments anchored to announced projects — provenance and source links
inside each file). The stage rebuilds only the affected mode's network
and compares against the stored access tables; deltas are never
negative by construction. Transit deltas inherit the approximate
lower-bound caveat (D-044): gains are overstated where the baseline
misses real routes.

| File | Contents |
|---|---|
| `scenario_{name}_r8.parquet` | Per-hex baseline / scenario / delta (`{col}_base`, `{col}_scen`, `{col}_delta`) for every job-access column of the affected mode(s) — the Δ-jobs map data |
| `scenarios_summary.parquet` | Population-weighted mean / median / Gini before and after per scenario, measure and weighting (raster + calibrated) |
| `scenarios_gains.parquet` | Residents gaining more than each configured cutoff of jobs at the headline threshold, per scenario and weighting |
| `scenarios_report.json` | Parameters, per-scenario network statistics (new stops/legs/routes, attachments, boosted edge-km), matrix statistics, results per weighting, generated caveats and a one-sentence result per scenario |

### Dashboard bundle (`almaty-access export`, Phase 7 — COMMITTED)

`data/app/` is the one exception to "processed data never goes into
git" (D-069): a curated copy of exactly the files the Streamlit
dashboard reads — the r8 access / jobs / equity / scenario tables, the
four r8 travel-time matrices, the long equity tables and the report
JSONs, ~7 MB in total — plus a deterministic `manifest.json` (names and
byte sizes, no timestamps). Committing it lets Streamlit Community
Cloud and any clone serve the dashboard with zero extra infrastructure;
everything inside is derived from the open sources below and inherits
their caveats (approximate lower-bound transit, free-flow car,
modelled jobs). `docs/webmap/index.html` (the static map GitHub Pages
serves) is committed for the same reason. Refresh both with
`almaty-access export` after a pipeline run; the sample equivalents
(`data/sample/app/`, `data/sample/webmap/`) stay git-ignored.

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
