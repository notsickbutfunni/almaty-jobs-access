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
