# Data directory

| Folder | Contents | In git? |
|---|---|---|
| `raw/` | Original downloads, never modified by hand | No (git-ignored) |
| `interim/` | Cleaned, clipped, reprojected layers (GeoParquet) | No (git-ignored) |
| `processed/` | Hex-level feature tables and model outputs | No (git-ignored) |
| `sample/` | Small sample dataset for tests, CI and the demo | **Yes** (committed) |

Only open data enters this project. Internal work data (Visum models,
zone matrices, internal surveys) must never appear here or anywhere else
in the repository (README §13).

## Registered datasets

Every file placed in `raw/` gets a row here: source, download date,
version and licence. Filled in from Phase 1 on.

| File | Source | Downloaded | Version | Licence |
|---|---|---|---|---|
| *(none yet — Phase 1)* | | | | |

## Sample dataset (`sample/`)

Currently **synthetic** data, generated deterministically from
`config/config.yaml` (bbox in central Almaty + seeded random points):

```
uv run almaty-access sample
```

| File | Schema | CRS |
|---|---|---|
| `boundary.parquet` | `name` (str), `geometry` (1 polygon = sample bbox) | EPSG:4326 |
| `poi_points.parquet` | `category` (str), `weight` (float), `geometry` (points) | EPSG:4326 |

The synthetic sample is released under **CC0** (it contains no upstream
data). In Phase 1 it is extended with schema-faithful clips of the real
layers for one district; those files will carry their upstream licences
(e.g. ODbL for OSM-derived layers, with attribution).

## Licence notes for upstream sources

- **OpenStreetMap** (ODbL): requires attribution ("© OpenStreetMap
  contributors") and share-alike for derived databases.
- **Overture Maps / Microsoft Building Footprints** (ODbL / CDLA-type).
- **Kontur / GHSL / WorldPop population** (CC BY): requires attribution.
- Exact terms are recorded per file in the table above when downloaded.
