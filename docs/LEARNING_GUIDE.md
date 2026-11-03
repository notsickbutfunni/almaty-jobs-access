# Learning Guide

Plain-language explanations of what each phase does, why each method was
chosen, and the key formulas — one section per phase. Written so that
every part of the project can be explained and defended in an interview.

---

## Phase 0 — Setup and foundations

### What was built

A reproducible project skeleton: a Python package (`src/almaty_access/`),
a central config file, a command-line interface, a caching convention,
a synthetic sample dataset, tests, linting and continuous integration.
No analysis yet — Phase 0 is about making every later phase cheap to
build, test and re-run.

### Why a `src/` package instead of loose scripts

The code lives in `src/almaty_access/` and is *installed* into the
environment (in editable mode). Consequences:

- `import almaty_access` works the same everywhere: in tests, notebooks,
  the CLI and CI. No `sys.path` hacks, no "works only from this folder".
- The `src/` layout means tests can only import the installed package,
  never accidentally pick up files by relative path — the same way any
  other user of the repo would import it.

### Why uv and a lockfile (reproducibility)

`pyproject.toml` declares *what* the project needs (e.g. "geopandas,
version 1.0 or newer"); `uv.lock` records *exactly which* versions were
resolved, down to file hashes. Anyone running `uv sync --locked` gets
bit-for-bit the same environment — this is what "reproducible" means in
practice. The README allowed conda or uv; uv was chosen because modern
binary wheels (geopandas via pyogrio, shapely 2, pyarrow, rasterio)
install cleanly on Windows and Linux without conda, and uv is very fast
in CI. The Python version itself is pinned too (`.python-version` →
3.12).

### The config as a single source of truth

`config/config.yaml` holds every tunable number: CRS codes, H3
resolutions, time thresholds, speeds, paths, the random seed.
`almaty_access.config.load_config()` parses it into frozen (immutable)
dataclasses and **validates everything up front**: the metric CRS must
actually be projected, resolutions must be 0–15, thresholds strictly
increasing, speeds positive, the bbox well-formed. The point of failing
fast: a config typo should die with a clear message at startup, not
produce silently wrong maps three stages later.

### The two-CRS rule (the most important geo concept here)

- **EPSG:4326 (WGS 84)** stores positions in *degrees* of longitude and
  latitude. Degrees are not a unit of distance: at Almaty's latitude
  (~43°), 1° of longitude ≈ 81 km but 1° of latitude ≈ 111 km. Measuring
  distance or area in degrees is simply wrong.
- **EPSG:32643 (UTM zone 43N)** is a *projected* CRS: coordinates in
  meters on a flat plane, accurate within its 6°-wide zone (72°E–78°E —
  Almaty at ~77°E fits). All buffering, lengths and areas happen here.

Convention: **compute in 32643, store and map in 4326.** The config
validator enforces that `crs.metric` is projected and `crs.storage` is
geographic, so the two cannot be swapped by accident.

### Determinism and the random seed

Anything random (sampling, ML later) must give the same result on every
run, or results can't be reproduced or debugged. Two mechanisms:

- Code that needs randomness creates an explicit generator:
  `rng = numpy.random.default_rng(config.random_seed)`. Explicit
  generators are the modern NumPy style — no hidden global state.
- `seed_everything(seed)` additionally seeds Python's and NumPy's global
  RNGs as a safety net for third-party libraries.

The test suite proves determinism: building the sample dataset twice
must produce identical files.

### GeoParquet as the storage format

Parquet is a compressed, columnar file format — fast to read, small on
disk, and it preserves data types exactly. GeoParquet adds a geometry
column and embeds the CRS *inside the file*, unlike shapefiles (slow,
10-character column-name limit, CRS in a sidecar file that gets lost).
`io.write_geoparquet` refuses to write a table that has no CRS — a
spatial file with unknown CRS is a time bomb.

### Caching: the `make` idea with file modification times

Rule: **rebuild an output if and only if it is missing, or any of its
inputs has a newer modification time.**

```
needs_rebuild(output, inputs) = output missing
                                OR any(mtime(input) > mtime(output))
```

`io.ensure_built(output, inputs, build_fn, force=...)` wraps every
pipeline step with that rule, so re-running the pipeline after changing
one input only recomputes what depends on it. Trade-off worth knowing:
mtimes are cheap but coarse — merely touching a file triggers a rebuild
even if its content is identical (content *hashing* would avoid that but
must read every byte of possibly-gigabyte files). This is exactly the
trade-off `make` has made for 40+ years. Nanosecond mtimes
(`st_mtime_ns`) are compared, which orders even same-second writes on
filesystems that store sub-second timestamps (NTFS, ext4); on coarse
ones (FAT/exFAT external drives) equal timestamps count as "fresh" —
again the same semantics as `make`. A forced rebuild (`force=True`)
still checks that all inputs exist; nothing ever builds from missing
inputs.

### The CLI (typer)

`almaty-access` has one sub-command per pipeline stage: `download`,
`prepare`, `features`, `model`, `access`, `equity`, `report`, plus two
utilities: `info` (validate + print the config) and `sample` (regenerate
the synthetic sample). Typer generates `--help` from function signatures
and docstrings. Stages that belong to later phases already exist in the
interface — they validate the config, then exit with code 2 and a
message naming their phase. That keeps the interface stable from day
one without pretending unimplemented work exists. Exit codes follow a
small convention: **1** = configuration/user error (clean one-line
message, no traceback), **2** = stage not implemented yet, **0** =
success.

### The synthetic sample dataset

Tests and CI must run offline in under 2 minutes, so they cannot
download real data. Phase 0 ships a tiny deterministic stand-in: a bbox
in central Almaty (~6.5 × 6.7 km) as the "boundary" and 200 seeded
random points as "POIs". It exercises the real machinery (GeoParquet
IO, CRS handling, caching, CLI) end to end. In Phase 1 it is extended
with schema-faithful clips of the real layers, so tests keep matching
reality.

### Tests, lint, CI — what each one buys

- **pytest**: every behavior promised above is asserted by a test —
  config validation (one test per failure mode), IO round-trips, the
  caching rule (by manipulating file mtimes), sample determinism, CLI
  behavior. If a later refactor breaks a promise, a test names it.
- **ruff**: one fast tool doing both linting (bug-prone patterns, import
  order) and formatting (replaces black). Style stops being a decision.
- **pre-commit**: runs ruff + small hygiene hooks on each commit, plus
  `nbstripout` (strips notebook outputs — keeps diffs readable and
  avoids committing rendered data) and a large-file guard (stops raw
  geodata slipping into git).
- **GitHub Actions CI**: a fresh Linux machine runs `uv sync --locked`,
  all pre-commit hooks (so the large-file guard and notebook stripping
  are enforced server-side too, even for a contributor who never ran
  `pre-commit install` or committed with `--no-verify`), the tests and
  a CLI smoke test on every push. CI passing is the proof that the repo
  is actually reproducible from scratch, not just "works on my
  machine".

### Interview Q&A seeds

- *Why pin dependencies?* Floating versions make results
  irreproducible and CI flaky; the lockfile freezes the full dependency
  tree, and upgrades become deliberate, reviewable commits.
- *Why mtime caching instead of hashing?* O(1) stat calls vs reading
  every byte; the false-rebuild cost of a touched-but-unchanged file is
  small compared to hashing multi-GB rasters. Same choice `make` makes.
- *Why validate the CRS semantics (projected vs geographic) and not
  just the string?* Because the typical silent failure in geospatial
  work is computing "areas" in square degrees. The validator makes the
  wrong configuration unrepresentable.
- *Why is the CLI command for a future phase present but exiting with
  an error?* Interface stability: docs, Make targets and muscle memory
  don't change as phases land; and an explicit "arrives in Phase N"
  beats a silent no-op or a crash.

---

## Phase 1 — Data acquisition and cleaning

### What was built

`almaty-access download` fills `data/raw/` from five open sources
(OSM, Overture buildings, Kontur population, Copernicus DEM, official
district statistics), and `almaty-access prepare` turns raw files into
clean, clipped, documented layers in `data/interim/`: boundary (city +
8 districts + 5 km buffer), categorized POIs, a merged building layer
with floors/function/GFA, population hexes at H3 r8 and r9, a
metric-CRS DEM with slope, three networks with node elevations and edge
grades, and an OSM-completeness table. Both commands cache per step
(`--only`, `--force`). The schema of every layer is written down once,
in `src/almaty_access/schemas.py` — the contract between stages.

### Three ways to read OSM, and why we use all three

- **osmnx (Overpass API)** builds the walking/cycling/driving graphs:
  it knows which highway tags belong to which mode, simplifies chains
  of nodes into single edges, and attaches edge lengths in meters. We
  keep the largest **strongly connected** component: routing needs to
  get there *and back*; unreachable islands would later produce fake
  "no access" results.
- **Geofabrik PBF + pyosmium** extracts bulk layers (every building,
  POI and land-use polygon). One cached ~215 MB country download, then
  local extraction — far more robust than asking the public Overpass
  server for hundreds of thousands of polygons.
- For the **sample**, a small Overpass XML export is converted into a
  tiny `.osm.pbf`, so the exact same extractor code runs in tests.

Lesson learned the hard way: the main Overpass instance was overloaded
during development (connection timeouts, 502s), so the code targets a
mirror and has a fallback chain. Public infrastructure fails;
pipelines must expect it.

### Boundary cleaning (why each step exists)

- `make_valid` repairs self-intersections — one invalid polygon crashes
  later overlays.
- **Sliver removal** drops polygon fragments under 1 ha left over from
  topology fixes.
- The **5 km buffer** fights the *edge effect*: someone at the city
  border can reach jobs outside it; clipping hard at the boundary would
  understate their access. Buffering happens in EPSG:32643, because "5
  km" is meaningless in degrees.
- District names are matched against OSM's Kazakh (`name`), Russian
  (`name:ru`) and Kazakh-variant tags, case-insensitively, and the step
  fails loudly listing what it found — never silently proceeds with 7
  districts.

### POIs: an ordered rule table instead of code

`config/poi_categories.yaml` maps raw tags to ~20 categories; the FIRST
matching rule wins, so specific rules (shop=supermarket → grocery) sit
above catch-alls (shop=* → retail_other). Because it is config, not
code, changing the category scheme (a Phase 3 sensitivity test) touches
no Python. Unmapped POIs are counted and dropped, never guessed.

### Buildings: merge, floors, function, GFA

- **Merge**: OSM and Overture both have footprints; where they overlap
  with **IoU ≥ 0.5** they are the same building and the OSM copy wins
  (richer tags). IoU = area(A∩B) / area(A∪B) — scale-free, 1.0 for
  identical shapes, 1/3 for half-overlapping equal squares, so 0.5
  separates "same building drawn twice" from "neighbors touching".
- **Function** (residential/commercial/industrial/public/unknown) is a
  priority ladder: the building's own type tag → its own shop/office/
  amenity tags → a majority vote of categorized POIs inside it → the
  land-use polygon under it → unknown. Honest `unknown` beats a wrong
  guess; the share of unknowns is itself a data-quality signal.
- **Floors**: `building:levels` tag → height/3 m → per-function default
  from config. The source of each estimate is stored (`levels_source`),
  so Phase 3 can test how sensitive the job model is to the defaults.
- **GFA** (gross floor area) = footprint m² × floors — the core
  job-capacity proxy. Area must be computed in EPSG:32643; in degrees
  it would be nonsense.

### Population: Kontur r8, dasymetric r9

Kontur already publishes population per H3 r8 hex (built from GHS-POP,
building footprints, OSM — CC BY). For r9 we split each parent hex's
population over its 7 children **proportionally to residential GFA**:

    pop(child) = pop(parent) × GFA_res(child) / Σ GFA_res(children)

with an equal split when a parent has no residential GFA at all. This
is *dasymetric* refinement — redistribute a known total using an
ancillary variable — and it is **mass-preserving**: the r9 layer sums
exactly to the r8 layer, which a test asserts. It is also a miniature
of the Phase 3 job model (same idea, different variable).

### Terrain

Copernicus GLO-30 (30 m) is read straight from public cloud COGs —
only the needed window ever downloads. Slope is computed on the metric
grid, where pixels and heights are both meters:

    slope° = atan( √((dz/dx)² + (dz/dy)²) ) · 180/π

Each network edge gets a **directed grade** = (elev(v) − elev(u)) /
length — positive uphill, negative downhill — ready for Tobler's hiking
function in Phase 4.

**The DSM trap (found by review, worth telling in interviews).** GLO-30
is a *surface* model: on a building roof it reads the roof, not the
street. Sampled naively, nodes under tall buildings got elevations up to
~70 m above ground and short edges got absurd grades (13.8% of walk
edges steeper than 10% in flat central Almaly, with |grade| correlating
with edge length — the signature of sampling noise, not terrain). Fix:
sample from a **moving-minimum filtered** DEM (3×3 window ≈ 80 m): the
minimum in a neighborhood approximates the ground between buildings.
That brought mean |grade| from 4.7% to 2.9% and >10% edges down to
6.8%; a hard cap at |0.3| removes the rest. Residual noise from 27 m
pixels on very short edges is accepted and documented — Tobler's
function is smooth, so the effect on travel times is bounded.

### District statistics: the two meanings of "employment"

The research pass found something better than the README hoped: the
official Taldau API serves, per district, both
**employed residents** (household survey — where workers *live*) and
**payroll jobs** (enterprise reports — where jobs *are located*). They
differ hugely (central Almaly: ~250k jobs vs ~132k employed residents;
bedroom-district Alatau: ~63k vs ~187k) — exactly the difference between
trip *production* and trip *attraction* in transport modeling. The jobs
series is the Phase 3 training target; mixing the two up would train
the model on the wrong quantity. The downloader fetches both, labels
them, and never overwrites a manually curated file.

### OSM completeness

Per district, what share of merged footprints came from OSM vs
Overture-only? A district where OSM contributed little also has fewer
tags and POIs — its model features are under-observed. The flag table
(`completeness.csv`) travels with the data so Phase 3 can show it next
to the predictions.

### Interview Q&A seeds

- *Why IoU and not just "intersects" for de-duplication?* Neighboring
  buildings touch (intersect) all the time; IoU ≥ 0.5 demands the
  shapes largely coincide.
- *Why `representative_point()` instead of `centroid` for
  point-in-polygon?* A concave building's centroid can fall outside the
  building; representative_point is guaranteed inside.
- *Why keep the largest strongly connected component?* With one-way
  streets, weak connectivity isn't enough: you could reach a node but
  never return; SCC guarantees round-trip routability.
- *Why is the r9 population layer "safe" to publish?* It is derived
  from open aggregates (Kontur + building geometry), contains no
  personal data, and conserves published totals.
- *Why does `prepare` fail when a district name is missing instead of
  warning?* Downstream everything joins on 8 districts; a silent 7 would
  corrupt every later result. Fail at the first step where the problem
  is visible and fixable.
