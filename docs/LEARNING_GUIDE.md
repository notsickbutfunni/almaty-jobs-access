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
