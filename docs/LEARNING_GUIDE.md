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

---

# Phase 2 - H3 grid and feature engineering

**What was built.** `almaty-access features` turns the cleaned layers
into two tidy tables - one row per hex at H3 resolution 8 (analysis,
~0.74 km2) and 9 (fine, ~0.1 km2) - with 86 features before pruning,
68 after, zero missing values. Everything a model may ask about a hex:
how much floor space and of what kind, what services operate there, how
the street network runs through it, how far the centre, transit and
job sub-centres are, how steep it is, and what its neighbourhood looks
like. `data/processed/features_report.json` records every transform and
every dropped column.

## The grid, and why it is bigger than the city

Hexes enter the grid when their **centroid** lies inside the city
boundary buffered by 5 km. The buffer ring (917 of 1,846 r8 cells)
carries features but no district: it exists so that neighbourhood
context - and, in Phase 4, travel times - do not pretend the world ends
at the administrative edge. A hex on the boundary is assigned to the
district holding the largest share of its area, and that share is
stored (`district_share`), because Phase 3 must aggregate hex estimates
back to district totals honestly.

## Splitting buildings between hexes

At r9 a hex is ~360 m across; a bazaar hall or factory can be nearly
hex-sized. Mass quantities (footprint m2, GFA m2) are therefore split
**area-weighted**: each hex receives the share of the building that
geometrically lies inside it (computed in EPSG:32643). Indivisible
facts (the building count, floor statistics) follow the representative
point. Rule of thumb worth remembering: *split what is additive, vote
with what is not.*

## Shannon entropy - measuring mixed use

For category counts in a hex, with p_i the share of category i:

    H = -sum_i p_i * ln(p_i)

H = 0 means a single-purpose hex (only pharmacies, say); ln(20) ~ 3.0
would be all twenty categories equally present. Mixed-use places are
where consumer-service jobs cluster, which is why the README wants
diversity as a feature, not just counts.

## Betweenness centrality - finding the through-streets

A node's betweenness is the share of shortest paths (between all node
pairs) passing through it. Retail follows passing trade, so corridors
with high betweenness are job locations even when the surrounding
density is modest. Exact computation is O(V*E) - about 2.8 billion
operations here - so we use Brandes sampling: shortest-path trees from
k=500 random sources (seeded, from config) give an unbiased estimate
in about two minutes. *Interview line: betweenness is a model of
through-movement potential, not observed traffic - and sampling noise
is part of the feature, which is one more reason tree models (Phase 3)
beat linear ones on such inputs.*

## Sub-centres with DBSCAN

"Distance to the nearest major sub-centre" needs sub-centres first.
DBSCAN over business POIs (office, finance, commerce) in metric
coordinates grows clusters from density (eps 400 m, >= 40 POIs) and
labels stragglers as noise - no cluster count chosen in advance, which
is the honest setting when you do not know how polycentric the city is.
The cluster nearest the configured historic centre is the centre
itself, so it is excluded; the five survivors (389/247/147/138/91 POIs)
are the sub-centres. On the tiny sample everything is one downtown
cluster, so the distance falls back to the centre distance - the
monocentric degenerate case, stated in the report rather than hidden.

## Why log1p, and the 3-point correlation trap

Counts and areas are heavy-tailed: most hexes hold little, a few hold
enormous mass. A linear scale would let a handful of hexes dominate any
distance-based computation, so the config marks count/area columns for
log1p (log of 1+x: defined at zero, gentle on small values). The
feature tables keep RAW values - transforms are applied at model time,
and the report lists the exact columns, so nothing is baked in twice.

Near-duplicates (|r| > 0.95 on the log view) are pruned with the
earlier of two expendable columns kept. The first sample run taught
lesson one: with only 3 cells, pruning "removed" 52 of 87 features -
through three points almost any two monotone series correlate near 1.
Hence MIN_CELLS_FOR_PRUNING = 30. *Correlation needs observations
before it means anything.* The first CITY run taught lesson two: a
purely mechanical rule happily deleted the README's core features -
total GFA lost to the building count (r = 0.96, they really do move
together in a mid-rise city), and mean elevation lost to road distance
(r = 0.97 only because Almaty's grid thins out exactly where the
mountains start - a geographic coincidence, not a duplicate). Hence
PROTECTED_FEATURES: the named core columns can never be dropped, and a
highly correlated pair of protected columns is *recorded* in the
report's `high_corr_kept` instead. Documented collinearity beats
silently missing semantics - and the random forest in Phase 3 does not
mind collinear inputs. The drops that DO happen have mechanical
causes: densities duplicate counts exactly (fixed hex area), k-ring
means duplicate sums (disk sizes are nearly constant), and
tr_stop_cnt duplicates poi_cnt_transit_stop (the same physical stops
counted by two blocks).

## Moran's I - proving the structure is spatial

    I = (n / S0) * sum_ij( w_ij * z_i * z_j ) / sum_i( z_i^2 )

with z the demeaned values, w_ij = 1 for H3 ring-1 neighbours
(row-standardized), S0 the weight sum. I ~ +1 means similar values
cluster; 0 means spatial noise. Significance comes from a permutation
test: shuffle the values over the same grid 199 times and ask how
extreme the observed I is (pseudo p-value). Implemented in ~60 lines in
`features/spatial.py` instead of adding a dependency - every step of it
is explainable, which is the point of this project. Urban features
SHOULD cluster; if GFA came out with I ~ 0 the join would be broken.
That makes Moran's I a data *test*, not just a statistic.

## What the adversarial review caught (and why it matters)

Four reviewer agents re-derived the phase from the README and attacked
the code AND the shipped parquet files; a verifier then had to
reproduce every major claim before it counted. All four majors were
real (full detail: DECISIONS D-034). Two make excellent interview
stories because each is a *class* of mistake, not a typo:

- **The directed-graph trap.** osmnx street layers are directed: a
  two-way street is stored twice with the same geometry. Degrees and
  betweenness collapsed direction correctly, but the road-length
  features summed raw rows - local streets (mostly two-way) came out
  1.98x too long while major roads (often one-way pairs) only 1.21x,
  silently distorting the street-mix signal. Lesson: *know whether
  each representation of a network is directed, and normalize at the
  feature boundary, not in your head.*
- **The TV-tower trap.** `building=no` is OSM's explicit "this is NOT
  a building" marker. It was missing from the drop list, so the
  371.5 m Almaty TV mast divided by 3 m per level became a 124-storey
  "public building" holding 60,000 m2 of phantom floor space on
  Kok-Tobe - which the k-ring features then smeared into every
  neighbouring hex. Lesson: *tag semantics are data cleaning; a
  height-to-floors heuristic needs a sanity bound* (we now warn above
  55 derived floors).

The review also forced honesty in the docs themselves: the first
version of this very chapter told a plausible story about population
being dropped against residential GFA "by construction" - the shipped
report showed that drop never happened (the real pair was population
vs building count). A narrative that contradicts your own artefacts
is worse than no narrative; everything here is now written FROM
features_report.json.

## Interview Q&A

- *Why compute features on the buffer if the model trains on city
  hexes?* Because k-ring context of an edge hex reaches outside the
  city; truncating there would systematically bias edge features
  downward - the classic edge effect.
- *What happens to a distance feature when its target class is absent?*
  It holds the documented fallback (the study-area diagonal) - finite
  and constant beats NaN or a fabricated zero. The sample demonstrates
  it: no motorway/trunk/primary road crosses the bbox, so
  net_dist_major_m is a constant 2,812 m there (the metro columns are
  real - four stations sit inside the bbox).
- *Why keep density features at all if they duplicate counts?* They are
  kept in the build for readability, and the pruning step (with the
  report) removes them transparently. Deleting them silently in code
  would hide the reasoning; the report IS the documentation.
- *Why prune on r8 and apply to r9?* Phase 3 models at r8; r9 must keep
  the identical schema so maps and the model share column meaning.
- *Why DBSCAN and not k-means?* k-means needs k and assigns every point
  to a cluster; DBSCAN discovers the number of dense cores and discards
  noise - matching "find the sub-centres, wherever and however many
  they are".
- *Is the betweenness comparable between sample and city?* No - it is
  normalized within each graph; comparisons are valid within one run,
  which is how the model uses it.

---

# Phase 3 - The job-density model

**What was built.** `almaty-access model` turns the 68 features and the
official district totals into jobs per hex at r8 and r9, with a
P10/P90 uncertainty band and a report that shows its work
(`model_report.json`). City result: 1,091,192 jobs placed, exactly
conserving every district total.

## Dasymetric disaggregation - the one idea the phase rests on

We know the WHAT (how many jobs per district, officially) but not the
WHERE within each district. Dasymetric disaggregation splits a known
total over small units using an auxiliary *weight surface*:

    jobs_hex = total_district * weight_hex / sum(weights in district)

Everything interesting is in the weights; the rescaling guarantees the
accounting (the "pycnophylactic" or mass-preserving property). This is
exactly how WorldPop maps population, applied to jobs.

## Model A - the benchmark you must not skip

Weights = GFA x a jobs-per-100-m2 coefficient per building function.
Because of the rescaling, only the RATIOS of the coefficients matter -
the notebook shows that doubling the commercial coefficient barely
re-ranks hexes. A transparent baseline like this is the anchor for
every claim the fancier model makes: beat it or explain why not.

## Model B - a forest trained where the truth lives

The forest cannot be trained at hex level (no hex labels exist), so it
is trained at DISTRICT level - 8 districts x 5 years of annual
targets - on log(jobs per km2), and then asked to predict the same
quantity per hex. Two technical points make that legal:

- **Unit consistency.** Every mass feature is converted to a density
  per km2 at BOTH levels (a model trained on district GFA sums would
  see hex sums ~1000x smaller and every split would misfire). Shares,
  means and distances pass through unchanged.
- **Static features.** The extra years do not add independent
  feature-target pairs (OSM is a 2026 snapshot); they stabilise each
  district target. Validation therefore holds out whole districts.

## Why log of the density?

Job density spans orders of magnitude; a forest trained on raw
density would spend all its splits on the one densest district. The
log makes relative errors comparable across districts - and
exp(prediction) is guaranteed positive, which a weight must be.

## LODO - and reading a mixed verdict honestly

Leave-one-district-out: train on 7, predict the 8th, 8 times. Results
on the city: **MAPE 34.1% (forest) vs 60.5% (baseline)** - the forest
roughly halves the relative error, winning the README criterion. But
the baseline keeps the lower MAE, because the forest underpredicts
held-out Almaly (75k vs 252k): **trees cannot extrapolate** - with the
densest district held out, nothing in training resembles it, so the
forest predicts something like the second-densest. That is a property
of the method, not a bug, and the honest move is to show both metrics
and explain the mechanism. The production surface is unaffected:
official totals pin every district, so LODO measures how well the
density SHAPE generalises, where the forest sees business POIs,
centrality and context that pure GFA cannot.

*Interview line: "my validation metric disagreed with itself - MAPE
said the forest, MAE said the baseline - and the resolution was
understanding WHY (extrapolation at the extreme), not picking the
flattering number."*

## Uncertainty from the forest itself

Every tree is a full model, so every tree yields a full weight
surface; rescale each one and you get 500 complete, total-conserving
job maps. Per-hex quantiles across them give P10/P50/P90 - wide where
the trees disagree. The central P50 surface is rescaled once more so
the shipped jobs_rf conserves totals exactly; the band conserves only
approximately (stated in the code).

## Explanation: permutation importance and SHAP

Impurity-based importances are biased toward high-cardinality
features; permutation importance (shuffle one feature, measure the
damage) is the safer default and ships in the report. SHAP values
(notebook) decompose each prediction into feature contributions. With
40 rows and collinear features, read the GROUP of top features -
business POI counts, commercial/large GFA, neighbourhood context -
not the exact ranking.

## Interview Q&A

- *Why not train directly at hex level?* No hex labels exist. Model C
  in the README stays open for the day a geocoded enterprise registry
  appears; it would also validate A and B.
- *Why is jobs-per-resident a sanity check and not a feature?* It IS
  the production/attraction contrast (Almaly 2.17, dormitory Alatau
  0.18) - exactly what a transport model of home-work flows needs to
  see, and the shape our result must reproduce to be believable.
- *Why equal-split jobs in a zero-weight district?* The official total
  is a fact; refusing to place it would silently delete jobs. Equal
  split is the explicit "we know nothing here" prior, and it logs.
- *Why fix the forest hyperparameters instead of tuning?* With 8
  districts, tuning against LODO is fitting the validator. Fixed,
  boring values + an honest error report beat an optimised story.
- *Why are buffer hexes zero jobs?* No official total covers them.
  The satellite towns' employment is real but unobserved - flagged as
  a caveat for edge-hex accessibility in Phase 4.

## What the Phase 3 adversarial review caught

Three confirmed majors, each a transferable lesson (full detail: D-043):

- **The positivity trap.** exp(prediction) is never zero, so the
  forest quietly placed a tenth of all city jobs on bare mountain
  hexes - 38.6% of Medeu's total in empty terrain - because rescaling
  must put the official total SOMEWHERE. The fix is the same one
  WorldPop uses for population: a binary capability mask (here: has
  floor space or POIs) multiplied into the weights. Lesson: *a
  dasymetric weight model needs an explicit zero, because "very
  small times a big rescale factor" is not zero.*
- **The two-resolutions trap.** District totals were resolved at r8
  and applied to the r9 surface; a border sliver district with r9
  hexes but no r8 hex got silent zeros (the committed sample proved
  it: 8 bostandyk hexes). Lesson: *when the same constraint is
  applied at two spatial scales, derive it at each scale - edge sets
  differ.*
- **The untested-assumption trap.** 97% of floor counts come from
  config defaults, and no test varied them until the review forced
  the README 3.4 sensitivity analysis into existence. It ended well -
  rank stability >= 0.994 under height and POI-mapping perturbations -
  but "it ended well" was EARNED, not assumed.

And one more for the record: the review demanded a leakage probe for
LODO (poison one district x100; its held-out prediction must stay
sane), which is now a permanent regression test. Validation code
deserves tests as much as model code does.
