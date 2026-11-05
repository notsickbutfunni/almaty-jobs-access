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

# Phase 4 - Travel times and accessibility

## The shape of the phase

Two steps, cached separately. `times` turns each mode's network into a
graph whose edge weights are SECONDS, then computes hex-to-hex travel
times. `measures` turns those times plus the Phase 3 job surface and the
POI layers into per-hex accessibility numbers. Keeping them separate
means you can re-tune a measure (a half-life, a catchment) without
recomputing twenty minutes of matrices.

## Tobler's hiking function - terrain enters the travel times

Almaty rises ~900 m from the steppe to the foothills, so flat-speed
walking times would flatter the south. Tobler's classic function gives
walking speed as a function of slope s (rise over run):

    W(s) = 6 * exp(-3.5 * |s + 0.05|)   [km/h]

Read it: fastest on a gentle 5% DOWNHILL (the |s + 0.05| is zero at
s = -0.05), decaying exponentially as the ground tilts away from that
optimum in either direction. On flat ground it gives 5.04 km/h; we
rescale the whole curve so flat equals the configured 4.8 km/h - the
SHAPE is Tobler's, the level is ours. Because Phase 1 stored a grade
per DIRECTED edge, climbing Dostyk avenue and descending it get
different speeds - which is exactly why the walk graph keeps both
directions of every street.

Cycling has no equally canonical formula, so the code states a
transparent assumption instead: lose 1 km/h per 1% of climb, gain
0.5 km/h per 1% of descent, never above 1.4x flat speed, never below a
pushing pace. When an assumption cannot be sourced, make it small,
visible and easy to change - do not dress it up as literature.

## Why scipy instead of networkx loops

A shortest-path call in networkx walks Python objects; scipy's
`csgraph.dijkstra` runs compiled code over a CSR sparse matrix. The
walk graph has 110,732 nodes and 300,120 directed edges; the 12,710
snapped r9 origins (175 of the grid's 12,885 hexes are off-network)
finish in about twelve seconds in batches of 512, producing
1.89 million origin-destination pairs within 60 minutes. The same job
as a networkx loop is hours. The recipe: map node ids to positions,
build a COO matrix of edge seconds, convert to CSR, call dijkstra with
`limit` (the search stops expanding past the largest threshold - a big
win), slice the dense result at destination columns only.

Two traps the tests now guard:
- **COO sums duplicates.** Two parallel edges between the same nodes
  (a street and its service alley) would be ADDED into one slower
  edge. Fastest-edge-wins must happen before matrix assembly.
- **A zero-weight edge is no edge.** Sparse graphs cannot tell an
  explicit 0 from an absent entry, so free arcs (alighting a bus) are
  floored at one second.

## Transit without a timetable - the headway model

With no GTFS, the honest model is frequency-based. Suppose a bus comes
every H minutes and you arrive at a random moment: your expected wait
is H/2. So the graph gets, per route and stop, a virtual "on the bus"
node; stepping onto it costs H/2, riding to the next stop costs
distance x circuity / assumed commercial speed, stepping off is free.
The walking network is PART of the same graph, so Dijkstra invents
access walks, egress walks and transfers on its own - and a transfer
automatically pays a fresh H/2, which is precisely what transferring
costs when you cannot time your arrival.

Everything assumed sits in config: speeds (bus 17, trolleybus 15,
metro 32 km/h commercial), headways (10/12/7 min), circuity per mode
(1.3 street, 1.05 tunnel). The metro gave the model a free validation:
the OSM relations carry exactly 11 stops per direction and ~13 km of
legs - the real line's 11 stations and 13.4 km.

The hard limitation to say out loud: two thirds of the route relations
in the Kazakhstan OSM extract list their ways but not their stops (the
Almaty-only share is unknowable - ways-only relations are never
geolocated), and a route without ordered stops cannot produce
stop-to-stop legs without inventing stop locations. The kept network
(55 routes, 291 stops) is therefore a LOWER BOUND on transit access;
hexes on unmapped-stop corridors fall back to walking. And the
`interval` tag is rarer still: only 2 of the 55 kept routes carry one,
so 96% of the waiting times rest on the config assumptions. Phase 5
must read transit "deserts" with both caveats in hand.

## Snapping, and the disconnected-island trap

Hex centroids are not on the network; each snaps to the nearest node
and the snap distance is walked at flat speed on top of every journey.
Two guards make this honest. First, snapping only targets the largest
STRONGLY connected component - the nearest node might be a gated
service loop from which the city is unreachable, and an origin snapped
there would report a fake desert. Second, a centroid more than 2 km
from any node (high pastures, the ski ridge) is excluded and counted
instead of being given a one-hour phantom walk.

## The measures

- **Cumulative opportunities** A_i(T) = sum of jobs in every hex within
  T minutes. Instantly explainable ("jobs within 30 minutes"), and the
  normalized version (% of ALL city jobs) is what lets walk, transit
  and car share one scale.
- **Gravity (Hansen)** A_i = sum_j O_j * exp(-beta * t_ij). Instead of
  pulling beta from thin air, derive it from a HALF-LIFE: beta =
  ln(2) / 20 min means a job 20 minutes away counts half of one next
  door. The report recomputes the surface at 10- and 30-minute
  half-lives; Spearman rho >= 0.997 against the central run, so the
  RANKING of hexes - all Phase 5 uses - does not hang on the choice.
- **Nearest facility** in walking minutes, for school / kindergarten /
  clinic / grocery / park / metro. Computed without a per-facility
  loop: one Dijkstra from a virtual super-source on the TRANSPOSED
  graph (reverse every street, start from all facilities at once) gives
  every node's distance TO its nearest facility exactly.
- **2SFCA** for services where capacity matters. Step 1: every facility
  hex j gets R_j = (its facilities) / (population able to reach j
  within 15 min) - a clinic besieged by a dense bedroom district is
  worth less per person. Step 2: every hex sums the R_j it can reach.
  Units: facilities per 1,000 residents, competition-adjusted. With no
  open capacity data, one facility = one unit of supply - a stated
  proxy, not a hidden one.

## Interview Q&A

**Q: Why not just buffer each hex 2 km and count jobs inside?**
Because people move on networks, not circles. In the foothills a 2 km
crow-fly buffer crosses ravines no street crosses; network time with
slope-aware speeds is the honest geometry. That is also why the car
map is so much flatter than the walk map: streets erase terrain, legs
do not.

**Q: Your transit numbers - would you defend them to a transit planner?**
As APPROXIMATE access under stated assumptions, yes - the config holds
the assumed speeds and headways, the report labels every transit field,
and the OSM-sourced network is a documented lower bound. As timetable
truth, no - and the honest next step is operator data (real headways,
real stop lists), which drops into the same config without code
changes.

**Q: Why is expected waiting half the headway?**
Arrivals are uncorrelated with the schedule when there is no schedule
to read. If buses come every H minutes and your arrival is uniform in
the interval, the mean residual wait is H/2. Every boarding pays it,
which makes transfers cost what they feel like they cost.

**Q: Car access looks near-uniform at 45 minutes. Is that a bug?**
It is free-flow by design (reference mode): median hex reaches ~96% of
jobs in 45 min. Add congestion and the picture changes - that is
exactly why the equity phase compares transit against car as a RATIO
per hex rather than trusting absolute car minutes.

## What went wrong before it went right (Phase 4 edition)

- **"Prefer stop positions" discarded data.** PTv2 lets mappers tag
  halts as stops AND platforms; one route had 13 stop positions but 56
  platforms. The first extractor preferred stops and silently dropped
  three quarters of that route. Fix: use whichever list is richer,
  never both. *Lesson: when two encodings coexist, measure which one
  carries the data before preferring either.*
- **One circuity for streets and tunnels.** Metro legs x1.3 inflated
  end-to-end rail times ~30%. Circuity is per mode now. *Lesson: any
  constant applied across modes deserves a per-mode sanity check
  against a known journey.*
- **The verdict cell caught nothing - by design.** Monotonicity over
  thresholds, transit never slower than the walk it contains,
  percentages bounded, gravity below the 60-min cumulative: none of
  these can fail silently anymore because the notebook refuses to
  finish if they do.

## What the Phase 4 adversarial review caught

Three confirmed majors, none of them travel-time math - all three were
about honesty and untested promises, which is its own lesson:

- **The scope-creep number.** The report said "8 routes have real
  interval tags" next to kept-network counts, but the 8 counted the
  whole Kazakhstan PBF; the kept Almaty network has 2. The travel
  times were right all along - the LABEL overstated how much real data
  anchored them, in the one layer whose whole story is "assumptions
  are labelled". *Lesson: every reported count must name its
  denominator. A true number in the wrong scope is a false claim.*
- **The untested defining property.** "Every boarding pays a fresh
  expected wait" is what makes the headway model honest about
  transfers - and no test ever built two routes. A refactor merging
  virtual nodes per stop would have made transfers free with the suite
  green. *Lesson: the sentence you would say first in an interview is
  the sentence that most needs a test.*
- **The documented test that did not exist.** DECISIONS claimed the
  route-leg dedup was "regression-tested"; only the street-edge twin
  was. The fix was writing the missing tests, not softening the claim -
  the same trap (sparse COO summing duplicates) is now pinned at all
  four places it bites. *Lesson: audit docs against the test suite the
  way you audit code against the spec.*

The round also produced the intrazonal rule (a hex reaches its own
opportunities for one snap leg, not an out-and-back - before it, 84
city hexes "could not reach themselves" in 15 minutes), stratified
spot-check pairs (a uniform draw across a 30 km city left 9 of 10
pairs unwalkable, useless for checking the walk mode), and the r8
granularity caveat where `near_school 12 min` and `0 schools in 15
min` coexist honestly because one measures to the point and the other
to its hex centroid.

# Phase 5 - Equity and spatial statistics

## From "how much" to "who" and "where"

Phase 4 gave every hex an accessibility number. Equity analysis changes
the unit: the question is about PEOPLE. So every statistic here weights
a hex by its residents - a hex of 5,000 people counts 5,000 times - and
only inhabited city hexes take part. An empty mountain cell with zero
access is geography, not injustice.

## Lorenz curve, Gini, Palma - three views of one sorted line

Line up all residents from the worst served to the best served. Walk
along the line and keep two running totals: the share of PEOPLE passed
so far, and the share of ALL ACCESSIBILITY they hold. Plot one against
the other: that is the Lorenz curve.

- If everyone had the same access, the first 30% of people would hold
  30% of the access: the curve is the diagonal.
- The more it sags below the diagonal, the more unequal.

**Gini** = twice the area between the curve and the diagonal. 0 means
perfect equality; values near 1 mean a few people hold nearly
everything. In formula form, with vertices (P_k, L_k):

    G = 1 - sum_k (P_k - P_{k-1}) * (L_k + L_{k-1})

Because everyone inside a hex shares one value, the curve is a chain of
straight segments - one per hex - and this sum is EXACT, not a
numerical approximation.

**Palma** ignores the middle and compares the tails: the access held by
the best-served 10% of residents divided by the access held by the
worst-served 40%. Under perfect equality it is 10/40 = 0.25. It answers
a question Gini blurs: how far apart are the extremes?

What Almaty looks like (r8, % of city jobs reachable; Gini on the
population raster, then on calibrated residents - see "The weights"
below):

| mode | Gini | Palma |
|---|---|---|
| car, 45 min (free-flow) | 0.005 / 0.004 | 0.25 |
| bike, 45 min | 0.25 / 0.23 | 0.76 |
| transit, 45 min (approximate) | 0.62 / 0.59 | 8.0 |
| walk, 30 min | 0.63 / 0.60 | 8.7 |

Read the table as a sentence: with a car (and no congestion) where you
live barely matters; on foot or by transit it decides almost
everything.

## The desert rule - and why the literal version found (almost) nothing

README: a desert is a hex with population above the 75th percentile and
accessibility below the 25th. Taken literally - both percentiles across
hexes - it flagged 0, 0, 0 and 1 of 712 hexes for the four r8 headline
measures. Not because the city is fair: because the two extremes rarely
meet. The "low access" threshold across hexes is set by hundreds of
near-empty outskirt cells, far below anything a dense hex experiences.

The fix is to ask the question about people on the access side: a
desert is a dense hex (population above the hex 75th percentile) whose
access is at or below the level under which the WORST-SERVED QUARTER OF
RESIDENTS live (the population-weighted 25th percentile). For transit
that finds 24 hexes with 146 thousand residents on the raster - 104
thousand of them in Alatau's microdistricts. The literal rule's count
is still printed next to every result, so nobody has to take the
reinterpretation on trust.

Two guards came out of the review and are worth knowing:

- **A relative rule always finds a "worst quarter".** Free-flow car
  access sits between 96% and 98%; the rule dutifully flagged 161 thousand
  residents as car "deserts". Now a measure whose worst-served quarter
  still reaches, ON AVERAGE, 95% of what the median resident reaches
  reports no deserts at all (car: 98.5%). The average matters: the
  first version compared the quarter's upper EDGE with the median, and
  when a quarter of residents share the median value that edge IS the
  median - however bad the tail below it.
- **Both halves of the rule use residents**, so the list changes with
  the population weights: 24 hexes on the raster, 33 calibrated, 21 in
  common. The notebook maps that common core separately.

## Moran's I and LISA - "clustered" is not the finding

**Global Moran's I** measures whether similar values sit next to each
other (+1 clustered, 0 random, -1 checkerboard). For accessibility it
comes out around 0.9 - and that is NOT a discovery. Two neighbouring
hexes can reach almost the same set of destinations, so their values
are similar by construction. Reporting "I = 0.9, p < 0.001" as a result
would be reporting arithmetic.

The informative tool is local. **LISA** (local Moran, Anselin 1995)
gives every hex its own statistic:

    I_i = z_i * (mean of z over the neighbours of i)

where z is the standardized value. Both high -> HH, a hot spot. Both
low -> LL, a cold spot. Opposite signs -> an outlier (HL, LH).
Significance comes from **conditional permutation**: keep hex i fixed,
deal its neighbours random values from the rest of the city a few
thousand times, and ask how often the shuffled I_i is as extreme as the
real one.

Result: the transit hot spot is the central core (24% of residents on
the raster, 34% calibrated); the cold spots are outer Alatau,
Nauryzbay, the Medeu foothills and north-east Turksib - large on the
map but only about 2% of residents. That contrast is itself a finding:
the worst clusters are thin in people, and MOST underserved residents
(the 51-62% below 5% of jobs) do not live in a statistical cold spot at
all - they live in ordinary, moderately bad places. Cold-spot maps
alone would miss them.

### The tie trap (worth telling in an interview)

Many accessibility measures are zero-inflated: 67% of inhabited r9
hexes cannot reach any kindergarten within the 15-minute catchment. A
zero hex with zero neighbours then TIES with every shuffle that also
deals it all zeros - and both p-values the library offers mishandle
ties, in opposite directions:

- its recommended two-sided p-value mirrors the observed percentile to
  the other tail and counts everything beyond it; the tie mass lands
  inside the count. For kindergartens no p-value came out below 0.084
  with 999 shuffles (0.090 with 9,999) - not one significant cell for a
  variable with global I = 0.77;
- its one-sided p-values compare floating-point numbers with >=; the
  observed value and its tied shuffles differ in the 16th digit, the
  ties are NOT counted, and a zero hex with two zero neighbours gets
  p = 0.001 (999 shuffles) although a third of the shuffles tie with
  it - the tie-inclusive tail is 0.33. For schools that made 1,446
  zero-region "significant cold spots";
- a third option avoids both traps but folds a skewed distribution
  about its mean, so it can only ever see one tail: 397 walking hot
  spots and not a single cold spot.

The project computes the p-value itself from the library's own
shuffles: count both tails with a float tolerance, double the smaller.
On tie-free data it matches the library within a few hexes; on
zero-inflated data it finds real provision hot spots and no fake cold
spots. *Lesson: a p-value is a counting rule. Know what yours counts
when values tie - and if your data has many zeros, they will.*

### One test per hex is thousands of tests

At alpha = 0.05, 3,000 tests produce ~150 "significant" hexes from pure
noise. **Benjamini-Hochberg** control keeps the expected share of false
discoveries below alpha: sort the p-values, find the largest k with
p_(k) <= k/m * alpha, keep those k. The report gives both counts (for
transit: 157 cold-spot and 78 hot-spot hexes at p < 0.05, of which 134
survive FDR).

A practical catch: permutations set a floor under the p-value. With 999
of them the smallest two-sided p is 0.002, and BH over ~3,300 tests
needs 133 simultaneous floor-level results before it admits one - so
the city run uses 9,999 (floor 0.0002, 14 needed). Where even that is
not enough the report says "unresolved" (null), not zero: "no cluster
survives" and "this run cannot tell" are different statements.

### A p-value from shuffles is itself a measurement

The pseudo p-value is a share counted over a finite number of shuffles,
so it has sampling error like any other share. The two-sided value
doubles a tail share t, counted over N shuffles:

    SE(p) = 2 * sqrt(t * (1 - t) / (N + 1)),   t = alpha / 2

At alpha = 0.05 that is about 0.010 with 999 shuffles and 0.003 with
9,999. A hex whose p-value sits within two standard errors of 0.05
(0.044 to 0.056 for the city run) is a coin toss: another random seed
can move it across the line. The report counts those hexes per measure
(`borderline`), and a one-off check with four seeds shows it matters:

| measure | hot spots in the report | across four seeds |
|---|---|---|
| transit 45 min (r8) | 78 | 78 every time |
| walk 30 min (r9) | 341 | 335-345 |
| schools 2SFCA | 168 | 162-183 |
| kindergartens 2SFCA | 240 | 230-240 |
| clinics 2SFCA | 219 | 219-279 |

Smooth measures are stable. Zero-heavy ones are not, and the reason is
worth understanding. Clinic 2SFCA is 62% zeros plus 13 extreme hexes
(a clinic next to almost no residents scores up to 35 per 1,000). A
shuffled neighbourhood beats an ordinary hot one almost only when it
happens to draw one of those 13 - about 2.4% of shuffles - so EVERY
hex with a neighbour mean between 1.4 and 1.9 gets the same p-value:
56 hexes at exactly p = 0.0486. Blocks like that cross the 0.05 line
together. *Lesson: a seed makes a result reproducible, not stable.
Before quoting "219 hot spots", ask how many of them a different seed
would keep - and report that number next to the count.*

## How much of "transit" is transit?

The transit network is a lower bound: only routes whose OSM relation
lists ordered stops could be modelled. Phase 5 turns that label into a
number. For 100% / 83% / 50% / 27% of residents (at 15 / 30 / 45 / 60
minutes; 100 / 80 / 43 / 23% with calibrated residents) the modelled
transit adds NOTHING over walking; 29% of residents (35% calibrated)
live in a hex with a modelled route, while 52% (50%) live in a hex
that has mapped stops but no modelled route. Run the headline on
walking alone and you get 65% instead of 62% (calibrated: 54.5%
instead of 51.5%).

So a "transit desert" here is usually a walking value in a place where
the mapping is thin (1 of the 24 transit desert hexes has a modelled
route; 19 contain mapped stops), and the transit ranking of districts
partly ranks OSM completeness. *Lesson: a caveat a reader can skip is
not a caveat. Put a number on the gap and print it next to the result.*

The same thinking applies to the **mode gap** (transit / car job
access): it divides a lower bound by an upper bound (free-flow car), so
its level ("87% of residents car-dependent", 82% calibrated) is biased
everywhere - its
map, where the ratio is highest, is the usable part. Always ask which
direction each ingredient of a ratio is biased before you quote it.

## The weights were the biggest surprise

Every number above is weighted by residents, and the residents come
from a population raster - a model. While reading the district table,
one line looked wrong: Auezov with fewer residents than Almaly. The
pipeline already held an official residence-based series (employed
residents per district, from the labour-force survey), so the check
took five minutes: the raster gives Auezov 0.42x and Almaly 0.60x their
official share, and the low-rise periphery up to 1.51x. Dense high-rise
districts are undercounted - and they are the best-served ones, so the
skew inflates every inequality number.

The response:

1. **Do not hide it**: it is a caveat in the report, with the numbers.
2. **Do not patch one number**: run the WHOLE analysis under two
   weightings - the raster, and residents rescaled so district totals
   follow the official shares. Same tables, same shapes, a `weights`
   column. The first attempt re-weighted only the headline; the review
   showed the desert list, the cluster shares and the gradient moved
   just as much while the docs claimed they did not.
3. **Find the weight hiding inside a measure.** 2SFCA divides
   facilities by the residents competing for them. Re-weighting its
   OUTPUT with calibrated residents gave means 21-29% higher - which is
   impossible, because 2SFCA conserves supply: its resident-weighted
   mean is simply facilities per resident. The measure has to be
   recomputed with calibrated demand. Then the mean stays put, and the
   real effect appears: Auezov falls from 1st to 5th of the districts
   for school provision. Its "provision hot spot" was the undercount of
   its own residents.
4. **Report the range**, not the convenient end of it: 62% on the
   raster, 51% calibrated.

What it did NOT do is silently replace the population layer - that
would change features, the job model and 2SFCA in three earlier
phases. Cross-phase changes are decisions, not fixes.

*Lessons: weights are data too - validate them against anything
official you can find. And when you test sensitivity to an input, trace
every place the input enters, including inside your measures.*

## Interview Q&A

**Q: Your Gini for car access is 0.005. Does that mean Almaty is fair?**
It means free-flow driving time barely varies inside a 30 km city, so a
model without congestion or car ownership sees everyone reaching ~97%
of jobs. It is a reference mode. The equity story is in the gap between
that line and the walk/transit lines.

**Q: Why not just map the bottom 25% of hexes as "deserts"?**
Because most of those hexes are empty. Hex-level percentiles answer a
question about land; equity needs percentiles of people. The literal
hex rule flagged at most one hex out of 712 - the person-weighted one
found 146 thousand residents.

**Q: You report "between 51% and 62%". Isn't a range a weakness?**
A single number would be false precision: the population raster is
demonstrably skewed between districts, and the two ends are the raw
raster and the one calibrated to district statistics. Knowing which
input moves the answer, and by how much, is the result.

**Q: How do you know your LISA clusters are not artifacts?**
Four guards: seeded conditional permutations (reproducible), tie-aware
p-values (the library's own break on zero-inflated data, in both
directions), FDR counts for multiple testing, and a count of the hexes
within Monte Carlo error of the 0.05 line - because a seed makes the
clusters reproducible, not stable (clinic hot spots range from 219 to
279 across four seeds; transit ones are 78 every time). Plus tests that
plant a hot and a cold blob on real H3 cells and require the method to
find them - and that check every significant label against the hex's
own value and its neighbours', so a label can never sit on the wrong
hex.

**Q: Is your transit result about transit?**
Partly. For 43-50% of residents (depending on the population weights)
the modelled network adds nothing over walking within 45 minutes, and
the report says so with numbers. The
honest reading is: "even with the transit we could model, X; on walking
alone, Y" - and the gap between a lower-bound network and reality is
the first thing better data would close.

## What went wrong before it went right (Phase 5 edition)

- **A rule that returned nothing.** The README's desert definition
  flagged at most one hex in 712. An empty result is a result to investigate,
  not to ship.
- **A statistic that could not be significant - and its twin that
  always was.** One library option left kindergartens without a single
  cluster; another turned 1,446 zero cells into school "cold spots".
  Neither was about the city.
- **Service measures at the wrong resolution.** 2SFCA at r8 had an
  infinite Palma ratio. Headline measures are now listed per
  resolution, each where it means something.
- **A district table that failed the smell test** - and led to the
  population-raster finding, the most consequential caveat of the
  phase.

## What the Phase 5 adversarial review caught

Three confirmed majors and a long tail, all in one family: the
sensitivity analysis I was proud of was only half built.

- **A robustness claim nobody had tested.** DECISIONS said the desert
  list "does not depend on these weights". It does - the rule uses
  residents twice - and a third of the calibrated list was missing from
  the shipped one. *Lesson: "X does not depend on Y" is a claim. Write
  the test that would fail if it were false, or do not write the
  sentence.*
- **A measure that was neither raw nor calibrated.** The half-calibrated
  2SFCA (see above). *Lesson: a number that cannot happen (a mean that
  moves when nothing was added) is the cheapest bug detector there is -
  know your measure's conservation law.*
- **Zero is not "unresolved".** FDR reported 0 survivors where the
  permutation count could not have produced any. The report now
  distinguishes the two.
- **The sample lied by label.** The committed sample ships synthetic
  district statistics, and the report called the weights derived from
  them "official Taldau figures". Now the source travels with the
  number.
- **A point mass in the data.** Twelve fine hexes hold nearly all
  residents of their coarse parent - 8,193 people on 133 m2 of tagged
  residential floor area - because the population split follows
  residential TAGS and private-house areas rarely carry them. They top
  every ranking by residents; they are flagged now, and the upstream
  fix is on the owner's list.
- **And a process lesson**: `pre-commit run --all-files` only sees
  files git already tracks. New files are invisible to it until staged.
  That - not a display glitch - is why "all hooks passed" and the
  commit still failed in Phase 4. Stage first, then run the hooks.

## The second review round - tests that could not fail

After the rewrite, a second, narrower review found no major problem in
the code - and one in the notebook, which still said "cluster locations
do not depend on the weighting". That is true for job measures and
false for 2SFCA, whose values are recomputed: the very distinction the
first round had been about. The rest were small, but they share a
theme worth remembering.

- **A test that cannot fail is documentation, not a test.** Three of
  mine could not: the float-tolerance example gave the right answer
  without the tolerance; the "name only stable districts" test had
  three districts, so its top-three intersection could never drop one;
  and the calibrated 2SFCA was checked only for supply conservation -
  which a lazy shortcut (divide the raster values by the district
  factor) also satisfies. The fix has a name: **mutation testing**.
  Break the code on purpose, one behaviour at a time, and run the test
  that is supposed to guard it. Twelve deliberate breaks, twelve
  failures - and the lazy shortcut was shown to pass the OLD test.
  *Lesson: after writing a test, ask "what wrong implementation would
  still pass this?" If you can name one, the test is not finished.*
- **Do not let code certify what it cannot know.** The report called
  the district statistics "official" whenever a source string looked
  like the statistics portal. Whether a hand-collected file is official
  is its owner's claim. The report now quotes the source and flags only
  one status the project itself can vouch for: its own synthetic sample.
- **A caveat is a claim about THIS dataset.** "2SFCA is zero-inflated"
  was printed on the sample, where no inhabited hex has a 2SFCA of
  zero. Caveats are now generated from measured values (and under both
  weightings), so a sentence appears only where it is true.
- **Check the checker.** Every figure the review handed over was
  re-measured before it went into the decision log. They all held -
  but one of my own older figures did not (1,440 "cold spots" is
  really 1,446), and re-measuring is also how the seed sensitivity of
  the hot-spot counts came to light.
