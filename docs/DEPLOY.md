# Deploying the dashboard and the static map

Phase 7 produces two shareable artifacts, both built by
`almaty-access export` and both **committed to the repository** so free
hosting can serve them with zero infrastructure (DECISIONS D-069):

| Artifact | Where it lives | What serves it |
|---|---|---|
| Dashboard data bundle (~7 MB, r8 parquets + report JSONs) | `data/app/` | Streamlit Community Cloud (or any host running the app) |
| Static web map (self-contained HTML, headline measure) | `docs/webmap/index.html` | GitHub Pages |

Everything in both is derived from open data; sources and licences are
in `data/README.md`. After re-running the city pipeline, refresh them
with `almaty-access export` and commit the diff.

## 1. The dashboard on Streamlit Community Cloud (free)

The app resolves its data in this order (see
`src/almaty_access/dashboard/data.py`): the `ALMATY_ACCESS_DATA`
environment variable → `data/app/` (the committed bundle) → a full
local `data/processed/` → the sample. On a fresh clone the committed
bundle is what it finds, which is exactly the point.

1. Push the repository to GitHub (public is fine — the bundle is open
   data).
2. Go to <https://share.streamlit.io>, sign in with GitHub, **Create
   app → from existing repo**.
3. Fill in:
   - **Repository**: `notsickbutfunni/almaty-jobs-access`
   - **Branch**: `main`
   - **Main file path**: `app/streamlit_app.py`
   - **Advanced → Python version**: 3.12
4. Deploy. Streamlit Cloud installs the package from `pyproject.toml`
   (the app only needs the installed `almaty_access` package plus
   `streamlit`/`pydeck`/`altair`, which are regular dependencies).

No secrets are needed: the basemap is key-less Carto, and the data is
in the repo. The free tier sleeps after inactivity and wakes on the
first visit.

### Alternative: Hugging Face Spaces

Create a Space with the **Streamlit** SDK, point it at this repository
(or push to the Space's git remote), and set `app_file:
app/streamlit_app.py` in the Space's README front matter. Same zero
secrets, same committed bundle.

### Running it locally

```bash
uv sync
uv run streamlit run app/streamlit_app.py
```

With a full `data/processed/` present the app prefers it over the
bundle; to pin a directory explicitly set `ALMATY_ACCESS_DATA`.

## 2. The static map on GitHub Pages

`docs/webmap/index.html` is one self-contained page (hex layer inlined,
map libraries from a CDN, key-less Carto basemap).

1. GitHub repository → **Settings → Pages**.
2. **Source**: "Deploy from a branch"; **Branch**: `main`, folder
   `/docs`. Save.
3. The map appears at
   `https://notsickbutfunni.github.io/almaty-jobs-access/webmap/` after
   the first Pages build (a minute or two).

GitHub Pages serves the `docs/` folder as-is; nothing else in `docs/`
is linked from the page, but it is all public documentation anyway.

## 3. Refreshing after a pipeline change

```bash
uv run almaty-access scenarios          # or whatever stage changed
uv run almaty-access export             # rewrites data/app + docs/webmap
git add data/app docs/webmap
git commit
```

`export` is cached make-style: it only rewrites when a processed input
is newer, `--force` overrides, and the bundle's `manifest.json` is
deterministic (no timestamps), so an unchanged pipeline produces a
byte-identical bundle and an empty git diff.
