"""Static one-page web map of the headline measure (README Phase 7).

For people who will not open the app: a single self-contained HTML file
(deck.gl H3 hexagons over a key-less Carto basemap) written to
``paths.webmap`` — ``docs/webmap/`` in the city config, so GitHub Pages
can serve it straight from the repository. Building it is offline; the
page loads its map libraries from a CDN when viewed.
"""

from __future__ import annotations

import json
import logging

import h3
import pandas as pd
import pydeck as pdk

from almaty_access.access.build import ACCESS_FILE_TEMPLATE
from almaty_access.config import Config
from almaty_access.dashboard import palette
from almaty_access.dashboard.charts import measure_label
from almaty_access.equity.build import HEX_FILE_TEMPLATES, RASTER
from almaty_access.equity.build import REPORT_FILE as EQUITY_REPORT_FILE
from almaty_access.io import ensure_built
from almaty_access.model.build import JOBS_FILE_TEMPLATE

logger = logging.getLogger(__name__)

WEBMAP_FILE = "index.html"
REPO_URL = "https://github.com/notsickbutfunni/almaty-jobs-access"

_TOOLTIP_HTML = (
    "<b>{district_label}</b><br/>"
    "residents: {population_label}<br/>"
    "estimated jobs: {jobs_label}<br/>"
    "{value_label}: {display}"
)


def build_webmap(config: Config, *, force: bool = False) -> None:
    """Write the static map (make-style cached on its processed inputs)."""
    res = config.h3.analysis_resolution
    processed = config.paths.processed
    inputs = [
        processed / ACCESS_FILE_TEMPLATE.format(res=res),
        processed / JOBS_FILE_TEMPLATE.format(res=res),
        processed / HEX_FILE_TEMPLATES[RASTER].format(res=res),
        processed / EQUITY_REPORT_FILE,
        config.source_path,
    ]
    output = config.paths.webmap / WEBMAP_FILE
    ensure_built(output, inputs, lambda: _write_webmap(config), force=force)


def _overlay_html(title: str, subtitle: str, bounds: tuple[float, float], unit: str) -> str:
    gradient = ", ".join(palette.SEQ_BLUE)
    low, high = bounds
    return f"""
<div style="position:absolute; top:12px; left:12px; z-index:1; max-width:340px;
            background:{palette.SURFACE}; color:{palette.INK};
            border:1px solid {palette.GRID_LINE};
            border-radius:8px; padding:12px 14px;
            font-family:system-ui,-apple-system,Segoe UI,sans-serif; font-size:13px;">
  <div style="font-weight:600; font-size:15px; margin-bottom:4px;">{title}</div>
  <div style="color:{palette.INK_2}; margin-bottom:8px;">{subtitle}</div>
  <div style="height:10px; border-radius:5px;
              background:linear-gradient(to right, {gradient});"></div>
  <div style="display:flex; justify-content:space-between; color:{palette.MUTED}; font-size:11px;">
    <span>{low:,.1f}{unit}</span><span>{high:,.1f}{unit}</span>
  </div>
  <div style="color:{palette.MUTED}; font-size:11px; margin-top:8px;">
    Open-data estimate; transit is approximate (a lower bound).
    Data: OSM, Kontur, Overture, Copernicus GLO-30, stat.gov.kz.
    <a href="{REPO_URL}" style="color:{palette.BLUE};">Method &amp; code</a>
  </div>
</div>
"""


def _write_webmap(config: Config) -> None:
    res = config.h3.analysis_resolution
    processed = config.paths.processed
    access = pd.read_parquet(processed / ACCESS_FILE_TEMPLATE.format(res=res))
    jobs = pd.read_parquet(processed / JOBS_FILE_TEMPLATE.format(res=res))
    equity = pd.read_parquet(processed / HEX_FILE_TEMPLATES[RASTER].format(res=res))
    with open(processed / EQUITY_REPORT_FILE, encoding="utf-8") as handle:
        report = json.load(handle)
    measure = str(report["headline"]["measure"])

    frame = access.loc[access["in_city"], ["h3", "district", measure]].copy()
    frame = frame.rename(columns={measure: "value"})
    frame = frame[pd.notna(frame["value"])].reset_index(drop=True)
    frame["population"] = frame["h3"].map(equity.set_index("h3")["population"])
    frame["jobs_rf"] = frame["h3"].map(jobs.set_index("h3")["jobs_rf"])

    fill, bounds = palette.sequential_rgba(frame["value"])
    unit = "%" if measure.startswith("accpct") else ""
    records = [
        {
            "h3": row.h3,
            "fill": fill[i].tolist(),
            "display": f"{row.value:,.1f}{unit}",
            "district_label": (
                "Outside districts" if pd.isna(row.district) else str(row.district).capitalize()
            ),
            "population_label": "-" if pd.isna(row.population) else f"{row.population:,.0f}",
            "jobs_label": "-" if pd.isna(row.jobs_rf) else f"{row.jobs_rf:,.0f}",
        }
        for i, row in enumerate(frame.itertuples(index=False))
    ]
    centers = [h3.cell_to_latlng(cell) for cell in frame["h3"]]
    latitude = sum(point[0] for point in centers) / len(centers)
    longitude = sum(point[1] for point in centers) / len(centers)

    title = measure_label(measure)
    layer = pdk.Layer(
        "H3HexagonLayer",
        data=records,
        get_hexagon="h3",
        get_fill_color="fill",
        stroked=False,
        extruded=False,
        pickable=True,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=latitude, longitude=longitude, zoom=9.7),
        map_provider="carto",
        map_style="light",
        tooltip={"html": _TOOLTIP_HTML.replace("{value_label}", title)},
    )
    output = config.paths.webmap / WEBMAP_FILE
    output.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(str(output), open_browser=False, notebook_display=False)

    page = output.read_text(encoding="utf-8")
    overlay = _overlay_html("Almaty jobs & accessibility", title, bounds, unit)
    page = page.replace("</body>", overlay + "</body>")
    page = page.replace("<title>pydeck</title>", "<title>Almaty jobs & accessibility</title>")
    # Committed text: LF endings, no trailing spaces, one final newline —
    # whatever platform built it, so re-exports never fight the fixers.
    page = "\n".join(line.rstrip() for line in page.splitlines()) + "\n"
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(page)
    logger.info("webmap: %d hexes, %s -> %s", len(records), measure, output)
