"""Download the Kontur population dataset (H3 r8) and clip it.

Kontur publishes one GeoPackage per country on HDX. We download it once
(cached in data/raw/), then clip to the study area and store the compact
clip as GeoParquet. The hexes come in Web Mercator; we store the clip in
the storage CRS with columns (h3, population, geometry).
"""

from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
import requests

from almaty_access.config import Config
from almaty_access.download.area import study_inputs, study_polygon
from almaty_access.io import ensure_built, write_geoparquet

logger = logging.getLogger(__name__)

COUNTRY_FILE = "kontur_population_country.gpkg"
FILE_NAME = "population_r8_raw.parquet"


USER_AGENT = "almaty-access-pipeline/0.1 (open-data research)"


def _download_file(url: str, destination: Path, *, timeout_s: int = 600) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s", url)
    with requests.get(
        url, stream=True, timeout=timeout_s, headers={"User-Agent": USER_AGENT}
    ) as response:
        response.raise_for_status()
        partial = destination.with_suffix(destination.suffix + ".part")
        with open(partial, "wb") as handle:
            shutil.copyfileobj(response.raw, handle, length=1024 * 1024)
        partial.replace(destination)
    logger.info("saved %s (%.1f MB)", destination, destination.stat().st_size / 1e6)


def download_country_gpkg(config: Config, *, force: bool = False) -> Path:
    """Fetch (and if needed gunzip) the Kontur country GeoPackage."""
    output = config.paths.raw / COUNTRY_FILE

    def build() -> None:
        url = config.sources.kontur_gpkg_url
        if url.endswith(".gz"):
            compressed = output.with_suffix(output.suffix + ".gz")
            _download_file(url, compressed)
            logger.info("decompressing %s", compressed.name)
            with gzip.open(compressed, "rb") as src, open(output, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            compressed.unlink()
        else:
            _download_file(url, output)

    ensure_built(output, [], build, force=force)
    return output


def download_population(config: Config, *, force: bool = False) -> Path:
    """Clip the country hex population to the study area."""
    output = config.paths.raw / FILE_NAME

    def build() -> None:
        from pyogrio import read_info

        country_path = download_country_gpkg(config, force=False)
        area = study_polygon(config)
        # The bbox filter is interpreted in the LAYER's CRS (Kontur ships
        # EPSG:3857), so project the study area's bounds into it first.
        layer_crs = read_info(country_path)["crs"]
        area_in_layer_crs = (
            gpd.GeoSeries([area], crs=config.crs.storage).to_crs(layer_crs).total_bounds
        )
        logger.info("population: reading %s (bbox filter in %s)", country_path.name, layer_crs)
        frame = gpd.read_file(country_path, bbox=tuple(area_in_layer_crs))
        frame = frame.to_crs(config.crs.storage)
        frame = frame[frame.geometry.intersects(area)]
        if frame.empty:
            raise RuntimeError(
                f"no population hexes intersect the study area; check 'area' in the "
                f"config and the CRS of {country_path.name}"
            )

        h3_column = next((c for c in frame.columns if c.lower() in ("h3", "h3_index")), None)
        if h3_column is None:
            raise ValueError(
                f"{country_path.name}: no H3 index column found (columns: {list(frame.columns)})"
            )
        population_column = next((c for c in frame.columns if "population" in c.lower()), None)
        if population_column is None:
            raise ValueError(f"{country_path.name}: no population column found")

        result = gpd.GeoDataFrame(
            {
                "h3": frame[h3_column].astype(str),
                "population": pd.to_numeric(frame[population_column], errors="coerce").fillna(0.0),
            },
            geometry=frame.geometry,
            crs=frame.crs,
        )
        resolutions = {h3.get_resolution(cell) for cell in result["h3"].head(100)}
        if resolutions != {config.h3.analysis_resolution}:
            raise ValueError(
                f"population hexes are at H3 resolution(s) {sorted(resolutions)}, "
                f"expected {config.h3.analysis_resolution}"
            )
        logger.info("population: %d hexes, %.0f residents", len(result), result["population"].sum())
        write_geoparquet(result, output)

    ensure_built(output, study_inputs(config), build, force=force)
    return output
