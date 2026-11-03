"""Download the Copernicus GLO-30 DEM for the study area.

GLO-30 is published as 1°x1° Cloud-Optimized GeoTIFFs on AWS Open Data
(no authentication). We open the tiles remotely and read only the window
covering the (buffered) study area, then mosaic and store the clip as
``data/raw/dem.tif`` in the storage CRS.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import rasterio
from rasterio.merge import merge

from almaty_access.config import Config
from almaty_access.download.area import study_bbox
from almaty_access.io import ensure_built

logger = logging.getLogger(__name__)

FILE_NAME = "dem.tif"


def tile_names(bbox: tuple[float, float, float, float]) -> list[str]:
    """GLO-30 tile names (1°x1°, named by their SW corner) covering a bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    names = []
    for lat in range(math.floor(min_lat), math.ceil(max_lat)):
        for lon in range(math.floor(min_lon), math.ceil(max_lon)):
            lat_part = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
            lon_part = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
            names.append(f"Copernicus_DSM_COG_10_{lat_part}_00_{lon_part}_00_DEM")
    return names


def download_dem(config: Config, *, force: bool = False) -> Path:
    """Windowed mosaic of the GLO-30 tiles over the buffered study area."""
    output = config.paths.raw / FILE_NAME

    def build() -> None:
        bbox = study_bbox(config, pad_degrees=0.01)
        names = tile_names(bbox)
        urls = [config.sources.dem_tile_url_template.format(name=name) for name in names]
        logger.info("dem: reading %d GLO-30 tile(s): %s", len(urls), ", ".join(names))
        sources = [rasterio.open(url) for url in urls]
        try:
            mosaic, transform = merge(sources, bounds=bbox)
            profile = sources[0].profile.copy()
            profile.update(
                driver="GTiff",
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=transform,
                compress="deflate",
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output, "w", **profile) as dst:
                dst.write(mosaic)
        finally:
            for src in sources:
                src.close()
        logger.info("dem: wrote %s (%dx%d px)", output, mosaic.shape[2], mosaic.shape[1])

    ensure_built(output, [], build, force=force)
    return output
