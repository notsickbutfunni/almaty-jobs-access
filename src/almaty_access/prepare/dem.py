"""Reproject the DEM to the metric CRS and derive slope.

Slope matters a lot in Almaty: the city climbs toward the mountains in
the south, which Phase 4 feeds into Tobler's hiking function (walking)
and uphill penalties (cycling). Slope is computed on the metric grid,
where both the pixel size and the elevation are in meters:

    slope_deg = degrees( arctan( sqrt((dz/dx)^2 + (dz/dy)^2) ) )
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from almaty_access.config import Config
from almaty_access.io import ensure_built

logger = logging.getLogger(__name__)

RAW_FILE = "dem.tif"
DEM_METRIC_FILE = "dem_32643.tif"
SLOPE_FILE = "slope_32643.tif"

NODATA = -9999.0


def compute_slope_deg(elevation: np.ndarray, pixel_x: float, pixel_y: float) -> np.ndarray:
    """Slope in degrees from an elevation grid with pixel sizes in meters.

    NaN cells stay NaN in the output: ``np.gradient`` would otherwise
    compute a finite central difference AT a nodata cell from its two
    valid neighbors — a fabricated slope exactly where there is no data.
    """
    dz_dy, dz_dx = np.gradient(elevation, pixel_y, pixel_x)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    slope[np.isnan(elevation)] = np.nan
    return slope


def prepare_dem(config: Config, *, force: bool = False) -> dict[str, Path]:
    """Write the metric-CRS DEM and its slope raster to data/interim/."""
    raw_path = config.paths.raw / RAW_FILE
    out_dem = config.paths.interim / DEM_METRIC_FILE
    out_slope = config.paths.interim / SLOPE_FILE

    def build() -> None:
        out_dem.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(raw_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, config.crs.metric, src.width, src.height, *src.bounds
            )
            profile = src.profile.copy()
            profile.update(
                crs=config.crs.metric,
                transform=transform,
                width=width,
                height=height,
                dtype="float32",
                nodata=NODATA,
                compress="deflate",
            )
            destination = np.full((height, width), NODATA, dtype="float32")
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=config.crs.metric,
                dst_nodata=NODATA,
                resampling=Resampling.bilinear,
            )
        with rasterio.open(out_dem, "w", **profile) as dst:
            dst.write(destination, 1)

        elevation = np.where(destination == NODATA, np.nan, destination)
        slope = compute_slope_deg(elevation, abs(transform.a), abs(transform.e))
        slope_out = np.where(np.isnan(slope), NODATA, slope).astype("float32")
        with rasterio.open(out_slope, "w", **profile) as dst:
            dst.write(slope_out, 1)
        valid = slope[~np.isnan(slope)]
        logger.info(
            "dem: %dx%d px at %.0f m; slope mean %.1f deg, p95 %.1f deg",
            width,
            height,
            abs(transform.a),
            float(valid.mean()) if valid.size else float("nan"),
            float(np.percentile(valid, 95)) if valid.size else float("nan"),
        )

    ensure_built([out_dem, out_slope], [raw_path, config.source_path], build, force=force)
    return {"dem": out_dem, "slope": out_slope}
