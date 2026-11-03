"""Attach terrain to the networks: node elevations and edge grades.

Nodes get an ``elevation_m`` sampled from the metric-CRS DEM; directed
edges get a ``grade`` = (elevation(v) - elevation(u)) / length. Phase 4
turns grades into mode-specific speeds (Tobler's hiking function for
walking, an uphill penalty for cycling).
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from almaty_access.config import Config
from almaty_access.io import ensure_built, read_geoparquet, write_geoparquet
from almaty_access.prepare.dem import DEM_METRIC_FILE
from almaty_access.schemas import NETWORK_MODES

logger = logging.getLogger(__name__)

#: |grade| above this is treated as a data error and clipped. Streets
#: rarely exceed ~15%; residual DSM noise above 30% is never real.
MAX_ABS_GRADE = 0.3

#: Moving-minimum window (pixels) applied to the DSM before sampling
#: node elevations. GLO-30 is a SURFACE model: on a building roof it sits
#: tens of meters above the street, and a local minimum approximates the
#: ground level nearby (3 px at ~27 m resolution = an ~80 m neighborhood).
DEM_MIN_FILTER_PX = 3


def raw_network_files(config: Config, mode: str) -> dict[str, Path]:
    """Paths of the raw node/edge files for a mode."""
    return {
        "nodes": config.paths.raw / f"network_{mode}_nodes.parquet",
        "edges": config.paths.raw / f"network_{mode}_edges.parquet",
    }


def interim_network_files(config: Config, mode: str) -> dict[str, Path]:
    """Paths of the prepared node/edge files for a mode."""
    return {
        "nodes": config.paths.interim / f"network_{mode}_nodes.parquet",
        "edges": config.paths.interim / f"network_{mode}_edges.parquet",
    }


def _ground_approximation(band: np.ndarray, window_px: int) -> np.ndarray:
    """Moving-minimum filter that tolerates NaN cells (kept as NaN)."""
    from scipy.ndimage import minimum_filter

    if window_px <= 1:
        return band
    filled = np.where(np.isnan(band), np.inf, band)
    smoothed = minimum_filter(filled, size=window_px, mode="nearest")
    return np.where(np.isinf(smoothed), np.nan, smoothed)


def attach_elevation(
    nodes: gpd.GeoDataFrame,
    dem_path: Path,
    metric_crs: str,
    *,
    smooth_window_px: int = DEM_MIN_FILTER_PX,
) -> gpd.GeoDataFrame:
    """Add an ``elevation_m`` column sampled from the metric DEM.

    The DEM is a surface model, so it is minimum-filtered first (see
    ``DEM_MIN_FILTER_PX``): without this, nodes under tall buildings get
    roof heights and short edges get absurd grades.
    """
    with rasterio.open(dem_path) as src:
        band = src.read(1).astype("float64")
        if src.nodata is not None:
            band = np.where(band == src.nodata, np.nan, band)
        band = _ground_approximation(band, smooth_window_px)
        metric = nodes.to_crs(metric_crs)
        rows, cols = rasterio.transform.rowcol(
            src.transform, metric.geometry.x.to_numpy(), metric.geometry.y.to_numpy()
        )
    rows = np.asarray(rows, dtype="int64")
    cols = np.asarray(cols, dtype="int64")
    inside = (rows >= 0) & (rows < band.shape[0]) & (cols >= 0) & (cols < band.shape[1])
    elevation = np.full(len(nodes), np.nan)
    elevation[inside] = band[rows[inside], cols[inside]]
    missing = int(np.isnan(elevation).sum())
    if missing:
        logger.warning("%d of %d nodes fall outside the DEM (elevation = NaN)", missing, len(nodes))
    result = nodes.copy()
    result["elevation_m"] = np.round(elevation, 1)
    return result


def edge_grades(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> np.ndarray:
    """Directed grade per edge: (elev_v - elev_u) / length, clipped."""
    elevation_by_node = nodes.set_index("node_id")["elevation_m"]
    elev_u = elevation_by_node.reindex(edges["u"]).to_numpy()
    elev_v = elevation_by_node.reindex(edges["v"]).to_numpy()
    length = edges["length_m"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        grade = np.where(length > 0, (elev_v - elev_u) / length, 0.0)
    extreme = int((np.abs(grade) > MAX_ABS_GRADE).sum())
    if extreme:
        logger.info("clipped %d extreme edge grades to ±%.1f", extreme, MAX_ABS_GRADE)
    return np.clip(np.nan_to_num(grade, nan=0.0), -MAX_ABS_GRADE, MAX_ABS_GRADE)


def prepare_networks(config: Config, *, force: bool = False) -> dict[str, dict[str, Path]]:
    """Add elevations and grades to every mode's network."""
    dem_path = config.paths.interim / DEM_METRIC_FILE
    written: dict[str, dict[str, Path]] = {}
    for mode in NETWORK_MODES:
        raw = raw_network_files(config, mode)
        out = interim_network_files(config, mode)

        def build(raw: dict[str, Path] = raw, out: dict[str, Path] = out, mode: str = mode) -> None:
            nodes = read_geoparquet(raw["nodes"])
            edges = read_geoparquet(raw["edges"])
            nodes = attach_elevation(nodes, dem_path, config.crs.metric)
            edges = edges.copy()
            edges["grade"] = np.round(edge_grades(edges, nodes), 4)
            write_geoparquet(nodes, out["nodes"])
            write_geoparquet(edges, out["edges"])
            logger.info(
                "network %s: %d nodes, %d edges, mean |grade| %.3f",
                mode,
                len(nodes),
                len(edges),
                float(np.abs(edges["grade"]).mean()),
            )

        ensure_built(
            [out["nodes"], out["edges"]],
            [raw["nodes"], raw["edges"], dem_path, config.source_path],
            build,
            force=force,
        )
        written[mode] = out
    return written
