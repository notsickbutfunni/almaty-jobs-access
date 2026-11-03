"""Shared helpers: what spatial extent should downloads cover?

Two modes (config ``area``): a bbox (used by the sample config), or the
city polygon from the already-downloaded boundary, buffered by
``buffer_km`` in the metric CRS (the real pipeline). Downloads that are
polygon-capable use :func:`study_polygon`; bbox-only sources use
:func:`study_bbox`.
"""

from __future__ import annotations

from pathlib import Path

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from almaty_access.config import Config
from almaty_access.io import read_geoparquet
from almaty_access.prepare.boundary import RAW_FILE as BOUNDARY_RAW_FILE


def _buffered_city(config: Config) -> BaseGeometry:
    boundary_path = config.paths.raw / BOUNDARY_RAW_FILE
    raw = read_geoparquet(boundary_path)
    city = raw[raw["level"] == "city"]
    if city.empty:
        raise ValueError(f"{boundary_path}: no rows with level == 'city'")
    buffer_m = config.buffer_km * 1000.0
    return (
        city.to_crs(config.crs.metric).geometry.buffer(buffer_m).to_crs(config.crs.storage)
    ).union_all()


def study_polygon(config: Config) -> BaseGeometry:
    """The polygon downloads should cover (bbox box, or buffered city)."""
    if config.area.bbox is not None:
        return box(*config.area.bbox)
    return _buffered_city(config)


def study_bbox(config: Config, *, pad_degrees: float = 0.0) -> tuple[float, float, float, float]:
    """The bbox downloads should cover, optionally padded (degrees)."""
    min_lon, min_lat, max_lon, max_lat = study_polygon(config).bounds
    return (
        min_lon - pad_degrees,
        min_lat - pad_degrees,
        max_lon + pad_degrees,
        max_lat + pad_degrees,
    )


def study_inputs(config: Config) -> list[Path]:
    """Cache inputs of extent-dependent downloads (the boundary, unless bbox)."""
    if config.area.bbox is not None:
        return []
    return [config.paths.raw / BOUNDARY_RAW_FILE]
