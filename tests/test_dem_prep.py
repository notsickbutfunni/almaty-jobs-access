"""Tests for DEM reprojection and slope computation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from almaty_access.config import Config
from almaty_access.prepare.dem import NODATA, RAW_FILE, compute_slope_deg, prepare_dem


def _write_raster(path: Path, data: np.ndarray, crs: str, transform) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)


def test_slope_on_uniform_ramp() -> None:
    # Elevation rises 3 m per 30 m pixel in x -> gradient 0.1 -> 5.71 degrees.
    elevation = np.tile(np.arange(20, dtype=float) * 3.0, (20, 1))
    slope = compute_slope_deg(elevation, 30.0, 30.0)
    expected = np.degrees(np.arctan(0.1))
    assert np.allclose(slope[5:15, 5:15], expected, atol=0.01)


def test_slope_flat_is_zero_and_nan_propagates() -> None:
    elevation = np.full((10, 10), 700.0)
    elevation[0, 0] = np.nan
    elevation[5, 5] = np.nan  # interior nodata: gradient would fabricate a value here
    slope = compute_slope_deg(elevation, 30.0, 30.0)
    assert np.allclose(slope[2:4, 2:4], 0.0)
    assert np.isnan(slope[0, 0])
    assert np.isnan(slope[5, 5])


def test_prepare_dem_reprojects_and_derives_slope(tmp_config: Config) -> None:
    transform = from_origin(76.90, 43.26, 0.0005, 0.0005)
    elevation = 700.0 + np.tile(np.arange(40, dtype=float) * 2.0, (40, 1))
    _write_raster(tmp_config.paths.raw / RAW_FILE, elevation, "EPSG:4326", transform)

    out = prepare_dem(tmp_config)

    with rasterio.open(out["dem"]) as src:
        assert src.crs.to_epsg() == 32643
        band = src.read(1)
        valid = band[band != NODATA]
        assert valid.size > 0
        assert 690 < valid.mean() < 790
    with rasterio.open(out["slope"]) as src:
        slope = src.read(1)
        valid = slope[slope != NODATA]
        assert valid.size > 0
        assert (valid >= 0).all()
        assert valid.max() < 15  # a gentle ramp stays a gentle slope
