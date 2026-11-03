"""Build the small synthetic sample dataset used by tests and CI.

The sample is a stand-in for real data: a rectangular "study area" in
central Almaty and a seeded scatter of synthetic points of interest
inside it. It exists so that every test and the CI pipeline run offline
in seconds. Phase 1 extends ``data/sample/`` with schema-faithful clips
of the real layers.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box

from almaty_access.config import Config
from almaty_access.io import write_geoparquet

POI_CATEGORIES = ("office", "retail", "education", "health", "industrial")

BOUNDARY_FILE = "boundary.parquet"
POI_FILE = "poi_points.parquet"


def build_sample(config: Config, out_dir: Path | None = None) -> dict[str, Path]:
    """Generate the synthetic sample layers and write them as GeoParquet.

    The result is fully determined by ``config.sample`` and
    ``config.random_seed``: running the function twice produces identical
    layers, which the test suite verifies.

    Parameters
    ----------
    config:
        Validated project configuration.
    out_dir:
        Where to write the files; defaults to ``config.paths.sample``.

    Returns
    -------
    dict[str, Path]
        Mapping of layer name ("boundary", "poi_points") to written file.
    """
    out = Path(out_dir) if out_dir is not None else config.paths.sample
    min_lon, min_lat, max_lon, max_lat = config.sample.bbox
    rng = np.random.default_rng(config.random_seed)

    boundary = gpd.GeoDataFrame(
        {"name": ["central-almaty-sample"]},
        geometry=[box(min_lon, min_lat, max_lon, max_lat)],
        crs=config.crs.storage,
    )

    n = config.sample.n_points
    lons = rng.uniform(min_lon, max_lon, size=n)
    lats = rng.uniform(min_lat, max_lat, size=n)
    categories = rng.choice(POI_CATEGORIES, size=n).tolist()
    weights = rng.uniform(0.5, 5.0, size=n).round(3)
    pois = gpd.GeoDataFrame(
        {"category": categories, "weight": weights},
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats, strict=True)],
        crs=config.crs.storage,
    )

    return {
        "boundary": write_geoparquet(boundary, out / BOUNDARY_FILE),
        "poi_points": write_geoparquet(pois, out / POI_FILE),
    }
