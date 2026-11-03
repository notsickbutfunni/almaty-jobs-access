"""Download Overture Maps building footprints for the study area.

Overture distributes GeoParquet on S3; the ``overturemaps`` package
streams just the bbox we ask for. We keep the columns the merge step
needs (see ``schemas.RAW_OVERTURE_BUILDINGS``) and store polygons in the
storage CRS.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

from almaty_access.config import Config
from almaty_access.download.area import study_bbox, study_inputs
from almaty_access.io import ensure_built, write_geoparquet

logger = logging.getLogger(__name__)

FILE_NAME = "overture_buildings.parquet"


def download_overture_buildings(config: Config, *, force: bool = False) -> Path:
    """Fetch the Overture buildings theme clipped to the study bbox."""
    output = config.paths.raw / FILE_NAME

    def build() -> None:
        from overturemaps import record_batch_reader  # heavy import, keep local

        bbox = study_bbox(config)
        logger.info("overture: downloading buildings for bbox %s", bbox)
        reader = record_batch_reader("building", bbox=bbox)
        table = reader.read_all()
        frame = gpd.GeoDataFrame.from_arrow(table)
        if frame.crs is None:
            frame = frame.set_crs(config.crs.storage)
        frame = frame.to_crs(config.crs.storage)

        result = gpd.GeoDataFrame(
            {
                "overture_id": frame["id"].astype(str),
                "subtype": frame.get("subtype"),
                "class": frame.get("class"),
                "height": frame.get("height"),
                "num_floors": frame.get("num_floors"),
            },
            geometry=frame.geometry,
            crs=frame.crs,
        )
        # Keep polygonal footprints only.
        result = result[result.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        logger.info("overture: %d building footprints", len(result))
        write_geoparquet(result, output)

    ensure_built(output, study_inputs(config), build, force=force)
    return output
