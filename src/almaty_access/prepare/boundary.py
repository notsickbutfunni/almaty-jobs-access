"""Clean the administrative boundary: city, 8 districts, 5 km buffer.

Fixes invalid geometries (``make_valid``), removes slivers (tiny
polygon parts left over from topology fixes), validates that exactly
the configured 8 districts are present, and builds the buffered study
area used to avoid edge effects (README §4). The buffer distance is
applied in the metric CRS; everything is stored in the storage CRS.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from almaty_access.config import Config
from almaty_access.io import ensure_built, read_geoparquet, write_geoparquet

logger = logging.getLogger(__name__)

RAW_FILE = "boundary_osm.parquet"
CITY_FILE = "boundary_city.parquet"
DISTRICTS_FILE = "boundary_districts.parquet"
BUFFER_FILE = "boundary_buffer.parquet"

#: Polygon parts smaller than this (in m²) are dropped as slivers (1 ha).
MIN_PART_M2 = 10_000.0


def _drop_slivers(gdf: gpd.GeoDataFrame, metric_crs: str) -> gpd.GeoDataFrame:
    """Explode multipolygons and drop parts smaller than ``MIN_PART_M2``."""
    metric = gdf.to_crs(metric_crs)
    exploded = metric.explode(index_parts=False)
    keep = exploded.geometry.area >= MIN_PART_M2
    dropped = int((~keep).sum())
    if dropped:
        logger.info("dropped %d sliver polygon part(s) < %.0f m2", dropped, MIN_PART_M2)
    kept = exploded.loc[keep]
    dissolve_cols = [c for c in kept.columns if c != kept.geometry.name]
    dissolved = kept.dissolve(by=dissolve_cols[0]) if dissolve_cols else kept.dissolve()
    return dissolved.reset_index().to_crs(gdf.crs)


def prepare_boundary(config: Config, *, force: bool = False) -> dict[str, Path]:
    """Build the city polygon, the 8 district polygons and the buffer."""
    raw_path = config.paths.raw / RAW_FILE
    out_city = config.paths.interim / CITY_FILE
    out_districts = config.paths.interim / DISTRICTS_FILE
    out_buffer = config.paths.interim / BUFFER_FILE

    def build() -> None:
        raw = read_geoparquet(raw_path)
        raw = raw.set_geometry(raw.geometry.make_valid())

        city_rows = raw[raw["level"] == "city"]
        if city_rows.empty:
            raise ValueError(f"{raw_path}: no rows with level == 'city'")
        city_geom = city_rows.geometry.union_all()
        city = gpd.GeoDataFrame({"name": [config.study_area]}, geometry=[city_geom], crs=raw.crs)

        district_rows = raw[raw["level"] == "district"]
        # OSM's primary 'name' is Kazakh; config names may be Russian, which
        # OSM carries in name:ru. Match case-insensitively across variants
        # (Turksib is tagged with a capitalization quirk in OSM).
        name_columns = [c for c in ("name", "name_ru", "name_kk") if c in district_rows.columns]
        matched: list[dict] = []
        missing: list[str] = []
        for district in config.districts:
            target = district.osm_name.strip().casefold()
            mask = pd.Series(False, index=district_rows.index)
            for column in name_columns:
                values = district_rows[column].astype("string").str.strip().str.casefold()
                mask |= values == target
            rows = district_rows[mask]
            if len(rows) == 0:
                missing.append(district.osm_name)
                continue
            geom = rows.geometry.union_all()
            matched.append(
                {"district": district.key, "name_osm": district.osm_name, "geometry": geom}
            )
        if missing:
            available = sorted(
                set(district_rows[name_columns].astype("string").stack().dropna().tolist())
            )
            raise ValueError(
                f"districts not found in {raw_path.name}: {missing}; names present: {available}"
            )
        districts = gpd.GeoDataFrame(pd.DataFrame(matched), geometry="geometry", crs=raw.crs)
        districts = _drop_slivers(districts, config.crs.metric)
        districts["area_km2"] = (
            districts.to_crs(config.crs.metric).geometry.area / 1_000_000.0
        ).round(2)

        buffer_m = config.buffer_km * 1000.0
        buffered_geom = (
            city.to_crs(config.crs.metric).geometry.buffer(buffer_m).to_crs(raw.crs).iloc[0]
        )
        buffered = gpd.GeoDataFrame(
            {"name": [f"{config.study_area} +{config.buffer_km:g} km"]},
            geometry=[buffered_geom],
            crs=raw.crs,
        )

        write_geoparquet(city, out_city)
        write_geoparquet(districts[["district", "name_osm", "area_km2", "geometry"]], out_districts)
        write_geoparquet(buffered, out_buffer)
        logger.info(
            "boundary: city %.0f km2, %d districts, buffer %.0f km2",
            city.to_crs(config.crs.metric).geometry.area.iloc[0] / 1e6,
            len(districts),
            buffered.to_crs(config.crs.metric).geometry.area.iloc[0] / 1e6,
        )

    ensure_built(
        [out_city, out_districts, out_buffer], [raw_path, config.source_path], build, force=force
    )
    return {"city": out_city, "districts": out_districts, "buffer": out_buffer}
