"""Download OpenStreetMap inputs: boundary, networks, and a PBF extract.

Three access paths, each chosen for what it is best at:

- **osmnx** (Overpass) for the city/district boundary and the three
  routable networks — it builds clean, simplified graphs.
- **Geofabrik PBF + pyosmium** for bulk layers (buildings, POIs, land
  use): one cached country download, then fast local extraction.
- For the bbox-mode sample dataset, a small Overpass XML export replaces
  Geofabrik and is converted to a tiny PBF, so the exact same extractor
  code runs on both.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import shapely

from almaty_access.config import Config
from almaty_access.download.area import study_bbox, study_inputs, study_polygon
from almaty_access.io import ensure_built, write_geoparquet
from almaty_access.prepare.boundary import RAW_FILE as BOUNDARY_FILE
from almaty_access.prepare.network import raw_network_files
from almaty_access.schemas import NETWORK_MODES, POI_TAG_KEYS

logger = logging.getLogger(__name__)

PBF_FILE = "osm_extract.osm.pbf"
BUILDINGS_FILE = "osm_buildings.parquet"
POIS_FILE = "osm_pois.parquet"
LANDUSE_FILE = "osm_landuse.parquet"

DISTRICT_ADMIN_LEVEL = "6"
# The main FOSSGIS instance (overpass-api.de) is frequently saturated and
# refuses connections; kumi.systems runs a high-capacity public mirror.
OVERPASS_API_BASE = "https://overpass.kumi.systems/api"
OVERPASS_URL = f"{OVERPASS_API_BASE}/interpreter"

_BUILDING_EXTRA_KEYS = ("building:levels", "height", "name", "amenity", "shop", "office")


_OVERPASS_BASES = (
    OVERPASS_API_BASE,  # kumi.systems mirror
    "https://overpass-api.de/api",  # main FOSSGIS instance
    "https://maps.mail.ru/osm/tools/overpass/api",  # VK mirror (close to KZ)
    "https://overpass.private.coffee/api",  # community mirror
)


def _tune_osmnx(ox, base: str = OVERPASS_API_BASE) -> None:
    """Point osmnx at an Overpass endpoint with a generous timeout.

    The default 180 s times out on boundary and city-scale network
    queries when the public servers are loaded; only the main instance
    has the /api/status rate-limit endpoint osmnx can poll.
    """
    ox.settings.requests_timeout = 300
    ox.settings.overpass_url = base
    ox.settings.overpass_rate_limit = "overpass-api.de" in base


def _with_overpass_rotation(ox, what: str, operation):
    """Run an osmnx Overpass operation, rotating endpoints on failure.

    Both public instances fail routinely under load (timeouts, 502s), so
    every Overpass-backed download tries each mirror once before giving up.
    """
    last_error: Exception | None = None
    for base in _OVERPASS_BASES:
        _tune_osmnx(ox, base)
        try:
            return operation()
        except Exception as exc:
            logger.warning("%s via %s failed: %s", what, base, exc)
            last_error = exc
    raise RuntimeError(f"all Overpass endpoints failed for {what}") from last_error


def _graph_with_fallback(ox, polygon, mode: str):
    """Build a mode's graph, rotating across Overpass endpoints."""
    return _with_overpass_rotation(
        ox,
        f"the {mode} network",
        lambda: ox.graph_from_polygon(polygon, network_type=mode, simplify=True, retain_all=False),
    )


def download_boundary(config: Config, *, force: bool = False) -> Path:
    """City polygon + district polygons from OSM into data/raw/."""
    output = config.paths.raw / BOUNDARY_FILE

    def build() -> None:
        import osmnx as ox

        _tune_osmnx(ox)
        city = ox.geocode_to_gdf(config.area.place)
        city_geometry = city.geometry.iloc[0]
        logger.info("boundary: geocoded %r", config.area.place)

        # A bbox query is much cheaper for the Overpass server than a
        # polygon filter; clip back to the city polygon afterwards.
        left, bottom, right, top = city.total_bounds
        districts = _with_overpass_rotation(
            ox,
            "the district boundaries",
            lambda: ox.features_from_bbox(
                (left, bottom, right, top),
                tags={"boundary": "administrative", "admin_level": DISTRICT_ADMIN_LEVEL},
            ),
        )
        districts = districts[districts.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        districts = districts[districts.geometry.intersects(city_geometry)]
        # osmnx ORs the tag filters, so the response also contains other
        # admin levels (oblast, villages); keep strictly the district level.
        districts = districts[districts["admin_level"].astype(str) == DISTRICT_ADMIN_LEVEL]
        logger.info("boundary: %d admin_level=%s polygons", len(districts), DISTRICT_ADMIN_LEVEL)

        rows = [
            {
                "name": str(
                    city.get("display_name", city.get("name", pd.Series(["city"]))).iloc[0]
                ),
                "name_ru": None,
                "name_kk": None,
                "admin_level": "4",
                "level": "city",
                "geometry": city_geometry,
            }
        ]
        # OSM tags the districts with Kazakh primary names ('Алмалы ауданы')
        # and Russian in name:ru; keep both so prepare can match either.
        for _, row in districts.iterrows():
            rows.append(
                {
                    "name": row.get("name"),
                    "name_ru": row.get("name:ru"),
                    "name_kk": row.get("name:kk"),
                    "admin_level": str(row.get("admin_level")),
                    "level": "district",
                    "geometry": row.geometry,
                }
            )
        frame = gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs=city.crs)
        write_geoparquet(frame.to_crs(config.crs.storage), output)

    ensure_built(output, [], build, force=force)
    return output


def _flatten(value: object) -> object:
    """OSM attributes can be lists after simplification; join them."""
    if isinstance(value, list):
        return ";".join(str(v) for v in value)
    return value


def download_networks(config: Config, *, force: bool = False) -> dict[str, dict[str, Path]]:
    """Walk, bike and drive graphs: simplified, largest SCC, with lengths."""
    polygon_inputs = study_inputs(config)
    written: dict[str, dict[str, Path]] = {}
    for mode in NETWORK_MODES:
        files = raw_network_files(config, mode)

        def build(mode: str = mode, files: dict[str, Path] = files) -> None:
            import osmnx as ox

            polygon = study_polygon(config)
            logger.info("network %s: querying Overpass (this can take a while)", mode)
            graph = _graph_with_fallback(ox, polygon, mode)
            graph = ox.truncate.largest_component(graph, strongly=True)
            nodes, edges = ox.convert.graph_to_gdfs(graph)

            nodes = nodes.reset_index()
            nodes_out = gpd.GeoDataFrame(
                {
                    "node_id": nodes["osmid"].astype("int64"),
                    "x": nodes["x"].astype(float),
                    "y": nodes["y"].astype(float),
                },
                geometry=nodes.geometry,
                crs=config.crs.storage,
            )

            edges = edges.reset_index()
            edges_out = gpd.GeoDataFrame(
                {
                    "u": edges["u"].astype("int64"),
                    "v": edges["v"].astype("int64"),
                    "key": edges["key"].astype("int64"),
                    "length_m": edges["length"].astype(float).round(1),
                    "highway": edges["highway"].map(_flatten).astype("string"),
                },
                geometry=edges.geometry,
                crs=config.crs.storage,
            )
            write_geoparquet(nodes_out, files["nodes"])
            write_geoparquet(edges_out, files["edges"])
            logger.info(
                "network %s: %d nodes / %d edges (largest SCC)",
                mode,
                len(nodes_out),
                len(edges_out),
            )

        ensure_built([files["nodes"], files["edges"]], polygon_inputs, build, force=force)
        written[mode] = files
    return written


_UA_HEADER = {"User-Agent": "almaty-access-pipeline/0.1 (open-data research)"}


def _overpass_xml(bbox: tuple[float, float, float, float], destination: Path) -> None:
    """Full OSM export of a small bbox via Overpass (sample dataset only).

    Tries the dedicated map-export endpoint first (made exactly for this),
    then falls back to an equivalent interpreter query on the mirror.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    attempts: list[tuple[str, str, dict]] = [
        (
            "GET",
            "https://overpass-api.de/api/map",
            {"params": {"bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}"}},
        ),
        (
            "POST",
            OVERPASS_URL,
            {
                "data": {
                    "data": (
                        f"[out:xml][timeout:300][bbox:{min_lat},{min_lon},{max_lat},{max_lon}];"
                        "nwr;(._;>;);out;"
                    )
                }
            },
        ),
    ]
    errors: list[str] = []
    for method, url, kwargs in attempts:
        try:
            logger.info("overpass: %s %s (bbox %s)", method, url, bbox)
            response = requests.request(method, url, timeout=360, headers=_UA_HEADER, **kwargs)
            response.raise_for_status()
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
            logger.warning("overpass attempt failed: %s", exc)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        logger.info("overpass: wrote %s (%.1f MB)", destination.name, len(response.content) / 1e6)
        return
    raise RuntimeError("all Overpass endpoints failed: " + "; ".join(errors))


def _xml_to_pbf(xml_path: Path, pbf_path: Path) -> None:
    import osmium

    class _Copy(osmium.SimpleHandler):
        def __init__(self, writer: osmium.SimpleWriter) -> None:
            super().__init__()
            self.writer = writer

        def node(self, n) -> None:
            self.writer.add_node(n)

        def way(self, w) -> None:
            self.writer.add_way(w)

        def relation(self, r) -> None:
            self.writer.add_relation(r)

    if pbf_path.exists():
        pbf_path.unlink()
    writer = osmium.SimpleWriter(str(pbf_path))
    try:
        _Copy(writer).apply_file(str(xml_path))
    finally:
        writer.close()


def download_osm_extract(config: Config, *, force: bool = False) -> Path:
    """The PBF the bulk extractor reads: Geofabrik country file, or —
    in bbox mode — a small Overpass export converted to PBF."""
    output = config.paths.raw / PBF_FILE

    def build() -> None:
        if config.area.bbox is not None:
            xml_path = output.with_suffix(".osm")
            _overpass_xml(study_bbox(config, pad_degrees=0.002), xml_path)
            _xml_to_pbf(xml_path, output)
            xml_path.unlink()
            logger.info("osm extract: wrote %s (%.1f MB)", output.name, output.stat().st_size / 1e6)
        else:
            from almaty_access.download.population import _download_file

            _download_file(config.sources.geofabrik_pbf_url, output, timeout_s=1800)

    ensure_built(output, [], build, force=force)
    return output


class _LayerExtractor:
    """Collect buildings, POIs and land use from a PBF within a bbox."""

    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        self.bbox = bbox
        self.buildings: list[dict] = []
        self.pois: list[dict] = []
        self.landuse: list[dict] = []

    def _in_bbox(self, lon: float, lat: float) -> bool:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat

    def handle_node(self, osm_id: str, lon: float, lat: float, tags: dict[str, str]) -> None:
        if not self._in_bbox(lon, lat):
            return
        if any(key in tags for key in POI_TAG_KEYS if key != "building"):
            self.pois.append(self._poi_row(osm_id, shapely.Point(lon, lat), tags))

    def handle_area(self, osm_id: str, geometry: shapely.Geometry, tags: dict[str, str]) -> None:
        point = shapely.get_point(shapely.get_exterior_ring(shapely.get_geometry(geometry, 0)), 0)
        if point is None or not self._in_bbox(point.x, point.y):
            return
        if "building" in tags:
            self.buildings.append(
                {
                    "osm_id": osm_id,
                    "building": tags.get("building"),
                    "building_levels": tags.get("building:levels"),
                    "height": tags.get("height"),
                    "name": tags.get("name"),
                    "amenity": tags.get("amenity"),
                    "shop": tags.get("shop"),
                    "office": tags.get("office"),
                    "geometry": geometry,
                }
            )
        if "landuse" in tags or tags.get("leisure") == "park":
            self.landuse.append(
                {
                    "osm_id": osm_id,
                    "landuse": tags.get("landuse"),
                    "leisure": tags.get("leisure"),
                    "geometry": geometry,
                }
            )
        poi_keys = [k for k in POI_TAG_KEYS if k != "building"]
        if any(key in tags for key in poi_keys) or self._is_job_building(tags):
            # representative_point is guaranteed INSIDE the footprint;
            # a concave building's centroid can fall outside it, which
            # would break the buildings-contain-POIs classification vote.
            self.pois.append(self._poi_row(osm_id, geometry.representative_point(), tags))

    @staticmethod
    def _is_job_building(tags: dict[str, str]) -> bool:
        return tags.get("building") in {
            "office",
            "retail",
            "industrial",
            "warehouse",
            "commercial",
            "manufacture",
        }

    @staticmethod
    def _poi_row(osm_id: str, geometry: shapely.Geometry, tags: dict[str, str]) -> dict:
        row = {"osm_id": osm_id, "name": tags.get("name"), "geometry": geometry}
        for key in POI_TAG_KEYS:
            row[key] = tags.get(key)
        return row


def extract_osm_layers(config: Config, *, force: bool = False) -> dict[str, Path]:
    """Run the PBF extractor; write buildings, POIs and land-use layers."""
    pbf_path = config.paths.raw / PBF_FILE
    outputs = {
        "buildings": config.paths.raw / BUILDINGS_FILE,
        "pois": config.paths.raw / POIS_FILE,
        "landuse": config.paths.raw / LANDUSE_FILE,
    }

    def build() -> None:
        import osmium

        bbox = study_bbox(config, pad_degrees=0.01)
        collector = _LayerExtractor(bbox)
        wkb_factory = osmium.geom.WKBFactory()

        class _Handler(osmium.SimpleHandler):
            def node(self, n) -> None:
                if n.tags:
                    collector.handle_node(
                        f"node/{n.id}", n.location.lon, n.location.lat, dict(n.tags)
                    )

            def area(self, a) -> None:
                tags = dict(a.tags)
                wanted = (
                    "building" in tags
                    or "landuse" in tags
                    or tags.get("leisure") == "park"
                    or any(key in tags for key in POI_TAG_KEYS)
                )
                if not wanted:
                    return
                try:
                    geometry = shapely.from_wkb(bytes.fromhex(wkb_factory.create_multipolygon(a)))
                except RuntimeError:
                    return
                kind = "way" if a.from_way() else "relation"
                collector.handle_area(f"{kind}/{a.orig_id()}", geometry, tags)

        logger.info("extracting layers from %s", pbf_path.name)
        _Handler().apply_file(str(pbf_path), locations=True, idx="flex_mem")

        layers = {
            "buildings": (collector.buildings, outputs["buildings"]),
            "pois": (collector.pois, outputs["pois"]),
            "landuse": (collector.landuse, outputs["landuse"]),
        }
        for name, (rows, path) in layers.items():
            if not rows:
                raise RuntimeError(f"extraction produced no {name} - wrong bbox or empty PBF?")
            frame = gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs="EPSG:4326")
            frame = frame.to_crs(config.crs.storage)
            write_geoparquet(frame, path)
            logger.info("extracted %s: %d features", name, len(frame))

    ensure_built(list(outputs.values()), [pbf_path], build, force=force)
    return outputs
