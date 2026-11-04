"""Load and validate the project configuration.

``config/config.yaml`` is the single source of truth for every tunable
parameter: coordinate reference systems, H3 resolutions, travel-time
thresholds, speeds, data paths and the random seed. Pipeline code never
hard-codes these values; it receives a validated :class:`Config` object,
so a change in the YAML reaches the whole pipeline consistently.

Validation is deliberately strict and fails fast with a clear message:
a config typo should die here, not three pipeline stages later.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pyproj import CRS
from pyproj.exceptions import CRSError

from almaty_access.schemas import BUILDING_FUNCTIONS

DEFAULT_CONFIG_PATH = Path("config") / "config.yaml"

_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_H3_MIN_RESOLUTION = 0
_H3_MAX_RESOLUTION = 15
_REQUIRED_SPEED_MODES = ("walk", "bike")
# numpy.random.seed and default_rng integer seeding accept at most 2**32 - 1.
_MAX_SEED = 2**32 - 1
_METRIC_UNIT_NAMES = {"metre", "meter", "m"}


@dataclass(frozen=True)
class CRSConfig:
    """Coordinate reference systems used throughout the project.

    ``metric`` is a projected CRS (units: meters) used for ALL distance and
    area calculations. ``storage`` is a geographic CRS (units: degrees) used
    only for storage and web maps.
    """

    metric: str
    storage: str

    @property
    def metric_crs(self) -> CRS:
        """The metric CRS as a ``pyproj.CRS`` object."""
        return CRS.from_user_input(self.metric)

    @property
    def storage_crs(self) -> CRS:
        """The storage CRS as a ``pyproj.CRS`` object."""
        return CRS.from_user_input(self.storage)


@dataclass(frozen=True)
class H3Config:
    """H3 grid resolutions: 8 for analysis, 9 for fine maps and walking."""

    analysis_resolution: int
    fine_resolution: int


@dataclass(frozen=True)
class TravelConfig:
    """Travel-time thresholds, flat-ground speeds and the transit window."""

    thresholds_min: tuple[int, ...]
    speeds_kmh: dict[str, float]
    departure_window: tuple[str, str]


@dataclass(frozen=True)
class PathsConfig:
    """Data directories, resolved to absolute paths against the repo root."""

    root: Path
    raw: Path
    interim: Path
    processed: Path
    sample: Path
    figures: Path


@dataclass(frozen=True)
class AreaConfig:
    """What to download: a geocodable place, optionally narrowed to a bbox.

    ``bbox`` is ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
    When set (the sample config does this), downloads that support it use
    the bbox instead of the full buffered city polygon.
    """

    place: str
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class DistrictConfig:
    """One administrative district: our short key + its OSM name tag."""

    key: str
    osm_name: str


@dataclass(frozen=True)
class SourcesConfig:
    """Verified upstream data sources (see DECISIONS.md for provenance)."""

    geofabrik_pbf_url: str
    kontur_gpkg_url: str
    overture_release: str
    dem_tile_url_template: str
    """Template with a ``{name}`` placeholder for the GLO-30 tile name."""


@dataclass(frozen=True)
class BuildingsConfig:
    """Building merge and floor-estimation parameters."""

    iou_threshold: float
    m_per_level: float
    default_levels: dict[str, float]


@dataclass(frozen=True)
class PoisConfig:
    """POI categorization parameters."""

    categories_file: Path


@dataclass(frozen=True)
class SubcentersConfig:
    """DBSCAN parameters for finding job sub-centres from business POIs."""

    eps_m: float
    min_samples: int
    max_centers: int


@dataclass(frozen=True)
class FeaturesConfig:
    """Phase 2 feature-engineering parameters (README §7 Phase 2)."""

    center_lat: float
    center_lon: float
    """Historic centre; distance features are measured from hex centroids."""
    large_gfa_m2: float
    """Buildings with GFA above this count as 'large'."""
    betweenness_k: int
    """Number of sampled sources for betweenness centrality (capped at n)."""
    subcenters: SubcentersConfig
    log1p_prefixes: tuple[str, ...]
    """Feature-name prefixes whose columns get log1p at model time."""
    context_columns: tuple[str, ...]
    """Features that receive k-ring neighbourhood sums and means."""
    context_k: tuple[int, ...]
    corr_threshold: float
    """|r| above this marks a feature pair as near-duplicates."""


@dataclass(frozen=True)
class RandomForestConfig:
    """Random-forest hyperparameters for the Phase 3 job model."""

    n_estimators: int
    min_samples_leaf: int
    max_features: float
    """Fraction of features tried per split, in (0, 1]."""


@dataclass(frozen=True)
class ModelConfig:
    """Phase 3 job-density model parameters (README §7 Phase 3)."""

    baseline_jobs_per_100m2: dict[str, float]
    """Model A coefficients; only relative sizes matter (rescaling)."""
    target_indicator: str
    target_year: int
    training_years: tuple[int, ...]
    random_forest: RandomForestConfig
    quantiles: tuple[float, float, float]
    """Reported uncertainty levels across trees: low / central / high."""


@dataclass(frozen=True)
class Config:
    """The full, validated project configuration."""

    study_area: str
    buffer_km: float
    area: AreaConfig
    districts: tuple[DistrictConfig, ...]
    crs: CRSConfig
    h3: H3Config
    travel: TravelConfig
    paths: PathsConfig
    sources: SourcesConfig
    buildings: BuildingsConfig
    pois: PoisConfig
    features: FeaturesConfig
    model: ModelConfig
    random_seed: int
    source_path: Path
    """The YAML file this configuration was loaded from. Pipeline steps
    list it among their cache inputs, so edits to tunable parameters
    (buffer, IoU threshold, floor defaults, ...) trigger rebuilds."""


def seed_everything(seed: int) -> None:
    """Seed Python's and NumPy's global random number generators.

    Pipeline code should still create explicit local generators with
    ``numpy.random.default_rng(config.random_seed)``; this global seeding is
    a safety net for third-party libraries that use the legacy global state.
    """
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read ``config.yaml``, validate every field and return a :class:`Config`.

    Relative paths in the ``paths`` section are resolved against the
    repository root: the parent of the config file's ``config/`` folder
    (the file normally lives at ``<root>/config/config.yaml``). If the
    file is NOT inside a folder named ``config``, its own folder is used
    as the root, so an ad-hoc config stays self-contained instead of
    silently placing data directories two levels above itself.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If any field is missing, has the wrong type or an invalid value.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"config error: {path} does not contain a YAML mapping")

    resolved = path.resolve()
    root = resolved.parent.parent if resolved.parent.name == "config" else resolved.parent

    project = _section(raw, "project")
    crs_section = _section(raw, "crs")
    h3_section = _section(raw, "h3")
    travel = _section(raw, "travel")
    paths_section = _section(raw, "paths")

    analysis_res = _resolution_from(h3_section.get("analysis_resolution"), "h3.analysis_resolution")
    fine_res = _resolution_from(h3_section.get("fine_resolution"), "h3.fine_resolution")
    if fine_res <= analysis_res:
        raise ValueError(
            "config error: 'h3.fine_resolution' must be finer (larger number) "
            f"than 'h3.analysis_resolution', got {fine_res} <= {analysis_res}"
        )

    return Config(
        study_area=_non_empty_str(project.get("study_area"), "project.study_area"),
        buffer_km=_non_negative_number(project.get("buffer_km"), "project.buffer_km"),
        area=_area_from(_section(raw, "area")),
        districts=_districts_from(raw.get("districts")),
        crs=CRSConfig(
            metric=_crs_from(crs_section.get("metric"), key="crs.metric", expect_projected=True),
            storage=_crs_from(
                crs_section.get("storage"), key="crs.storage", expect_projected=False
            ),
        ),
        h3=H3Config(analysis_resolution=analysis_res, fine_resolution=fine_res),
        travel=TravelConfig(
            thresholds_min=_thresholds_from(travel.get("thresholds_min")),
            speeds_kmh=_speeds_from(travel.get("speeds_kmh")),
            departure_window=_window_from(travel.get("departure_window")),
        ),
        paths=_paths_from(paths_section, root),
        sources=_sources_from(_section(raw, "sources")),
        buildings=_buildings_from(_section(raw, "buildings")),
        pois=_pois_from(_section(raw, "pois"), root),
        features=_features_from(_section(raw, "features")),
        model=_model_from(_section(raw, "model")),
        random_seed=_seed_from(raw.get("random_seed")),
        source_path=resolved,
    )


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config error: section '{key}' is missing or not a mapping")
    return value


def _non_empty_str(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config error: '{key}' must be a non-empty string")
    return value


def _is_number(value: Any) -> bool:
    """A finite real number: bool, NaN and infinity do not count."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _non_negative_number(value: Any, key: str) -> float:
    if not _is_number(value) or value < 0:
        raise ValueError(f"config error: '{key}' must be a non-negative number")
    return float(value)


def _positive_int(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config error: '{key}' must be a positive integer")
    return value


def _seed_from(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not (0 <= value <= _MAX_SEED):
        raise ValueError(
            f"config error: 'random_seed' must be an integer between 0 and {_MAX_SEED} "
            "(the range NumPy accepts)"
        )
    return value


def _crs_from(value: Any, *, key: str, expect_projected: bool) -> str:
    code = _non_empty_str(value, key)
    try:
        crs = CRS.from_user_input(code)
    except CRSError as exc:
        raise ValueError(f"config error: '{key}' is not a valid CRS: {code!r}") from exc
    if expect_projected:
        if not crs.is_projected:
            raise ValueError(
                f"config error: '{key}' must be a projected CRS (meters) "
                f"for distance and area math, got {code!r}"
            )
        units = {axis.unit_name.lower() for axis in crs.axis_info}
        if not units <= _METRIC_UNIT_NAMES:
            raise ValueError(
                f"config error: '{key}' must use metre units for distance and area math, "
                f"but {code!r} uses {sorted(units)}"
            )
    if not expect_projected and not crs.is_geographic:
        raise ValueError(
            f"config error: '{key}' must be a geographic CRS (degrees) "
            f"for storage and web maps, got {code!r}"
        )
    return code


def _resolution_from(value: Any, key: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not (_H3_MIN_RESOLUTION <= value <= _H3_MAX_RESOLUTION)
    ):
        raise ValueError(
            f"config error: '{key}' must be an integer H3 resolution between "
            f"{_H3_MIN_RESOLUTION} and {_H3_MAX_RESOLUTION}"
        )
    return value


def _thresholds_from(value: Any) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(t, int) and not isinstance(t, bool) and t > 0 for t in value)
    ):
        raise ValueError(
            "config error: 'travel.thresholds_min' must be a non-empty list of positive integers"
        )
    if any(later <= earlier for earlier, later in itertools.pairwise(value)):
        raise ValueError("config error: 'travel.thresholds_min' must be strictly increasing")
    return tuple(value)


def _speeds_from(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError("config error: 'travel.speeds_kmh' must be a mapping of mode -> km/h")
    speeds: dict[str, float] = {}
    for mode, speed in value.items():
        if not _is_number(speed) or speed <= 0:
            raise ValueError(f"config error: 'travel.speeds_kmh.{mode}' must be a positive number")
        speeds[str(mode)] = float(speed)
    for required in _REQUIRED_SPEED_MODES:
        if required not in speeds:
            raise ValueError(f"config error: 'travel.speeds_kmh' is missing mode '{required}'")
    return speeds


def _window_from(value: Any) -> tuple[str, str]:
    valid = (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(v, str) and _TIME_RE.match(v) for v in value)
    )
    if not valid:
        raise ValueError(
            "config error: 'travel.departure_window' must be two 'HH:MM' strings, "
            "e.g. ['07:00', '09:00']"
        )
    start, end = value
    if start >= end:
        raise ValueError("config error: 'travel.departure_window' start must be before end")
    return (start, end)


def _bbox_from(value: Any) -> tuple[float, float, float, float]:
    if not (isinstance(value, list) and len(value) == 4 and all(_is_number(v) for v in value)):
        raise ValueError("config error: 'area.bbox' must be [min_lon, min_lat, max_lon, max_lat]")
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in value)
    if not (-180 <= min_lon < max_lon <= 180) or not (-90 <= min_lat < max_lat <= 90):
        raise ValueError(
            "config error: 'area.bbox' must satisfy -180 <= min_lon < max_lon <= 180 "
            "and -90 <= min_lat < max_lat <= 90"
        )
    return (min_lon, min_lat, max_lon, max_lat)


def _area_from(section: dict[str, Any]) -> AreaConfig:
    place = _non_empty_str(section.get("place"), "area.place")
    bbox_raw = section.get("bbox")
    bbox = None if bbox_raw is None else _bbox_from(bbox_raw)
    return AreaConfig(place=place, bbox=bbox)


def _districts_from(value: Any) -> tuple[DistrictConfig, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("config error: 'districts' must be a non-empty list")
    districts: list[DistrictConfig] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(
                f"config error: 'districts[{index}]' must be a mapping with key and osm_name"
            )
        districts.append(
            DistrictConfig(
                key=_non_empty_str(item.get("key"), f"districts[{index}].key"),
                osm_name=_non_empty_str(item.get("osm_name"), f"districts[{index}].osm_name"),
            )
        )
    keys = [district.key for district in districts]
    if len(set(keys)) != len(keys):
        raise ValueError("config error: district keys must be unique")
    return tuple(districts)


def _url_from(value: Any, key: str) -> str:
    url = _non_empty_str(value, key)
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"config error: '{key}' must be an http(s) URL")
    return url


def _sources_from(section: dict[str, Any]) -> SourcesConfig:
    template = _url_from(section.get("dem_tile_url_template"), "sources.dem_tile_url_template")
    if "{name}" not in template:
        raise ValueError(
            "config error: 'sources.dem_tile_url_template' must contain a '{name}' placeholder"
        )
    return SourcesConfig(
        geofabrik_pbf_url=_url_from(section.get("geofabrik_pbf_url"), "sources.geofabrik_pbf_url"),
        kontur_gpkg_url=_url_from(section.get("kontur_gpkg_url"), "sources.kontur_gpkg_url"),
        overture_release=_non_empty_str(
            section.get("overture_release"), "sources.overture_release"
        ),
        dem_tile_url_template=template,
    )


def _buildings_from(section: dict[str, Any]) -> BuildingsConfig:
    iou = section.get("iou_threshold")
    if not _is_number(iou) or not (0 < iou <= 1):
        raise ValueError("config error: 'buildings.iou_threshold' must be a number in (0, 1]")
    m_per_level = section.get("m_per_level")
    if not _is_number(m_per_level) or m_per_level <= 0:
        raise ValueError("config error: 'buildings.m_per_level' must be a positive number")
    defaults_raw = section.get("default_levels")
    if not isinstance(defaults_raw, dict) or not defaults_raw:
        raise ValueError(
            "config error: 'buildings.default_levels' must map building function -> levels"
        )
    defaults: dict[str, float] = {}
    for function, levels in defaults_raw.items():
        if not _is_number(levels) or levels <= 0:
            raise ValueError(
                f"config error: 'buildings.default_levels.{function}' must be a positive number"
            )
        defaults[str(function)] = float(levels)
    missing = [f for f in BUILDING_FUNCTIONS if f not in defaults]
    if missing:
        raise ValueError(
            f"config error: 'buildings.default_levels' is missing functions: {missing}"
        )
    return BuildingsConfig(
        iou_threshold=float(iou), m_per_level=float(m_per_level), default_levels=defaults
    )


def _str_tuple(value: Any, key: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(v, str) and v.strip() for v in value)
    ):
        raise ValueError(f"config error: '{key}' must be a non-empty list of non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"config error: '{key}' must not contain duplicates")
    return tuple(value)


def _features_from(section: dict[str, Any]) -> FeaturesConfig:
    center = section.get("center")
    if not isinstance(center, dict):
        raise ValueError("config error: 'features.center' must be a mapping with 'lat' and 'lon'")
    lat, lon = center.get("lat"), center.get("lon")
    if not _is_number(lat) or not (-90 <= lat <= 90):
        raise ValueError("config error: 'features.center.lat' must be a latitude in [-90, 90]")
    if not _is_number(lon) or not (-180 <= lon <= 180):
        raise ValueError("config error: 'features.center.lon' must be a longitude in [-180, 180]")

    large = section.get("large_gfa_m2")
    if not _is_number(large) or large <= 0:
        raise ValueError("config error: 'features.large_gfa_m2' must be a positive number")

    sub_raw = section.get("subcenters")
    if not isinstance(sub_raw, dict):
        raise ValueError(
            "config error: 'features.subcenters' must be a mapping with "
            "eps_m, min_samples and max_centers"
        )
    eps = sub_raw.get("eps_m")
    if not _is_number(eps) or eps <= 0:
        raise ValueError("config error: 'features.subcenters.eps_m' must be a positive number")
    subcenters = SubcentersConfig(
        eps_m=float(eps),
        min_samples=_positive_int(sub_raw.get("min_samples"), "features.subcenters.min_samples"),
        max_centers=_positive_int(sub_raw.get("max_centers"), "features.subcenters.max_centers"),
    )

    context_k = section.get("context_k")
    valid_k = (
        isinstance(context_k, list)
        and bool(context_k)
        and all(isinstance(k, int) and not isinstance(k, bool) and k > 0 for k in context_k)
        and all(later > earlier for earlier, later in itertools.pairwise(context_k))
    )
    if not valid_k:
        raise ValueError(
            "config error: 'features.context_k' must be a strictly increasing list "
            "of positive integers"
        )

    threshold = section.get("corr_threshold")
    if not _is_number(threshold) or not (0 < threshold <= 1):
        raise ValueError("config error: 'features.corr_threshold' must be a number in (0, 1]")

    return FeaturesConfig(
        center_lat=float(lat),
        center_lon=float(lon),
        large_gfa_m2=float(large),
        betweenness_k=_positive_int(section.get("betweenness_k"), "features.betweenness_k"),
        subcenters=subcenters,
        log1p_prefixes=_str_tuple(section.get("log1p_prefixes"), "features.log1p_prefixes"),
        context_columns=_str_tuple(section.get("context_columns"), "features.context_columns"),
        context_k=tuple(context_k),
        corr_threshold=float(threshold),
    )


def _model_from(section: dict[str, Any]) -> ModelConfig:
    coefficients_raw = section.get("baseline_jobs_per_100m2")
    if not isinstance(coefficients_raw, dict) or not coefficients_raw:
        raise ValueError(
            "config error: 'model.baseline_jobs_per_100m2' must map building function -> number"
        )
    coefficients: dict[str, float] = {}
    for function, value in coefficients_raw.items():
        if not _is_number(value) or value < 0:
            raise ValueError(
                f"config error: 'model.baseline_jobs_per_100m2.{function}' must be >= 0"
            )
        coefficients[str(function)] = float(value)
    missing = [f for f in BUILDING_FUNCTIONS if f not in coefficients]
    if missing:
        raise ValueError(
            f"config error: 'model.baseline_jobs_per_100m2' is missing functions: {missing}"
        )
    if sum(coefficients.values()) <= 0:
        raise ValueError(
            "config error: 'model.baseline_jobs_per_100m2' must have at least one "
            "positive coefficient (an all-zero surface cannot be rescaled)"
        )

    def _year(value: Any, key: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or not (2000 <= value <= 2100):
            raise ValueError(f"config error: '{key}' must be a year between 2000 and 2100")
        return value

    years_raw = section.get("training_years")
    valid_years = (
        isinstance(years_raw, list)
        and bool(years_raw)
        # Validate element types BEFORE comparing: a mixed list like
        # ["2021", 2022] must give a config error, not a TypeError.
        and all(isinstance(y, int) and not isinstance(y, bool) for y in years_raw)
        and all(later > earlier for earlier, later in itertools.pairwise(years_raw))
    )
    if not valid_years:
        raise ValueError(
            "config error: 'model.training_years' must be a strictly increasing list of years"
        )
    training_years = tuple(_year(year, "model.training_years") for year in years_raw)

    forest_raw = section.get("random_forest")
    if not isinstance(forest_raw, dict):
        raise ValueError(
            "config error: 'model.random_forest' must be a mapping with "
            "n_estimators, min_samples_leaf and max_features"
        )
    max_features = forest_raw.get("max_features")
    if not _is_number(max_features) or not (0 < max_features <= 1):
        raise ValueError(
            "config error: 'model.random_forest.max_features' must be a fraction in (0, 1]"
        )
    forest = RandomForestConfig(
        n_estimators=_positive_int(
            forest_raw.get("n_estimators"), "model.random_forest.n_estimators"
        ),
        min_samples_leaf=_positive_int(
            forest_raw.get("min_samples_leaf"), "model.random_forest.min_samples_leaf"
        ),
        max_features=float(max_features),
    )

    quantiles_raw = section.get("quantiles")
    valid_quantiles = (
        isinstance(quantiles_raw, list)
        and len(quantiles_raw) == 3
        and all(_is_number(q) and 0 < q < 1 for q in quantiles_raw)
        and quantiles_raw[0] < quantiles_raw[1] < quantiles_raw[2]
    )
    if not valid_quantiles:
        raise ValueError(
            "config error: 'model.quantiles' must be three strictly increasing values in (0, 1)"
        )

    return ModelConfig(
        baseline_jobs_per_100m2=coefficients,
        target_indicator=_non_empty_str(section.get("target_indicator"), "model.target_indicator"),
        target_year=_year(section.get("target_year"), "model.target_year"),
        training_years=training_years,
        random_forest=forest,
        quantiles=(float(quantiles_raw[0]), float(quantiles_raw[1]), float(quantiles_raw[2])),
    )


def _pois_from(section: dict[str, Any], root: Path) -> PoisConfig:
    value = _non_empty_str(section.get("categories_file"), "pois.categories_file")
    candidate = Path(value)
    resolved = candidate if candidate.is_absolute() else root / candidate
    if not resolved.exists():
        raise ValueError(f"config error: 'pois.categories_file' does not exist: {resolved}")
    return PoisConfig(categories_file=resolved)


def _paths_from(section: dict[str, Any], root: Path) -> PathsConfig:
    resolved: dict[str, Path] = {}
    for key in ("raw", "interim", "processed", "sample", "figures"):
        value = _non_empty_str(section.get(key), f"paths.{key}")
        candidate = Path(value)
        resolved[key] = candidate if candidate.is_absolute() else root / candidate
    return PathsConfig(root=root, **resolved)
