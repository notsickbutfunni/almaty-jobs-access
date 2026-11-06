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

from almaty_access.schemas import BUILDING_FUNCTIONS, EQUITY_BENEFIT_PREFIXES, TRANSIT_MODES

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
class TransitAssumptionsConfig:
    """Assumed parameters of the OSM-fallback transit network (Phase 4).

    No public GTFS exists for Almaty (DECISIONS D-016), so transit travel
    times come from OSM route relations plus these ASSUMPTIONS. They are
    deliberately editable: anyone with real operational data (commercial
    speeds, peak headways) should refine them here.
    """

    speeds_kmh: dict[str, float]
    """Assumed commercial (stop-to-stop, dwell included) speed per mode."""
    headways_min: dict[str, float]
    """Assumed peak headway per mode; waiting time = half the headway."""
    circuity: dict[str, float]
    """Path distance between consecutive stops is approximated as
    straight-line distance times this per-mode factor (>= 1). Street
    modes wind through the grid (~1.3); metro tunnels run nearly
    straight (~1.05)."""


@dataclass(frozen=True)
class AccessConfig:
    """Phase 4 accessibility parameters (README §7 Phase 4)."""

    gravity_half_life_min: float
    """Travel time at which a job's gravity weight halves: beta = ln2/T."""
    sfca_catchment_min: int
    """Walk catchment (minutes) of the two-step floating catchment area."""
    park_min_m2: float
    """Smaller green areas do not count as park destinations (README: >1 ha)."""
    snap_max_m: float
    """Hexes whose centroid is farther than this from the network are
    treated as off-network and excluded from matrices (counted in the report)."""
    transit: TransitAssumptionsConfig
    car_speeds_kmh: dict[str, float]
    """Free-flow car speed by OSM highway class; must include 'default'."""


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
class EquityConfig:
    """Phase 5 equity and spatial-statistics parameters (README §7 Phase 5)."""

    min_population: float
    """Hexes with fewer residents are uninhabited: they join no
    distribution, LISA run or desert ranking (equity is about people)."""
    headline_analysis: tuple[str, ...]
    """Benefit columns of the ANALYSIS-resolution access table that get
    the full treatment (Lorenz curve, district table, LISA, deserts,
    bivariate classes). The FIRST entry drives the one-sentence result."""
    headline_fine: tuple[str, ...]
    """The same for the FINE-resolution table — walking and service
    measures, which are granularity noise at the coarse resolution."""
    lisa_permutations: int
    lisa_alpha: float
    """Significance level for LISA pseudo p-values (two-sided)."""
    desert_pop_quantile: float
    desert_access_quantile: float
    """Access desert = population above the first quantile (across
    hexes) AND access at or below the second, taken across RESIDENTS
    (README: 75th / 25th percentile; DECISIONS D-055)."""
    job_share_cutoffs_pct: tuple[float, ...]
    """'Share of residents reaching fewer than X% of city jobs' cutoffs."""
    headline_cutoff_pct: float
    """The cutoff quoted in the one-sentence result (one of the above)."""
    nearest_thresholds_min: tuple[int, ...]
    """Walking-time thresholds for 'share of residents within T minutes
    of the nearest facility'."""
    center_distance_bands_km: tuple[float, ...]
    """Upper edges of the distance-to-centre bands (a last open band
    collects everything beyond)."""


@dataclass(frozen=True)
class ScenariosConfig:
    """Phase 6 scenario-analysis parameters (README §7 Phase 6)."""

    dir: Path
    """Directory of scenario GeoJSON files (one scenario per file)."""
    attach_max_m: float
    """An ``extends`` line claims an existing route only when one of the
    line's endpoints lies within this distance of the route's first or
    last stop — farther means a drawing error, not an extension."""
    stop_spacing_m: float
    """Default spacing of synthesized stops along a new line that has no
    explicit stop points (overridable per line)."""
    bike_buffer_m: float
    """Default half-width of a bike_boost corridor: bike edges mostly
    inside it get the boosted flat speed (overridable per feature)."""
    gain_cutoffs_jobs: tuple[float, ...]
    """Cutoffs for "residents gaining more than X jobs" (README Phase 6),
    applied to the per-hex gain at the headline threshold."""


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
    access: AccessConfig
    equity: EquityConfig
    scenarios: ScenariosConfig
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
        access=_access_from(
            _section(raw, "access"), _thresholds_from(travel.get("thresholds_min"))
        ),
        equity=_equity_from(_section(raw, "equity")),
        scenarios=_scenarios_from(_section(raw, "scenarios"), root),
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


def _mode_numbers(value: Any, key: str) -> dict[str, float]:
    """A mapping with a positive number for EXACTLY the transit modes."""
    if not isinstance(value, dict) or sorted(value) != sorted(TRANSIT_MODES):
        raise ValueError(f"config error: '{key}' must map exactly the modes {list(TRANSIT_MODES)}")
    numbers: dict[str, float] = {}
    for mode, number in value.items():
        if not _is_number(number) or number <= 0:
            raise ValueError(f"config error: '{key}.{mode}' must be a positive number")
        numbers[str(mode)] = float(number)
    return numbers


def _access_from(section: dict[str, Any], thresholds_min: tuple[int, ...]) -> AccessConfig:
    half_life = section.get("gravity_half_life_min")
    if not _is_number(half_life) or half_life <= 0:
        raise ValueError("config error: 'access.gravity_half_life_min' must be a positive number")

    catchment = section.get("sfca_catchment_min")
    if (
        not isinstance(catchment, int)
        or isinstance(catchment, bool)
        or not (0 < catchment <= max(thresholds_min))
    ):
        raise ValueError(
            "config error: 'access.sfca_catchment_min' must be a positive integer "
            f"<= max(travel.thresholds_min) = {max(thresholds_min)} "
            "(the catchment must fit inside the stored travel-time matrices)"
        )

    park_min = section.get("park_min_m2")
    if not _is_number(park_min) or park_min < 0:
        raise ValueError("config error: 'access.park_min_m2' must be a non-negative number")

    snap_max = section.get("snap_max_m")
    if not _is_number(snap_max) or snap_max <= 0:
        raise ValueError("config error: 'access.snap_max_m' must be a positive number")

    transit_raw = section.get("transit")
    if not isinstance(transit_raw, dict):
        raise ValueError(
            "config error: 'access.transit' must be a mapping with "
            "speeds_kmh, headways_min and circuity"
        )
    circuity = _mode_numbers(transit_raw.get("circuity"), "access.transit.circuity")
    below_one = [mode for mode, value in circuity.items() if value < 1]
    if below_one:
        raise ValueError(
            f"config error: 'access.transit.circuity' must be >= 1 for every mode "
            f"(paths are never shorter than the straight line), got < 1 for {below_one}"
        )
    transit = TransitAssumptionsConfig(
        speeds_kmh=_mode_numbers(transit_raw.get("speeds_kmh"), "access.transit.speeds_kmh"),
        headways_min=_mode_numbers(transit_raw.get("headways_min"), "access.transit.headways_min"),
        circuity=circuity,
    )

    car_raw = section.get("car_speeds_kmh")
    if not isinstance(car_raw, dict) or not car_raw:
        raise ValueError("config error: 'access.car_speeds_kmh' must map highway class -> km/h")
    car_speeds: dict[str, float] = {}
    for highway, speed in car_raw.items():
        if not _is_number(speed) or speed <= 0:
            raise ValueError(
                f"config error: 'access.car_speeds_kmh.{highway}' must be a positive number"
            )
        car_speeds[str(highway)] = float(speed)
    if "default" not in car_speeds:
        raise ValueError(
            "config error: 'access.car_speeds_kmh' must include a 'default' entry "
            "(fallback for unlisted highway classes)"
        )

    return AccessConfig(
        gravity_half_life_min=float(half_life),
        sfca_catchment_min=catchment,
        park_min_m2=float(park_min),
        snap_max_m=float(snap_max),
        transit=transit,
        car_speeds_kmh=car_speeds,
    )


def _increasing_numbers(value: Any, key: str, *, upper: float | None = None) -> tuple[float, ...]:
    """A non-empty, strictly increasing list of positive numbers (<= upper)."""
    valid = (
        isinstance(value, list)
        and bool(value)
        and all(_is_number(v) and v > 0 for v in value)
        and all(later > earlier for earlier, later in itertools.pairwise(value))
        and (upper is None or value[-1] <= upper)
    )
    if not valid:
        bound = "" if upper is None else f" (at most {upper:g})"
        raise ValueError(
            f"config error: '{key}' must be a strictly increasing list of positive numbers{bound}"
        )
    return tuple(float(v) for v in value)


def _open_unit_fraction(value: Any, key: str) -> float:
    if not _is_number(value) or not (0 < value < 1):
        raise ValueError(f"config error: '{key}' must be a number strictly between 0 and 1")
    return float(value)


def _equity_from(section: dict[str, Any]) -> EquityConfig:
    min_population = section.get("min_population")
    if not _is_number(min_population) or min_population <= 0:
        raise ValueError(
            "config error: 'equity.min_population' must be a positive number "
            "(zero-resident hexes carry no weight in any population-weighted measure)"
        )

    headline_raw = section.get("headline_measures")
    if not isinstance(headline_raw, dict) or sorted(headline_raw) != ["analysis", "fine"]:
        raise ValueError(
            "config error: 'equity.headline_measures' must be a mapping with exactly "
            "the keys 'analysis' and 'fine' (one measure list per H3 resolution)"
        )
    headline: dict[str, tuple[str, ...]] = {}
    for level in ("analysis", "fine"):
        key = f"equity.headline_measures.{level}"
        measures = _str_tuple(headline_raw.get(level), key)
        not_benefit = [m for m in measures if not m.startswith(EQUITY_BENEFIT_PREFIXES)]
        if not_benefit:
            raise ValueError(
                f"config error: '{key}' entries must be benefit columns starting with one "
                f"of {list(EQUITY_BENEFIT_PREFIXES)} (more = better); got {not_benefit}"
            )
        headline[level] = measures

    permutations = section.get("lisa_permutations")
    if not isinstance(permutations, int) or isinstance(permutations, bool) or permutations < 99:
        raise ValueError(
            "config error: 'equity.lisa_permutations' must be an integer >= 99 "
            "(fewer permutations cannot resolve p-values below 0.01)"
        )
    alpha = section.get("lisa_alpha")
    if not _is_number(alpha) or not (0 < alpha <= 0.5):
        raise ValueError("config error: 'equity.lisa_alpha' must be a number in (0, 0.5]")
    smallest_p = 2.0 / (permutations + 1)  # two-sided pseudo p-values
    if alpha < smallest_p:
        raise ValueError(
            "config error: 'equity.lisa_alpha' is below the smallest two-sided pseudo "
            f"p-value 2/(permutations+1) = {smallest_p:.4f}; nothing could ever be significant"
        )

    cutoffs = _increasing_numbers(
        section.get("job_share_cutoffs_pct"), "equity.job_share_cutoffs_pct", upper=100.0
    )
    headline_cutoff = section.get("headline_cutoff_pct")
    if not _is_number(headline_cutoff) or float(headline_cutoff) not in cutoffs:
        raise ValueError(
            "config error: 'equity.headline_cutoff_pct' must be one of "
            f"'equity.job_share_cutoffs_pct' {list(cutoffs)}"
        )

    thresholds = section.get("nearest_thresholds_min")
    valid_thresholds = (
        isinstance(thresholds, list)
        and bool(thresholds)
        and all(isinstance(t, int) and not isinstance(t, bool) and t > 0 for t in thresholds)
        and all(later > earlier for earlier, later in itertools.pairwise(thresholds))
    )
    if not valid_thresholds:
        raise ValueError(
            "config error: 'equity.nearest_thresholds_min' must be a strictly increasing "
            "list of positive integers"
        )

    return EquityConfig(
        min_population=float(min_population),
        headline_analysis=headline["analysis"],
        headline_fine=headline["fine"],
        lisa_permutations=permutations,
        lisa_alpha=float(alpha),
        desert_pop_quantile=_open_unit_fraction(
            section.get("desert_pop_quantile"), "equity.desert_pop_quantile"
        ),
        desert_access_quantile=_open_unit_fraction(
            section.get("desert_access_quantile"), "equity.desert_access_quantile"
        ),
        job_share_cutoffs_pct=cutoffs,
        headline_cutoff_pct=float(headline_cutoff),
        nearest_thresholds_min=tuple(thresholds),
        center_distance_bands_km=_increasing_numbers(
            section.get("center_distance_bands_km"), "equity.center_distance_bands_km"
        ),
    )


def _scenarios_from(section: dict[str, Any], root: Path) -> ScenariosConfig:
    value = _non_empty_str(section.get("dir"), "scenarios.dir")
    candidate = Path(value)
    resolved = candidate if candidate.is_absolute() else root / candidate
    if not resolved.is_dir():
        raise ValueError(f"config error: 'scenarios.dir' is not a directory: {resolved}")

    def positive(key: str) -> float:
        number = section.get(key)
        if not _is_number(number) or number <= 0:
            raise ValueError(f"config error: 'scenarios.{key}' must be a positive number")
        return float(number)

    return ScenariosConfig(
        dir=resolved,
        attach_max_m=positive("attach_max_m"),
        stop_spacing_m=positive("stop_spacing_m"),
        bike_buffer_m=positive("bike_buffer_m"),
        gain_cutoffs_jobs=_increasing_numbers(
            section.get("gain_cutoffs_jobs"), "scenarios.gain_cutoffs_jobs"
        ),
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
