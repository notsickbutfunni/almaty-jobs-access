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
class SampleConfig:
    """Parameters of the synthetic sample dataset used by tests and CI."""

    bbox: tuple[float, float, float, float]
    n_points: int


@dataclass(frozen=True)
class Config:
    """The full, validated project configuration."""

    study_area: str
    buffer_km: float
    crs: CRSConfig
    h3: H3Config
    travel: TravelConfig
    paths: PathsConfig
    sample: SampleConfig
    random_seed: int


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
    sample_section = _section(raw, "sample")

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
        sample=SampleConfig(
            bbox=_bbox_from(sample_section.get("bbox")),
            n_points=_positive_int(sample_section.get("n_points"), "sample.n_points"),
        ),
        random_seed=_seed_from(raw.get("random_seed")),
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
        raise ValueError("config error: 'sample.bbox' must be [min_lon, min_lat, max_lon, max_lat]")
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in value)
    if not (-180 <= min_lon < max_lon <= 180) or not (-90 <= min_lat < max_lat <= 90):
        raise ValueError(
            "config error: 'sample.bbox' must satisfy -180 <= min_lon < max_lon <= 180 "
            "and -90 <= min_lat < max_lat <= 90"
        )
    return (min_lon, min_lat, max_lon, max_lat)


def _paths_from(section: dict[str, Any], root: Path) -> PathsConfig:
    resolved: dict[str, Path] = {}
    for key in ("raw", "interim", "processed", "sample", "figures"):
        value = _non_empty_str(section.get(key), f"paths.{key}")
        candidate = Path(value)
        resolved[key] = candidate if candidate.is_absolute() else root / candidate
    return PathsConfig(root=root, **resolved)
