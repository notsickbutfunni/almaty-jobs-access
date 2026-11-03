"""Tests for configuration loading and validation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from almaty_access.config import Config, load_config


def _mutated_config(config_path: Path, tmp_path: Path, mutate: Callable[[dict], None]) -> Path:
    """Copy the real config, apply ``mutate`` to the parsed dict, write to tmp."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    mutate(raw)
    out = tmp_path / "config.yaml"
    out.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return out


def test_repo_config_is_valid(repo_config: Config) -> None:
    assert repo_config.crs.metric_crs.to_epsg() == 32643
    assert repo_config.crs.metric_crs.is_projected
    assert repo_config.crs.storage_crs.to_epsg() == 4326
    assert repo_config.crs.storage_crs.is_geographic
    assert repo_config.h3.analysis_resolution == 8
    assert repo_config.h3.fine_resolution == 9
    assert repo_config.travel.thresholds_min == (15, 30, 45, 60)
    assert repo_config.travel.speeds_kmh["walk"] == pytest.approx(4.8)
    assert repo_config.travel.speeds_kmh["bike"] == pytest.approx(15.0)
    assert repo_config.random_seed == 42
    assert repo_config.sample.n_points > 0


def test_paths_resolve_against_repo_root(repo_config: Config, config_path: Path) -> None:
    root = config_path.resolve().parent.parent
    assert repo_config.paths.root == root
    assert repo_config.paths.raw == root / "data" / "raw"
    assert repo_config.paths.sample == root / "data" / "sample"
    assert repo_config.paths.figures == root / "reports" / "figures"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does-not-exist.yaml")


def _set(path: str, value: Any) -> Callable[[dict], None]:
    """Build a mutator that sets a dotted key, e.g. 'crs.metric', to a value."""

    def mutate(raw: dict) -> None:
        keys = path.split(".")
        node = raw
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value

    return mutate


def _delete(path: str) -> Callable[[dict], None]:
    def mutate(raw: dict) -> None:
        keys = path.split(".")
        node = raw
        for key in keys[:-1]:
            node = node[key]
        del node[keys[-1]]

    return mutate


INVALID_CASES = [
    pytest.param(_set("crs.metric", "EPSG:4326"), id="metric-crs-not-projected"),
    pytest.param(_set("crs.storage", "EPSG:32643"), id="storage-crs-not-geographic"),
    pytest.param(_set("crs.metric", "not-a-crs"), id="metric-crs-garbage"),
    pytest.param(_set("crs.metric", "EPSG:2263"), id="metric-crs-feet-not-meters"),
    pytest.param(_set("project.buffer_km", float("nan")), id="buffer-nan"),
    pytest.param(_set("h3.fine_resolution", 8), id="fine-res-not-finer"),
    pytest.param(_set("h3.analysis_resolution", 16), id="resolution-out-of-range"),
    pytest.param(_set("travel.thresholds_min", [30, 15, 45]), id="thresholds-not-increasing"),
    pytest.param(_set("travel.thresholds_min", []), id="thresholds-empty"),
    pytest.param(_set("travel.speeds_kmh.walk", -1.0), id="speed-negative"),
    pytest.param(_set("travel.speeds_kmh.walk", float("nan")), id="speed-nan"),
    pytest.param(_set("travel.speeds_kmh.bike", float("inf")), id="speed-inf"),
    pytest.param(_delete("travel.speeds_kmh.walk"), id="speed-walk-missing"),
    pytest.param(_set("travel.departure_window", ["9:00", "11:00"]), id="window-bad-format"),
    pytest.param(_set("travel.departure_window", ["09:00", "07:00"]), id="window-reversed"),
    pytest.param(_set("sample.bbox", [76.97, 43.22, 76.89, 43.28]), id="bbox-min-max-swapped"),
    pytest.param(_set("sample.n_points", 0), id="n-points-zero"),
    pytest.param(_set("random_seed", -1), id="seed-negative"),
    pytest.param(_set("random_seed", 2**40), id="seed-above-numpy-max"),
    pytest.param(_set("random_seed", "42"), id="seed-not-int"),
    pytest.param(_delete("paths.interim"), id="path-missing"),
]


@pytest.mark.parametrize("mutate", INVALID_CASES)
def test_invalid_config_raises(
    config_path: Path, tmp_path: Path, mutate: Callable[[dict], None]
) -> None:
    bad_path = _mutated_config(config_path, tmp_path, mutate)
    with pytest.raises(ValueError):
        load_config(bad_path)


def test_config_outside_config_folder_roots_in_own_folder(
    config_path: Path, tmp_path: Path
) -> None:
    """A config not living in <root>/config/ uses its own folder as root."""
    copied = _mutated_config(config_path, tmp_path, lambda raw: None)
    config = load_config(copied)
    assert config.paths.root == tmp_path
    assert config.paths.raw == tmp_path / "data" / "raw"
