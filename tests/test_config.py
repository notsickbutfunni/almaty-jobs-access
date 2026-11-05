"""Tests for configuration loading and validation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from almaty_access.config import Config, load_config


def _mutated_config(config_path: Path, tmp_path: Path, mutate: Callable[[dict], None]) -> Path:
    """Copy the real config, apply ``mutate`` to the parsed dict, write to tmp.

    The POI categories file is referenced by an absolute path first, so a
    config copied into tmp still passes the file-exists validation (unless
    the mutation under test deliberately breaks it).
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    raw["pois"]["categories_file"] = str((config_path.parent / "poi_categories.yaml").resolve())
    mutate(raw)
    out = tmp_path / "config.yaml"
    out.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
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
    assert len(repo_config.districts) == 8
    assert repo_config.area.place
    assert repo_config.area.bbox is None
    assert repo_config.buildings.iou_threshold == pytest.approx(0.5)
    assert set(repo_config.buildings.default_levels) >= {"residential", "unknown"}
    assert repo_config.pois.categories_file.exists()
    assert repo_config.sources.geofabrik_pbf_url.startswith("https://")
    assert 43 < repo_config.features.center_lat < 44  # inside Almaty
    assert 76 < repo_config.features.center_lon < 78
    assert repo_config.features.large_gfa_m2 == pytest.approx(2000.0)
    assert repo_config.features.subcenters.eps_m > 0
    assert repo_config.features.context_k == (1, 2)
    assert 0 < repo_config.features.corr_threshold <= 1
    assert all(prefix for prefix in repo_config.features.log1p_prefixes)
    assert repo_config.model.target_indicator == "payroll_jobs_located"
    assert set(repo_config.model.baseline_jobs_per_100m2) >= {"commercial", "unknown"}
    assert repo_config.model.quantiles == (0.1, 0.5, 0.9)
    assert repo_config.model.random_forest.n_estimators >= 100
    assert repo_config.access.gravity_half_life_min == pytest.approx(20.0)
    assert repo_config.access.sfca_catchment_min <= max(repo_config.travel.thresholds_min)
    assert set(repo_config.access.transit.speeds_kmh) == {"bus", "trolleybus", "subway"}
    assert set(repo_config.access.transit.headways_min) == {"bus", "trolleybus", "subway"}
    assert set(repo_config.access.transit.circuity) == {"bus", "trolleybus", "subway"}
    assert all(value >= 1.0 for value in repo_config.access.transit.circuity.values())
    circuity = repo_config.access.transit.circuity
    assert circuity["subway"] < circuity["bus"]  # tunnels run straighter than streets
    assert "default" in repo_config.access.car_speeds_kmh
    assert repo_config.access.snap_max_m > 0
    assert repo_config.equity.min_population > 0
    assert repo_config.equity.headline_analysis[0] == "accpct_jobs_transit_45"
    assert all(m.startswith(("acc_", "accpct_", "sfca_")) for m in repo_config.equity.headline_fine)
    assert repo_config.equity.lisa_permutations >= 999
    assert 0 < repo_config.equity.lisa_alpha <= 0.05
    assert repo_config.equity.desert_pop_quantile == pytest.approx(0.75)
    assert repo_config.equity.desert_access_quantile == pytest.approx(0.25)
    assert repo_config.equity.headline_cutoff_pct in repo_config.equity.job_share_cutoffs_pct
    assert 15 in repo_config.equity.nearest_thresholds_min


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
    pytest.param(_set("area.bbox", [76.97, 43.22, 76.89, 43.28]), id="bbox-min-max-swapped"),
    pytest.param(_set("area.place", ""), id="area-place-empty"),
    pytest.param(_set("districts", []), id="districts-empty"),
    pytest.param(
        _set("districts", [{"key": "a", "osm_name": "X"}, {"key": "a", "osm_name": "Y"}]),
        id="districts-duplicate-keys",
    ),
    pytest.param(_set("sources.geofabrik_pbf_url", "ftp://x"), id="source-not-http"),
    pytest.param(
        _set("sources.dem_tile_url_template", "https://example.com/dem.tif"),
        id="dem-template-no-placeholder",
    ),
    pytest.param(_set("buildings.iou_threshold", 1.5), id="iou-out-of-range"),
    pytest.param(_delete("buildings.default_levels.unknown"), id="default-levels-missing"),
    pytest.param(_set("pois.categories_file", "config/nope.yaml"), id="pois-file-missing"),
    pytest.param(_set("random_seed", -1), id="seed-negative"),
    pytest.param(_set("random_seed", 2**40), id="seed-above-numpy-max"),
    pytest.param(_set("random_seed", "42"), id="seed-not-int"),
    pytest.param(_delete("paths.interim"), id="path-missing"),
    pytest.param(_set("features.center.lat", 100), id="center-lat-out-of-range"),
    pytest.param(_set("features.center", [43.2, 76.9]), id="center-not-a-mapping"),
    pytest.param(_set("features.large_gfa_m2", 0), id="large-gfa-zero"),
    pytest.param(_set("features.betweenness_k", 0), id="betweenness-k-zero"),
    pytest.param(_set("features.subcenters.min_samples", 0), id="subcenter-min-samples-zero"),
    pytest.param(_set("features.subcenters.eps_m", -5), id="subcenter-eps-negative"),
    pytest.param(_set("features.log1p_prefixes", []), id="log1p-prefixes-empty"),
    pytest.param(_set("features.log1p_prefixes", ["bf_", "bf_"]), id="log1p-prefixes-duplicate"),
    pytest.param(_set("features.context_columns", [""]), id="context-columns-empty-string"),
    pytest.param(_set("features.context_k", [2, 1]), id="context-k-not-increasing"),
    pytest.param(_set("features.corr_threshold", 1.5), id="corr-threshold-above-one"),
    pytest.param(_delete("features"), id="features-section-missing"),
    pytest.param(_set("model.baseline_jobs_per_100m2.commercial", -1), id="coefficient-negative"),
    pytest.param(
        _delete("model.baseline_jobs_per_100m2.unknown"), id="coefficient-missing-function"
    ),
    pytest.param(_set("model.target_year", 1999), id="target-year-out-of-range"),
    pytest.param(_set("model.training_years", [2022, 2021]), id="training-years-not-increasing"),
    pytest.param(_set("model.training_years", ["2021", 2022]), id="training-years-mixed-types"),
    pytest.param(_set("model.random_forest.n_estimators", 0), id="rf-estimators-zero"),
    pytest.param(_set("model.random_forest.max_features", 1.5), id="rf-max-features-above-one"),
    pytest.param(_set("model.quantiles", [0.1, 0.9]), id="quantiles-not-three"),
    pytest.param(_set("model.quantiles", [0.5, 0.1, 0.9]), id="quantiles-not-increasing"),
    pytest.param(_delete("model"), id="model-section-missing"),
    pytest.param(_set("access.gravity_half_life_min", 0), id="half-life-zero"),
    pytest.param(_set("access.sfca_catchment_min", 90), id="catchment-above-cutoff"),
    pytest.param(_set("access.sfca_catchment_min", 15.5), id="catchment-not-int"),
    pytest.param(_set("access.park_min_m2", -1), id="park-min-negative"),
    pytest.param(_set("access.snap_max_m", 0), id="snap-max-zero"),
    pytest.param(_set("access.transit.circuity", 1.3), id="circuity-not-a-mapping"),
    pytest.param(_set("access.transit.circuity.subway", 0.8), id="circuity-below-one"),
    pytest.param(_delete("access.transit.speeds_kmh.subway"), id="transit-speed-mode-missing"),
    pytest.param(_set("access.transit.speeds_kmh.bus", -5), id="transit-speed-negative"),
    pytest.param(
        _set("access.transit.headways_min", {"bus": 10, "trolleybus": 12, "subway": 7, "tram": 5}),
        id="transit-headway-unknown-mode",
    ),
    pytest.param(_delete("access.car_speeds_kmh.default"), id="car-speeds-no-default"),
    pytest.param(_set("access.car_speeds_kmh.primary", 0), id="car-speed-zero"),
    pytest.param(_delete("access"), id="access-section-missing"),
    pytest.param(_set("equity.min_population", 0), id="equity-min-population-zero"),
    pytest.param(
        _set("equity.headline_measures", ["accpct_jobs_transit_45"]),
        id="equity-headline-not-a-mapping",
    ),
    pytest.param(_delete("equity.headline_measures.fine"), id="equity-headline-level-missing"),
    pytest.param(_set("equity.headline_measures.analysis", []), id="equity-headline-empty"),
    pytest.param(
        _set("equity.headline_measures.fine", ["near_clinic_walk_min"]),
        id="equity-headline-not-a-benefit-measure",
    ),
    pytest.param(_set("equity.lisa_permutations", 50), id="equity-permutations-too-few"),
    pytest.param(_set("equity.lisa_alpha", 0.0001), id="equity-alpha-below-resolution"),
    pytest.param(_set("equity.lisa_alpha", 0.9), id="equity-alpha-above-half"),
    pytest.param(_set("equity.desert_pop_quantile", 1.0), id="equity-quantile-not-open"),
    pytest.param(_set("equity.desert_access_quantile", 0), id="equity-access-quantile-zero"),
    pytest.param(
        _set("equity.job_share_cutoffs_pct", [5, 1, 10]), id="equity-cutoffs-not-increasing"
    ),
    pytest.param(_set("equity.job_share_cutoffs_pct", [5, 150]), id="equity-cutoff-above-100"),
    pytest.param(_set("equity.headline_cutoff_pct", 7), id="equity-headline-cutoff-not-listed"),
    pytest.param(_set("equity.nearest_thresholds_min", [10, 12.5]), id="equity-thresholds-not-int"),
    pytest.param(_set("equity.center_distance_bands_km", [5, 5]), id="equity-bands-not-increasing"),
    pytest.param(_delete("equity"), id="equity-section-missing"),
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
