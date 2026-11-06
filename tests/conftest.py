"""Shared fixtures for the test suite.

All tests run offline on the committed configuration and the committed
sample dataset; nothing here touches the network.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from almaty_access.config import Config, PathsConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config_path() -> Path:
    """Path to the repository's real config file."""
    return REPO_ROOT / "config" / "config.yaml"


@pytest.fixture(scope="session")
def repo_config(config_path: Path) -> Config:
    """The repository's real configuration, loaded once per session."""
    return load_config(config_path)


@pytest.fixture(scope="session")
def sample_config() -> Config:
    """The sample-dataset configuration (paths under data/sample/)."""
    return load_config(REPO_ROOT / "config" / "config.sample.yaml")


@pytest.fixture(scope="session")
def sample_pipeline(sample_config: Config, tmp_path_factory: pytest.TempPathFactory) -> Config:
    """The whole pipeline run ONCE on the committed sample, into tmp.

    prepare -> features -> model -> access; the model, access and equity
    test modules all read this one run (treat its directories as
    read-only: copy before mutating, as the no-transit fixture does).
    """
    from almaty_access.access.build import build_measures, build_times
    from almaty_access.features.build import build_features
    from almaty_access.model.build import build_model
    from almaty_access.prepare.boundary import prepare_boundary
    from almaty_access.prepare.buildings import prepare_buildings
    from almaty_access.prepare.dem import prepare_dem
    from almaty_access.prepare.network import prepare_networks
    from almaty_access.prepare.pois import prepare_pois
    from almaty_access.prepare.population import prepare_population
    from almaty_access.prepare.transit import prepare_transit

    tmp = tmp_path_factory.mktemp("sample_pipeline")
    paths = PathsConfig(
        root=sample_config.paths.root,
        raw=sample_config.paths.raw,
        interim=tmp / "interim",
        processed=tmp / "processed",
        sample=sample_config.paths.sample,
        figures=tmp / "figures",
        app=tmp / "app",
        webmap=tmp / "webmap",
    )
    config = dataclasses.replace(sample_config, paths=paths)
    prepare_boundary(config)
    prepare_dem(config)
    prepare_pois(config)
    prepare_buildings(config)
    prepare_population(config)
    prepare_networks(config)
    prepare_transit(config)
    build_features(config)
    build_model(config)
    build_times(config)
    build_measures(config)
    return config


@pytest.fixture(scope="session")
def dashboard_sample(sample_pipeline: Config) -> Config:
    """The sample run extended with the equity and scenarios stages.

    This is everything the Phase 7 dashboard reads; both dashboard test
    modules share it (the builds are cached, so whichever module runs
    first pays once).
    """
    from almaty_access.equity.build import build_equity
    from almaty_access.scenarios.build import build_scenarios

    build_equity(sample_pipeline)
    build_scenarios(sample_pipeline)
    return sample_pipeline


@pytest.fixture()
def tmp_config(repo_config: Config, tmp_path: Path) -> Config:
    """Repo config with all data paths redirected into tmp_path."""
    paths = PathsConfig(
        root=tmp_path,
        raw=tmp_path / "raw",
        interim=tmp_path / "interim",
        processed=tmp_path / "processed",
        sample=tmp_path / "sample",
        figures=tmp_path / "figures",
        app=tmp_path / "app",
        webmap=tmp_path / "webmap",
    )
    return dataclasses.replace(repo_config, paths=paths)


@pytest.fixture()
def sample_run_config(sample_config: Config, tmp_path: Path) -> Config:
    """Sample config reading committed raw data but writing into tmp_path."""
    paths = PathsConfig(
        root=sample_config.paths.root,
        raw=sample_config.paths.raw,
        interim=tmp_path / "interim",
        processed=tmp_path / "processed",
        sample=sample_config.paths.sample,
        figures=tmp_path / "figures",
        app=tmp_path / "app",
        webmap=tmp_path / "webmap",
    )
    return dataclasses.replace(sample_config, paths=paths)
