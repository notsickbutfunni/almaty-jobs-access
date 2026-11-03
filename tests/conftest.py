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
    )
    return dataclasses.replace(sample_config, paths=paths)
