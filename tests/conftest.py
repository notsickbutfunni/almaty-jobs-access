"""Shared fixtures for the test suite.

All tests run offline on the committed configuration and the small
synthetic sample dataset; nothing here touches the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from almaty_access.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config_path() -> Path:
    """Path to the repository's real config file."""
    return REPO_ROOT / "config" / "config.yaml"


@pytest.fixture(scope="session")
def repo_config(config_path: Path) -> Config:
    """The repository's real configuration, loaded once per session."""
    return load_config(config_path)
