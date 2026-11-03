"""Tests for the synthetic sample dataset."""

from __future__ import annotations

from pathlib import Path

from geopandas.testing import assert_geodataframe_equal

from almaty_access.config import Config
from almaty_access.io import read_geoparquet
from almaty_access.sample import POI_CATEGORIES, build_sample


def test_build_sample_is_deterministic(repo_config: Config, tmp_path: Path) -> None:
    first = build_sample(repo_config, out_dir=tmp_path / "a")
    second = build_sample(repo_config, out_dir=tmp_path / "b")
    for layer in ("boundary", "poi_points"):
        assert_geodataframe_equal(read_geoparquet(first[layer]), read_geoparquet(second[layer]))


def test_sample_points_match_config(repo_config: Config, tmp_path: Path) -> None:
    written = build_sample(repo_config, out_dir=tmp_path)
    pois = read_geoparquet(written["poi_points"])
    boundary = read_geoparquet(written["boundary"])

    assert len(pois) == repo_config.sample.n_points
    assert pois.crs is not None and pois.crs.to_epsg() == 4326
    assert boundary.crs is not None and boundary.crs.to_epsg() == 4326
    assert set(pois["category"]).issubset(set(POI_CATEGORIES))
    assert (pois["weight"] > 0).all()

    min_lon, min_lat, max_lon, max_lat = repo_config.sample.bbox
    bounds = pois.total_bounds  # (minx, miny, maxx, maxy)
    assert bounds[0] >= min_lon and bounds[1] >= min_lat
    assert bounds[2] <= max_lon and bounds[3] <= max_lat


def test_committed_sample_matches_config(repo_config: Config, tmp_path: Path) -> None:
    """The files committed in data/sample/ must equal a fresh regeneration.

    If this fails after changing the sample config or the generator,
    refresh the committed files with: uv run almaty-access sample
    """
    regenerated = build_sample(repo_config, out_dir=tmp_path)
    for fresh_path in regenerated.values():
        committed_path = repo_config.paths.sample / fresh_path.name
        assert committed_path.exists(), (
            f"missing committed sample file {committed_path}; run: uv run almaty-access sample"
        )
        assert_geodataframe_equal(read_geoparquet(committed_path), read_geoparquet(fresh_path))
