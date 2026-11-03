"""Tests for GeoParquet IO and the mtime-based caching logic."""

from __future__ import annotations

import os
import time
from pathlib import Path

import geopandas as gpd
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Point

from almaty_access.io import ensure_built, needs_rebuild, read_geoparquet, write_geoparquet


def _points_gdf(crs: str | None = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": ["a", "b"], "value": [1.5, 2.5]},
        geometry=[Point(76.9, 43.25), Point(76.95, 43.24)],
        crs=crs,
    )


def _set_age(path: Path, seconds_ago: float) -> None:
    """Set a file's modification time to ``seconds_ago`` before now."""
    timestamp_ns = time.time_ns() - int(seconds_ago * 1_000_000_000)
    os.utime(path, ns=(timestamp_ns, timestamp_ns))


def test_geoparquet_roundtrip(tmp_path: Path) -> None:
    gdf = _points_gdf()
    path = write_geoparquet(gdf, tmp_path / "nested" / "points.parquet")
    assert path.exists()
    loaded = read_geoparquet(path)
    assert loaded.crs is not None
    assert loaded.crs.to_epsg() == 4326
    assert_geodataframe_equal(loaded, gdf)


def test_write_without_crs_refused(tmp_path: Path) -> None:
    gdf = _points_gdf(crs=None)
    with pytest.raises(ValueError, match="no CRS"):
        write_geoparquet(gdf, tmp_path / "points.parquet")


def test_read_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_geoparquet(tmp_path / "absent.parquet")


def test_needs_rebuild_when_output_missing(tmp_path: Path) -> None:
    source = tmp_path / "in.txt"
    source.write_text("x")
    assert needs_rebuild(tmp_path / "out.txt", [source]) is True


def test_needs_rebuild_false_when_output_fresh(tmp_path: Path) -> None:
    source = tmp_path / "in.txt"
    source.write_text("x")
    output = tmp_path / "out.txt"
    output.write_text("y")
    _set_age(source, seconds_ago=60)
    assert needs_rebuild(output, [source]) is False


def test_needs_rebuild_true_when_input_newer(tmp_path: Path) -> None:
    source = tmp_path / "in.txt"
    source.write_text("x")
    output = tmp_path / "out.txt"
    output.write_text("y")
    _set_age(output, seconds_ago=60)
    assert needs_rebuild(output, [source]) is True


def test_needs_rebuild_missing_input_raises(tmp_path: Path) -> None:
    output = tmp_path / "out.txt"
    output.write_text("y")
    with pytest.raises(FileNotFoundError, match="missing input"):
        needs_rebuild(output, [tmp_path / "absent.txt"])


def test_ensure_built_runs_once_then_skips(tmp_path: Path) -> None:
    source = tmp_path / "in.txt"
    source.write_text("x")
    _set_age(source, seconds_ago=60)
    output = tmp_path / "out.txt"
    calls: list[int] = []

    def build() -> None:
        calls.append(1)
        output.write_text("built")

    assert ensure_built(output, [source], build) is True
    assert ensure_built(output, [source], build) is False
    assert len(calls) == 1
    assert ensure_built(output, [source], build, force=True) is True
    assert len(calls) == 2


def test_ensure_built_multiple_outputs(tmp_path: Path) -> None:
    """A step with several outputs rebuilds when ANY of them is missing."""
    source = tmp_path / "in.txt"
    source.write_text("x")
    _set_age(source, seconds_ago=60)
    out_a = tmp_path / "a.txt"
    out_b = tmp_path / "b.txt"

    def build() -> None:
        out_a.write_text("a")
        out_b.write_text("b")

    assert ensure_built([out_a, out_b], [source], build) is True
    assert ensure_built([out_a, out_b], [source], build) is False
    out_b.unlink()
    assert ensure_built([out_a, out_b], [source], build) is True


def test_ensure_built_force_still_requires_inputs(tmp_path: Path) -> None:
    """force=True must not bypass the missing-input check."""
    with pytest.raises(FileNotFoundError, match="missing input"):
        ensure_built(tmp_path / "out.txt", [tmp_path / "absent.txt"], lambda: None, force=True)


def test_ensure_built_raises_when_output_not_produced(tmp_path: Path) -> None:
    source = tmp_path / "in.txt"
    source.write_text("x")

    def lazy_build() -> None:
        pass

    with pytest.raises(RuntimeError, match="did not produce"):
        ensure_built(tmp_path / "out.txt", [source], lazy_build)
