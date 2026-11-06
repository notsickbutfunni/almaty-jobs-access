"""Phase 7 dashboard data layer, on the committed sample (offline).

Covers: data-directory resolution, the loader, every map layer (values
traced back to their source columns, color rules, NaN handling), the
click-to-reach math (toy matrix + cross-check against the stored access
value), chart specs, download round-trips, and the export stages (the
committed bundle and the static web map), including caching, orphan
cleanup and manifest determinism.
"""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from almaty_access.config import Config
from almaty_access.dashboard import charts, downloads, layers, palette
from almaty_access.dashboard.bundle import MANIFEST_FILE, build_bundle, bundle_file_names
from almaty_access.dashboard.data import (
    ENV_VAR,
    REQUIRED_FILES,
    Tables,
    candidate_dirs,
    load_tables,
    missing_files,
    resolve_data_dir,
)
from almaty_access.dashboard.layers import LayerState, build_catalog, build_layer
from almaty_access.dashboard.reach import reach_summary, reachable
from almaty_access.dashboard.webmap import WEBMAP_FILE, build_webmap


@pytest.fixture(scope="module")
def tables(dashboard_sample: Config) -> Tables:
    return load_tables(dashboard_sample.paths.processed)


# ---------------------------------------------------------------- data


def test_sample_run_is_complete_and_loads(dashboard_sample: Config, tables: Tables) -> None:
    assert missing_files(dashboard_sample.paths.processed) == []
    assert tables.scenario_names == ("demo_abay",)
    assert set(tables.tt) == {"walk", "bike", "transit", "car"}
    assert set(tables.equity) == {"raster", "calibrated"}
    assert set(tables.scenario_modes("demo_abay")) == {"transit", "bike"}


def test_candidate_dirs_order(tmp_path: Path) -> None:
    ordered = candidate_dirs(tmp_path, override=tmp_path / "override")
    assert ordered[0] == tmp_path / "override"
    assert ordered[-3:] == [
        tmp_path / "data" / "app",
        tmp_path / "data" / "processed",
        tmp_path / "data" / "sample" / "processed",
    ]


def test_resolve_data_dir_env_var(
    dashboard_sample: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR, str(dashboard_sample.paths.processed))
    assert resolve_data_dir(tmp_path) == dashboard_sample.paths.processed


def test_resolve_data_dir_rejects_incomplete_explicit_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path))  # exists but has nothing
    with pytest.raises(FileNotFoundError, match="missing required files"):
        resolve_data_dir(tmp_path)


def test_resolve_data_dir_reports_all_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(FileNotFoundError, match="no complete dashboard data"):
        resolve_data_dir(tmp_path)


# ------------------------------------------------------------- palette


def test_sequential_rgba_monotone_and_nan_transparent() -> None:
    values = pd.Series([0.0, 5.0, np.nan, 10.0])
    fill, (low, high) = palette.sequential_rgba(values)
    assert fill.shape == (4, 4)
    assert fill[2, 3] == 0  # NaN row is transparent
    assert fill[0, 3] == palette.FILL_ALPHA
    # higher value -> darker (smaller RGB sum) on the blue ramp
    assert fill[3, :3].sum() < fill[1, :3].sum() < fill[0, :3].sum()
    assert low == pytest.approx(np.nanpercentile([0, 5, 10], 2))
    assert high == pytest.approx(np.nanpercentile([0, 5, 10], 98))


def test_sequential_rgba_reverse_flips_direction() -> None:
    values = pd.Series([0.0, 10.0])
    normal, _ = palette.sequential_rgba(values)
    reverse, _ = palette.sequential_rgba(values, reverse=True)
    assert (normal[0, :3] == reverse[1, :3]).all()
    assert (normal[1, :3] == reverse[0, :3]).all()


def test_sequential_rgba_constant_values_paint_mid_ramp() -> None:
    fill, _ = palette.sequential_rgba(pd.Series([3.0, 3.0, 3.0]))
    assert (fill[0] == fill[1]).all() and (fill[1] == fill[2]).all()
    assert fill[0, 3] == palette.FILL_ALPHA


def test_categorical_rgba_unknown_is_transparent() -> None:
    fill = palette.categorical_rgba(pd.Series(["HH", "nope", None]), {"HH": palette.RED})
    assert tuple(fill[0, :3]) == palette.hex_to_rgb(palette.RED)
    assert fill[1, 3] == 0 and fill[2, 3] == 0


def test_blend_matches_notebook_convention() -> None:
    # blend(color, 1) is the color itself; blend(color, 0) the surface.
    assert palette.blend(palette.RED, 1.0) == palette.RED
    assert palette.blend(palette.RED, 0.0) == palette.SURFACE


# -------------------------------------------------------------- layers


def test_catalog_reflects_sample_columns(tables: Tables) -> None:
    catalog = build_catalog(tables)
    assert catalog.thresholds == (15, 30, 45, 60)
    assert catalog.opportunities[0] == "jobs"
    assert "schools" in catalog.opportunities
    assert set(catalog.lisa_modes) == {"walk", "bike", "transit", "car"}
    assert catalog.gap_thresholds == (15, 30, 45, 60)
    assert catalog.scenarios == ("demo_abay",)


def test_access_layer_matches_source_column(tables: Tables) -> None:
    layer = build_layer(tables, LayerState(layer="access"))
    source = tables.access.set_index("h3")["accpct_jobs_transit_45"]
    for row in layer.table.itertuples():
        assert row.value == pytest.approx(source[row.h3])
    assert layer.table["in_city"].all()
    assert layer.ramp == palette.SEQ_BLUE and not layer.ramp_reversed
    assert all(len(fill) == 4 for fill in layer.table["fill"])
    assert "lower bound" in layer.caption


def test_access_layer_scenario_swaps_to_scen_values(tables: Tables) -> None:
    state = LayerState(layer="access", scenario="demo_abay", mode="transit")
    layer = build_layer(tables, state)
    source = tables.scenarios["demo_abay"].set_index("h3")["accpct_jobs_transit_45_scen"]
    for row in layer.table.itertuples():
        assert row.value == pytest.approx(source[row.h3])
    assert "with" in layer.title


def test_access_layer_unchanged_mode_falls_back_to_baseline(tables: Tables) -> None:
    state = LayerState(layer="access", scenario="demo_abay", mode="car")
    layer = build_layer(tables, state)
    source = tables.access.set_index("h3")["accpct_jobs_car_45"]
    for row in layer.table.itertuples():
        assert row.value == pytest.approx(source[row.h3])
    assert "does not change this mode" in layer.caption


def test_scenario_delta_layer_is_nonnegative(tables: Tables) -> None:
    state = LayerState(layer="scenario_delta", scenario="demo_abay", mode="transit")
    layer = build_layer(tables, state)
    assert (layer.table["value"] >= 0).all()
    source = tables.scenarios["demo_abay"].set_index("h3")["accpct_jobs_transit_45_delta"]
    for row in layer.table.itertuples():
        assert row.value == pytest.approx(source[row.h3])


def test_gap_layer_is_reversed_ramp(tables: Tables) -> None:
    layer = build_layer(tables, LayerState(layer="gap", threshold=45))
    assert layer.ramp_reversed
    assert "overstates car dependence" in layer.caption
    source = tables.equity["raster"].set_index("h3")["modegap_transit_car_45"]
    for row in layer.table.itertuples():
        assert row.value == pytest.approx(source[row.h3])


def test_lisa_layer_handles_all_na_sample(tables: Tables) -> None:
    """The sample grid is too small for LISA verdicts: everything is <NA>,
    and the layer must come back empty rather than crash."""
    layer = build_layer(tables, LayerState(layer="lisa", mode="transit"))
    assert layer.table.empty
    assert layer.legend == ()


def test_lisa_layer_categorical_legend(tables: Tables) -> None:
    """Inject one verdict per hex to exercise the label/color mapping."""
    equity = {weighting: frame.copy() for weighting, frame in tables.equity.items()}
    column = layers.lisa_column(equity["raster"], "transit")
    equity["raster"][column] = ["HH", "LL", "ns"]
    patched = dataclasses.replace(tables, equity=equity)
    layer = build_layer(patched, LayerState(layer="lisa", mode="transit"))
    assert list(layer.table["value"]) == [
        palette.LISA_LABELS["HH"],
        palette.LISA_LABELS["LL"],
        palette.LISA_LABELS["ns"],
    ]
    legend = dict(layer.legend)
    assert legend[palette.LISA_LABELS["HH"]] == palette.RED
    assert legend[palette.LISA_LABELS["LL"]] == palette.BLUE
    assert tuple(layer.table["fill"].iloc[0][:3]) == palette.hex_to_rgb(palette.RED)


def test_deserts_layer_covers_inhabited_only(tables: Tables) -> None:
    layer = build_layer(tables, LayerState(layer="deserts", mode="transit"))
    equity = tables.equity["raster"]
    inhabited_city = int((equity["inhabited"] & equity["in_city"]).sum())
    assert len(layer.table) == inhabited_city


def test_population_layer_follows_weighting(tables: Tables) -> None:
    raster = build_layer(tables, LayerState(layer="population", weighting="raster"))
    calibrated = build_layer(tables, LayerState(layer="population", weighting="calibrated"))
    for weighting, layer in (("raster", raster), ("calibrated", calibrated)):
        source = tables.equity[weighting].set_index("h3")["population"]
        for row in layer.table.itertuples():
            assert row.value == pytest.approx(source[row.h3])


def test_jobs_uncertainty_is_p90_minus_p10(tables: Tables) -> None:
    layer = build_layer(tables, LayerState(layer="jobs", uncertainty=True))
    jobs = tables.jobs.set_index("h3")
    for row in layer.table.itertuples():
        expected = jobs.loc[row.h3, "jobs_rf_p90"] - jobs.loc[row.h3, "jobs_rf_p10"]
        assert row.value == pytest.approx(expected)


@pytest.mark.parametrize(
    "state, message",
    [
        (LayerState(layer="nope"), "unknown layer"),
        (LayerState(layer="access", opportunity="schools", mode="transit"), "walking only"),
        (LayerState(layer="access", threshold="grav"), "no percent variant"),
        (
            LayerState(layer="access", opportunity="schools", mode="walk", metric="accpct"),
            "plain counts only",
        ),
        (LayerState(layer="scenario_delta"), "needs a scenario"),
        (
            LayerState(layer="scenario_delta", scenario="demo_abay", mode="car"),
            "changes only",
        ),
        (LayerState(layer="access", weighting="official"), "unknown weighting"),
    ],
)
def test_invalid_states_raise(tables: Tables, state: LayerState, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_layer(tables, state)


# --------------------------------------------------------------- reach


def test_reachable_toy_matrix() -> None:
    tt = pd.DataFrame(
        {
            "origin_h3": ["A", "A", "A", "B"],
            "dest_h3": ["A", "B", "C", "A"],
            "minutes": [3.0, 20.0, 40.0, 5.0],
        }
    )
    reached = reachable(tt, "A", 30)
    assert list(reached["h3"]) == ["A", "B"]  # sorted by minutes, C too far
    assert reachable(tt, "B", 30)["h3"].tolist() == ["A"]
    assert reachable(tt, "C", 60).empty  # never an origin


def test_reach_summary_cross_checks_stored_access(tables: Tables) -> None:
    origin = tables.access.loc[tables.access["in_city"], "h3"].iloc[0]
    summary = reach_summary(tables, origin, "walk", 45)
    assert summary.n_reachable >= 1  # at least itself
    assert summary.jobs_stored is not None
    # jobs_rf is stored rounded; allow the accumulated rounding slack.
    slack = max(1.0, 0.002 * summary.jobs_stored)
    assert summary.jobs_sum == pytest.approx(summary.jobs_stored, abs=slack)
    assert list(summary.facilities_walk.columns) == ["facility", "count"]
    assert (summary.facilities_walk["count"] >= 0).all()
    assert list(summary.nearest_walk.columns) == ["facility", "minutes"]


def test_reach_summary_unknown_hex_raises(tables: Tables) -> None:
    with pytest.raises(ValueError, match="not in the analysis grid"):
        reach_summary(tables, "8f00000000000000", "walk", 45)


# -------------------------------------------------------------- charts


def test_chart_specs_build_and_carry_data(tables: Tables) -> None:
    measures = list(dict.fromkeys(tables.districts["measure"]))
    assert measures
    for measure in measures:
        for frame in (tables.districts, tables.lorenz):
            assert charts.resolution_for(frame, measure, "raster") in (8, 9)
        bars = charts.district_bars(tables.districts, measure, "raster").to_dict()
        assert bars["layer"], "bars + labels expected"
        lorenz = charts.lorenz_curve(tables.lorenz, tables.summary, measure, "calibrated")
        assert "Gini" in lorenz.to_dict()["title"]
    gains = charts.gains_bars(tables.gains, "demo_abay").to_dict()
    scale = gains["layer"][0]["encoding"]["color"]["scale"]
    assert scale["range"] == [palette.LIGHT_BLUE, palette.BLUE]


def test_job_measure_resolution_prefers_r8(tables: Tables) -> None:
    assert charts.resolution_for(tables.districts, "accpct_jobs_transit_45", "raster") == 8


def test_measure_label_formats() -> None:
    assert charts.measure_label("accpct_jobs_transit_45") == (
        "Share of city jobs reachable · transit · 45 min (%)"
    )
    assert charts.measure_label("acc_jobs_bike_grav") == (
        "Jobs reachable · bike · gravity-weighted"
    )
    assert "2SFCA" in charts.measure_label("sfca_schools_walk")
    assert charts.measure_label("something_else") == "something_else"


# ----------------------------------------------------------- downloads


def test_download_roundtrips(tables: Tables) -> None:
    frame = tables.access[["h3", "district", "acc_jobs_walk_15"]].head(3)
    raw_csv = downloads.csv_bytes(frame)
    assert raw_csv.startswith("﻿".encode())  # Excel-friendly BOM
    back = pd.read_csv(io.BytesIO(raw_csv))
    assert list(back.columns) == list(frame.columns)
    assert len(back) == len(frame)
    back_parquet = pd.read_parquet(io.BytesIO(downloads.parquet_bytes(frame)))
    pd.testing.assert_frame_equal(back_parquet.reset_index(drop=True), frame.reset_index(drop=True))


# -------------------------------------------------------------- export


@pytest.fixture()
def export_config(dashboard_sample: Config, tmp_path: Path) -> Config:
    """The sample run, with export outputs pointed into this test's tmp."""
    paths = dataclasses.replace(
        dashboard_sample.paths, app=tmp_path / "app", webmap=tmp_path / "webmap"
    )
    return dataclasses.replace(dashboard_sample, paths=paths)


def test_bundle_contains_loader_requirements(export_config: Config) -> None:
    build_bundle(export_config)
    app_dir = export_config.paths.app
    assert missing_files(app_dir) == []
    loaded = load_tables(app_dir)  # the bundle serves the dashboard as-is
    assert loaded.scenario_names == ("demo_abay",)
    # byte-identical copies, not re-encodings
    name = REQUIRED_FILES[0]
    assert (app_dir / name).read_bytes() == (export_config.paths.processed / name).read_bytes()
    manifest = json.loads((app_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert set(manifest["files"]) == set(bundle_file_names(export_config.paths.processed))
    for file_name, size in manifest["files"].items():
        assert (app_dir / file_name).stat().st_size == size


def test_bundle_is_cached_and_deterministic(export_config: Config) -> None:
    build_bundle(export_config)
    manifest_path = export_config.paths.app / MANIFEST_FILE
    first = manifest_path.read_bytes()
    stamp = manifest_path.stat().st_mtime_ns
    build_bundle(export_config)  # fresh -> skipped
    assert manifest_path.stat().st_mtime_ns == stamp
    build_bundle(export_config, force=True)  # rebuilt -> identical bytes
    assert manifest_path.read_bytes() == first


def test_bundle_removes_orphan_scenarios(export_config: Config) -> None:
    build_bundle(export_config)
    orphan = export_config.paths.app / "scenario_gone_r8.parquet"
    orphan.write_bytes(b"stale")
    build_bundle(export_config)
    assert not orphan.exists()


def test_webmap_is_selfcontained_and_cached(export_config: Config) -> None:
    build_webmap(export_config)
    page_path = export_config.paths.webmap / WEBMAP_FILE
    page = page_path.read_text(encoding="utf-8")
    assert "H3HexagonLayer" in page
    assert "cartocdn" in page  # key-less basemap style
    assert "mapboxApiAccessToken" not in page or '"mapboxApiAccessToken": ""' in page
    assert "lower bound" in page
    some_hex = pd.read_parquet(export_config.paths.processed / "access_r8.parquet")["h3"].iloc[0]
    assert some_hex in page
    stamp = page_path.stat().st_mtime_ns
    build_webmap(export_config)  # fresh -> skipped
    assert page_path.stat().st_mtime_ns == stamp
