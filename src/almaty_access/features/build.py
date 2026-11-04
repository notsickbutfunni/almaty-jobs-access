"""Assemble the hex feature tables (``almaty-access features``).

Order of operations (README §7 Phase 2):

1. one grid per resolution (buffered boundary, district-assigned),
2. feature blocks, each a pure function over the interim layers,
3. k-ring context columns for the configured key features,
4. a near-duplicate check: of every pair with |r| > threshold on the
   log-transformed **analysis-resolution** table, the later column is
   dropped — from BOTH tables, so r8 and r9 keep identical schemas,
5. a hard NaN/inf gate: the model phases must never meet a missing
   value whose meaning is undocumented.

Outputs: ``grid_r{8,9}.parquet`` (GeoParquet), ``features_r{8,9}.parquet``
(plain parquet — join the grid on ``h3`` for maps), ``subcenters.parquet``
and ``features_report.json`` (what was transformed, dropped and why).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from almaty_access.config import Config
from almaty_access.download.osm import LANDUSE_FILE
from almaty_access.features.built_form import built_form_features
from almaty_access.features.centrality import centrality_features, find_subcenters
from almaty_access.features.context import kring_features
from almaty_access.features.grid import GRID_FILE_TEMPLATE, build_grid, study_domain
from almaty_access.features.landuse import landuse_features
from almaty_access.features.network import network_features, node_betweenness
from almaty_access.features.pois import poi_features
from almaty_access.features.population import population_features
from almaty_access.features.terrain import terrain_features
from almaty_access.features.transit import transit_features
from almaty_access.io import ensure_built, read_geoparquet, write_geoparquet
from almaty_access.prepare.boundary import BUFFER_FILE, CITY_FILE, DISTRICTS_FILE
from almaty_access.prepare.buildings import OUTPUT_FILE as BUILDINGS_FILE
from almaty_access.prepare.dem import DEM_METRIC_FILE, SLOPE_FILE
from almaty_access.prepare.network import interim_network_files
from almaty_access.prepare.pois import OUTPUT_FILE as POIS_FILE
from almaty_access.prepare.pois import load_category_rules

logger = logging.getLogger(__name__)

FEATURES_FILE_TEMPLATE = "features_r{res}.parquet"
SUBCENTERS_FILE = "subcenters.parquet"
REPORT_FILE = "features_report.json"
POPULATION_FILE_TEMPLATE = "population_r{res}.parquet"  # written by prepare_population

#: Pearson r over fewer observations than this is noise, not structure
#: (through 3 points almost any two monotone features correlate ~1), so
#: near-duplicate pruning is skipped on tiny grids like the sample's.
MIN_CELLS_FOR_PRUNING = 30

#: Features the README §7 Phase 2 table names directly and the EDA /
#: Phase 3 model lean on: pruning may never drop them, however high the
#: correlation (the first city run dropped total GFA in favour of the
#: building COUNT, and mean elevation against road distance — a
#: geographic coincidence, not a duplicate). Pairs of protected columns
#: above the threshold are recorded in the report instead.
PROTECTED_FEATURES: frozenset[str] = frozenset(
    {
        "bf_gfa_m2",
        "bf_footprint_m2",
        "bf_bldg_cnt",
        "poi_cnt_total",
        "poi_entropy",
        "pop_residents",
        "net_intersection_cnt",
        "net_dist_major_m",
        "cent_dist_center_m",
        "ter_elev_mean_m",
        "ter_slope_mean_deg",
    }
)


def _assert_finite(table: pd.DataFrame, res: int) -> None:
    """Fail loudly when any feature is NaN or infinite."""
    values = table.replace([np.inf, -np.inf], np.nan)
    bad = values.isna().sum()
    bad = bad[bad > 0]
    if not bad.empty:
        raise RuntimeError(
            f"features r{res}: NaN/inf values in {bad.to_dict()} — every feature "
            "must be finite with a documented meaning (README §7 Phase 2)"
        )


def log1p_columns(table: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    """The columns the config marks for log1p at model time."""
    return [column for column in table.columns if column.startswith(prefixes)]


def _transformed(table: pd.DataFrame, prefixes: tuple[str, ...]) -> pd.DataFrame:
    result = table.copy()
    for column in log1p_columns(table, prefixes):
        if float(result[column].min()) < 0:
            logger.warning("column %s has negative values; log1p skipped", column)
            continue
        result[column] = np.log1p(result[column])
    return result


def near_duplicate_drops(
    table: pd.DataFrame,
    prefixes: tuple[str, ...],
    threshold: float,
    protected: frozenset[str] = PROTECTED_FEATURES,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Greedy near-duplicate pruning on the transformed table.

    Of every pair with |r| > threshold, the non-protected member is
    dropped; between two unprotected columns the earlier one (blocks
    are assembled core-first) survives. A pair of PROTECTED columns is
    never dropped — it is returned as a kept-pair so the report still
    documents the collinearity. Correlations use the log1p view because
    raw heavy-tailed counts make Pearson r meaningless.

    Returns ``(drops, high_corr_kept)``.
    """
    corr = _transformed(table, prefixes).corr().abs()
    columns = list(corr.columns)
    dropped: dict[str, dict[str, object]] = {}
    kept_pairs: list[dict[str, object]] = []
    for i, left in enumerate(columns):
        if left in dropped:
            continue
        for right in columns[i + 1 :]:
            if right in dropped:
                continue
            r = corr.loc[left, right]
            if not (pd.notna(r) and float(r) > threshold):
                continue
            if right not in protected:
                dropped[right] = {"dropped": right, "kept": left, "r": round(float(r), 4)}
            elif left not in protected:
                dropped[left] = {"dropped": left, "kept": right, "r": round(float(r), 4)}
                break  # left is gone; stop comparing it
            else:
                kept_pairs.append({"a": left, "b": right, "r": round(float(r), 4)})
    return list(dropped.values()), kept_pairs


def _feature_table(
    config: Config,
    res: int,
    grid: gpd.GeoDataFrame,
    layers: dict[str, gpd.GeoDataFrame],
    subcenters: gpd.GeoDataFrame,
    betweenness: pd.Series,
    fallback_distance_m: float,
    categories: list[str],
) -> pd.DataFrame:
    metric = config.crs.metric
    interim = config.paths.interim
    built = built_form_features(
        layers["buildings"], grid, res, metric, config.features.large_gfa_m2
    )
    population = read_geoparquet(interim / POPULATION_FILE_TEMPLATE.format(res=res))
    blocks = [
        built,
        poi_features(layers["pois"], grid, res, categories),
        landuse_features(layers["landuse"], grid, metric),
        network_features(
            layers["drive_nodes"],
            layers["drive_edges"],
            layers["walk_edges"],
            grid,
            res,
            metric,
            betweenness,
            fallback_distance_m,
        ),
        transit_features(layers["pois"], grid, res, metric, fallback_distance_m),
        population_features(population, built, grid, res),
        centrality_features(grid, subcenters, config),
        terrain_features(
            interim / DEM_METRIC_FILE,
            interim / SLOPE_FILE,
            grid,
            res,
            config.crs.storage,
            metric,
        ),
    ]
    core = pd.concat(blocks, axis=1)
    context = kring_features(core, config.features.context_columns, config.features.context_k)
    table = pd.concat([core, context], axis=1)
    _assert_finite(table, res)
    return table


def build_features(config: Config, *, force: bool = False) -> list[Path]:
    """Build grids + feature tables with make-style caching."""
    interim = config.paths.interim
    processed = config.paths.processed
    resolutions = (config.h3.analysis_resolution, config.h3.fine_resolution)

    outputs = [
        *(processed / GRID_FILE_TEMPLATE.format(res=res) for res in resolutions),
        *(processed / FEATURES_FILE_TEMPLATE.format(res=res) for res in resolutions),
        processed / SUBCENTERS_FILE,
        processed / REPORT_FILE,
    ]
    inputs = [
        interim / BUFFER_FILE,
        interim / CITY_FILE,
        interim / DISTRICTS_FILE,
        interim / BUILDINGS_FILE,
        interim / POIS_FILE,
        *(interim / POPULATION_FILE_TEMPLATE.format(res=res) for res in resolutions),
        interim_network_files(config, "drive")["nodes"],
        interim_network_files(config, "drive")["edges"],
        interim_network_files(config, "walk")["edges"],
        interim / DEM_METRIC_FILE,
        interim / SLOPE_FILE,
        config.paths.raw / LANDUSE_FILE,
        config.pois.categories_file,
        config.source_path,
    ]

    def build() -> None:
        layers = {
            "buildings": read_geoparquet(interim / BUILDINGS_FILE),
            "pois": read_geoparquet(interim / POIS_FILE),
            "landuse": read_geoparquet(config.paths.raw / LANDUSE_FILE),
            "drive_nodes": read_geoparquet(interim_network_files(config, "drive")["nodes"]),
            "drive_edges": read_geoparquet(interim_network_files(config, "drive")["edges"]),
            "walk_edges": read_geoparquet(interim_network_files(config, "walk")["edges"]),
        }
        districts = read_geoparquet(interim / DISTRICTS_FILE)
        buffer_polygon = read_geoparquet(interim / BUFFER_FILE).geometry.iloc[0]
        domain = study_domain(config, buffer_polygon)
        bounds = (
            gpd.GeoSeries([domain], crs=config.crs.storage).to_crs(config.crs.metric).total_bounds
        )
        # An upper bound on any within-domain distance: the bbox diagonal.
        # Used when a distance target (metro, major road) has no instance
        # in the study area, so the column stays finite and comparable.
        fallback_distance_m = float(np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]))

        categories = list(
            dict.fromkeys(
                rule.category for rule in load_category_rules(config.pois.categories_file)
            )
        )
        subcenters = find_subcenters(layers["pois"], config)
        # Buffer-zone clusters (satellite towns) stay eligible — they are
        # real job destinations for edge hexes — but are labelled so maps
        # and DECISIONS can tell them apart from in-city sub-centres.
        city_polygon = read_geoparquet(interim / CITY_FILE).geometry.iloc[0]
        subcenters["in_city"] = (
            subcenters.geometry.within(city_polygon) if len(subcenters) else pd.Series(dtype=bool)
        )
        betweenness = node_betweenness(
            layers["drive_edges"], config.features.betweenness_k, config.random_seed
        )

        grids: dict[int, gpd.GeoDataFrame] = {}
        tables: dict[int, pd.DataFrame] = {}
        for res in resolutions:
            grids[res] = build_grid(config, res, buffer_polygon, districts)
            tables[res] = _feature_table(
                config,
                res,
                grids[res],
                layers,
                subcenters,
                betweenness,
                fallback_distance_m,
                categories,
            )

        analysis = tables[config.h3.analysis_resolution]
        pruning_skipped = ""
        kept_pairs: list[dict[str, object]] = []
        if len(analysis) < MIN_CELLS_FOR_PRUNING:
            pruning_skipped = (
                f"only {len(analysis)} cells at the analysis resolution "
                f"(< {MIN_CELLS_FOR_PRUNING}); correlations would be noise"
            )
            logger.warning("near-duplicate pruning skipped: %s", pruning_skipped)
            drops = []
        else:
            drops, kept_pairs = near_duplicate_drops(
                analysis, config.features.log1p_prefixes, config.features.corr_threshold
            )
        drop_names = [str(item["dropped"]) for item in drops]
        if drop_names:
            logger.info("dropping %d near-duplicate features: %s", len(drop_names), drop_names)

        processed.mkdir(parents=True, exist_ok=True)
        for res in resolutions:
            write_geoparquet(grids[res], processed / GRID_FILE_TEMPLATE.format(res=res))
            table = tables[res].drop(columns=drop_names)
            ids = grids[res][["h3", "res", "district", "in_city"]].set_index("h3")
            final = pd.concat([ids, table], axis=1).reset_index()
            final.to_parquet(processed / FEATURES_FILE_TEMPLATE.format(res=res), index=False)
            logger.info("features r%d: %d cells x %d features", res, len(final), table.shape[1])
        write_geoparquet(subcenters, processed / SUBCENTERS_FILE)

        report = {
            "resolutions": list(resolutions),
            "n_cells": {f"r{res}": len(grids[res]) for res in resolutions},
            "n_features_before_pruning": int(analysis.shape[1]),
            "n_features_after_pruning": int(analysis.shape[1] - len(drop_names)),
            "near_duplicates_dropped": drops,
            "high_corr_kept": kept_pairs,
            "protected_features": sorted(PROTECTED_FEATURES),
            "pruning_skipped_reason": pruning_skipped,
            "corr_threshold": config.features.corr_threshold,
            "log1p_prefixes": list(config.features.log1p_prefixes),
            "log1p_columns": log1p_columns(
                analysis.drop(columns=drop_names),
                config.features.log1p_prefixes,
            ),
            "n_subcenters": len(subcenters),
            "betweenness_k": config.features.betweenness_k,
            "random_seed": config.random_seed,
        }
        (processed / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")

    ensure_built(outputs, inputs, build, force=force)
    return outputs
