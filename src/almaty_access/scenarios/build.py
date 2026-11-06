"""Phase 6 orchestrator: what each scenario changes, and for whom.

One cached step, runnable through ``almaty-access scenarios``. For every
scenario file it merges the drawn lines into the baseline transit tables
(or boosts the drawn bike corridors), rebuilds ONLY the affected mode's
graph, recomputes the analysis-resolution travel-time matrix in memory
and re-derives the job-access measures with the same Phase 4 code the
baseline used — so with nothing drawn, the pipeline reproduces the
stored baseline exactly (a tested guarantee). Outputs per hex (baseline,
scenario, delta), the README's three answers — the Δ-jobs map data,
"population gaining more than X jobs" and the change in Gini — under
both Phase 5 population weightings, and a report with one sentence per
scenario.

A scenario only ADDS arcs to a graph or speeds edges up, and the
rounding applied to measures is monotone, so every delta is >= 0; a
negative delta is a bug by construction and stops the build.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from almaty_access.access.build import (
    ACCESS_FILE_TEMPLATE,
    grid_points,
    street_graph,
)
from almaty_access.access.graph import ModeGraph, build_graph
from almaty_access.access.matrix import hex_times
from almaty_access.access.measures import cumulative, gravity
from almaty_access.access.speed import tobler_walk_kmh
from almaty_access.access.transit import build_transit_graph
from almaty_access.config import Config
from almaty_access.download.stats import FILE_NAME as STATS_FILE
from almaty_access.equity.inequality import distribution_summary
from almaty_access.equity.weights import (
    apply_factors,
    calibrated_weights,
    district_resident_shares,
)
from almaty_access.features.build import FEATURES_FILE_TEMPLATE
from almaty_access.features.grid import GRID_FILE_TEMPLATE
from almaty_access.io import ensure_built, read_geoparquet
from almaty_access.model.build import JOBS_FILE_TEMPLATE
from almaty_access.prepare.network import interim_network_files
from almaty_access.prepare.transit import LEGS_FILE, ROUTES_FILE, STOPS_FILE
from almaty_access.scenarios.bike import boosted_edge_seconds
from almaty_access.scenarios.definition import Scenario, load_scenarios
from almaty_access.scenarios.transit import apply_transit_scenario
from almaty_access.schemas import EQUITY_WEIGHTINGS

logger = logging.getLogger(__name__)

SCENARIO_FILE_TEMPLATE = "scenario_{name}_r{res}.parquet"
SUMMARY_FILE = "scenarios_summary.parquet"
GAINS_FILE = "scenarios_gains.parquet"
REPORT_FILE = "scenarios_report.json"

RASTER, CALIBRATED = EQUITY_WEIGHTINGS

_HEADLINE_RE = re.compile(r"^accpct_jobs_([a-z]+)_(\d+)$")


def _num(value: float, digits: int) -> float | None:
    """Rounded value for the JSON report; None where no number exists
    (a Gini over an all-zero distribution) — NaN is not strict JSON."""
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def affected_columns(mode: str, thresholds: tuple[int, ...]) -> list[str]:
    """Access-table job columns a scenario of this mode recomputes."""
    columns: list[str] = []
    for threshold in thresholds:
        columns.append(f"acc_jobs_{mode}_{threshold}")
        columns.append(f"accpct_jobs_{mode}_{threshold}")
    columns.append(f"acc_jobs_{mode}_grav")
    return columns


def headline_measure(config: Config, mode: str) -> tuple[str, int]:
    """The mode's headline measure and its threshold (minutes).

    Reuses the Phase 5 configuration: the first analysis-resolution
    headline entry of the form ``accpct_jobs_{mode}_{T}`` — the gains and
    the Gini change are quoted for the same measure the equity phase
    quotes, so the two reports talk about the same thing.
    """
    for measure in config.equity.headline_analysis:
        match = _HEADLINE_RE.match(measure)
        if match and match.group(1) == mode:
            return measure, int(match.group(2))
    raise ValueError(
        f"config error: 'equity.headline_measures.analysis' has no accpct_jobs_{mode}_<T> "
        f"entry — needed to report scenario gains for mode '{mode}'"
    )


def job_measures(
    tt: pd.DataFrame,
    jobs: pd.Series,
    thresholds: tuple[int, ...],
    half_life: float,
    index: pd.Index,
    total_jobs: float,
    mode: str,
) -> dict[str, np.ndarray]:
    """Cumulative / share / gravity job access, rounded like Phase 4.

    This mirrors ``access.build.build_measures`` line by line (including
    the rounding and the share computed from UNROUNDED counts), so an
    unchanged network reproduces the stored baseline columns exactly.
    """
    out: dict[str, np.ndarray] = {}
    cum = cumulative(tt, jobs, thresholds)
    for threshold in thresholds:
        values = cum[threshold].reindex(index).fillna(0.0).to_numpy()
        out[f"acc_jobs_{mode}_{threshold}"] = np.round(values, 1)
        out[f"accpct_jobs_{mode}_{threshold}"] = np.round(values / total_jobs * 100.0, 2)
    grav = gravity(tt, jobs, half_life).reindex(index).fillna(0.0).to_numpy()
    out[f"acc_jobs_{mode}_grav"] = np.round(grav, 1)
    return out


def build_scenarios(config: Config, *, force: bool = False) -> None:
    """Per-scenario hex tables, the summary/gains tables and the report."""
    processed = config.paths.processed
    res = config.h3.analysis_resolution
    scenarios = load_scenarios(config)

    outputs = [
        *(processed / SCENARIO_FILE_TEMPLATE.format(name=s.name, res=res) for s in scenarios),
        processed / SUMMARY_FILE,
        processed / GAINS_FILE,
        processed / REPORT_FILE,
    ]
    inputs = [
        *(scenario.path for scenario in scenarios),
        config.paths.interim / STOPS_FILE,
        config.paths.interim / ROUTES_FILE,
        config.paths.interim / LEGS_FILE,
        *interim_network_files(config, "walk").values(),
        *interim_network_files(config, "bike").values(),
        processed / GRID_FILE_TEMPLATE.format(res=res),
        processed / FEATURES_FILE_TEMPLATE.format(res=res),
        processed / JOBS_FILE_TEMPLATE.format(res=res),
        processed / ACCESS_FILE_TEMPLATE.format(res=res),
        config.source_path,
    ]
    stats_path = config.paths.raw / STATS_FILE
    if stats_path.exists():  # optional: without it the calibrated weighting is skipped
        inputs.append(stats_path)
    # A per-hex file of a scenario that was renamed or deleted must not
    # survive next to a report that no longer mentions it.
    orphans = [path for path in processed.glob("scenario_*_r*.parquet") if path not in set(outputs)]

    def build() -> None:
        for path in orphans:
            logger.info("removing stale %s", path.name)
            path.unlink()
        _build(config, scenarios)

    ensure_built(outputs, inputs, build, force=force or bool(orphans))


def _build(config: Config, scenarios: list[Scenario]) -> None:
    processed = config.paths.processed
    res = config.h3.analysis_resolution
    thresholds = config.travel.thresholds_min
    cutoff = float(max(thresholds))
    walk_flat = config.travel.speeds_kmh["walk"]
    bike_flat = config.travel.speeds_kmh["bike"]
    half_life = config.access.gravity_half_life_min

    access = pd.read_parquet(processed / ACCESS_FILE_TEMPLATE.format(res=res))
    index = pd.Index(access["h3"], name="h3")
    in_city = access["in_city"].to_numpy(dtype=bool)
    jobs = (
        pd.read_parquet(processed / JOBS_FILE_TEMPLATE.format(res=res), columns=["h3", "jobs_rf"])
        .set_index("h3")["jobs_rf"]
        .reindex(index)
        .fillna(0.0)
    )
    total_jobs = float(jobs.to_numpy()[in_city].sum())

    population = (
        pd.read_parquet(
            processed / FEATURES_FILE_TEMPLATE.format(res=res), columns=["h3", "pop_residents"]
        )
        .set_index("h3")["pop_residents"]
        .reindex(index)
        .fillna(0.0)
    )
    inhabited = in_city & (population.to_numpy() >= config.equity.min_population)
    weightings, weighting_info = _population_weightings(config, access, population, inhabited)

    grid = grid_points(config, res)
    loaded: dict[str, Any] = {}

    def walk_graph() -> ModeGraph:
        if "walk" not in loaded:
            loaded["walk"] = street_graph(
                config, "walk", lambda e: tobler_walk_kmh(e["grade"].to_numpy(), walk_flat)
            )
        return loaded["walk"]

    def transit_tables() -> tuple:
        if "transit" not in loaded:
            loaded["transit"] = (
                read_geoparquet(config.paths.interim / STOPS_FILE),
                pd.read_parquet(config.paths.interim / ROUTES_FILE),
                pd.read_parquet(config.paths.interim / LEGS_FILE),
            )
        return loaded["transit"]

    def bike_network() -> tuple:
        if "bike" not in loaded:
            files = interim_network_files(config, "bike")
            loaded["bike"] = (read_geoparquet(files["nodes"]), read_geoparquet(files["edges"]))
        return loaded["bike"]

    summary_rows: list[dict] = []
    gains_rows: list[dict] = []
    report_scenarios: dict[str, dict] = {}
    any_transit = False
    for scenario in scenarios:
        frame = access[["h3", "res", "district", "in_city"]].copy()
        network_stats: dict[str, dict] = {}
        matrix_stats: dict[str, dict] = {}
        scen_values: dict[str, np.ndarray] = {}

        if "transit" in scenario.modes:
            any_transit = True
            stops, routes, legs = transit_tables()
            application = apply_transit_scenario(scenario, stops, routes, legs, config=config)
            graph = build_transit_graph(
                walk_graph(),
                application.stops,
                application.routes,
                application.legs,
                walk_flat_kmh=walk_flat,
                speeds_kmh=application.speeds_kmh,
                circuity=application.circuity,
                metric_crs=config.crs.metric,
            )
            tt, matrix_stats["transit"] = hex_times(
                graph,
                grid,
                grid,
                walk_flat_kmh=walk_flat,
                cutoff_min=cutoff,
                snap_max_m=config.access.snap_max_m,
            )
            scen_values |= job_measures(
                tt, jobs, thresholds, half_life, index, total_jobs, "transit"
            )
            network_stats["transit"] = application.stats

        if "bike" in scenario.modes:
            nodes, edges = bike_network()
            edge_seconds, network_stats["bike"] = boosted_edge_seconds(
                edges, scenario.boosts, flat_kmh=bike_flat, metric_crs=config.crs.metric
            )
            graph = build_graph(nodes, edges, edge_seconds, config.crs.metric)
            tt, matrix_stats["bike"] = hex_times(
                graph,
                grid,
                grid,
                walk_flat_kmh=walk_flat,
                cutoff_min=cutoff,
                snap_max_m=config.access.snap_max_m,
            )
            scen_values |= job_measures(tt, jobs, thresholds, half_life, index, total_jobs, "bike")

        columns = [c for mode in scenario.modes for c in affected_columns(mode, thresholds)]
        for column in columns:
            base = access[column].to_numpy(dtype="float64")
            scen = scen_values[column]
            delta = scen - base
            worst = float(delta.min()) if len(delta) else 0.0
            if worst < 0:
                raise RuntimeError(
                    f"scenario {scenario.name}: negative delta {worst:.3f} in {column} — "
                    "a scenario only adds arcs, so this is a pipeline bug"
                )
            frame[f"{column}_base"] = base
            frame[f"{column}_scen"] = scen
            frame[f"{column}_delta"] = delta
        out_path = processed / SCENARIO_FILE_TEMPLATE.format(name=scenario.name, res=res)
        frame.to_parquet(out_path, index=False)
        logger.info("%s: %d hexes x %d columns", out_path.name, len(frame), len(frame.columns))

        entry = _evaluate(
            config,
            scenario,
            frame,
            inhabited,
            weightings,
            thresholds,
            summary_rows,
            gains_rows,
        )
        entry["network"] = network_stats
        entry["matrix"] = matrix_stats
        report_scenarios[scenario.name] = entry

    pd.DataFrame(summary_rows).to_parquet(processed / SUMMARY_FILE, index=False)
    pd.DataFrame(gains_rows).to_parquet(processed / GAINS_FILE, index=False)

    report: dict[str, Any] = {
        "approximate_transit": True,
        "resolution": res,
        "params": {
            "attach_max_m": config.scenarios.attach_max_m,
            "stop_spacing_m": config.scenarios.stop_spacing_m,
            "bike_buffer_m": config.scenarios.bike_buffer_m,
            "gain_cutoffs_jobs": list(config.scenarios.gain_cutoffs_jobs),
            "thresholds_min": list(thresholds),
            "gravity_half_life_min": half_life,
        },
        "total_jobs": round(total_jobs, 1),
        "weightings": weighting_info,
        "scenarios": report_scenarios,
        "caveats": _global_caveats(any_transit),
    }
    (processed / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")


def _population_weightings(
    config: Config,
    access: pd.DataFrame,
    population: pd.Series,
    inhabited: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """Raster residents and, when the series exists, calibrated residents.

    Factors are derived on the inhabited domain and carried to every
    in-city hex — the same recipe the equity build uses (D-058), so a
    scenario's Gini-before equals the equity report's Gini.
    """
    raster = population.to_numpy(dtype="float64")
    weightings = {RASTER: raster}
    info: dict[str, Any] = {RASTER: {"method": "Kontur raster residents as prepared in Phase 1"}}
    calibration = district_resident_shares(config)
    if calibration is None:
        info[CALIBRATED] = {
            "skipped_reason": f"no usable resident series in {STATS_FILE}",
        }
        return weightings, info
    domain_population = population[inhabited]
    domain_district = access.loc[inhabited, "district"]
    _, factors = calibrated_weights(domain_population, domain_district, calibration.shares)
    weightings[CALIBRATED] = apply_factors(
        population, access["district"].where(access["in_city"]), factors
    )
    info[CALIBRATED] = {
        "method": calibration.method,
        "synthetic": calibration.synthetic,
        "factors": {name: round(value, 4) for name, value in factors.items()},
    }
    return weightings, info


def _evaluate(
    config: Config,
    scenario: Scenario,
    frame: pd.DataFrame,
    inhabited: np.ndarray,
    weightings: dict[str, np.ndarray],
    thresholds: tuple[int, ...],
    summary_rows: list[dict],
    gains_rows: list[dict],
) -> dict:
    """Distribution changes and gains for one scenario; returns its report entry."""
    res = config.h3.analysis_resolution
    cutoffs = config.scenarios.gain_cutoffs_jobs
    entry: dict[str, Any] = {
        "title": scenario.title,
        "note": scenario.note,
        "sources": list(scenario.sources),
        "modes": list(scenario.modes),
        "results": {},
    }

    relative = [
        column
        for mode in scenario.modes
        for column in affected_columns(mode, thresholds)
        if column.startswith("accpct_") or column.endswith("_grav")
    ]
    headline_mode = scenario.modes[0]
    headline, headline_threshold = headline_measure(config, headline_mode)
    gain_column = f"acc_jobs_{headline_mode}_{headline_threshold}_delta"
    headline_numbers: dict[str, dict] = {}

    for weighting, weights in weightings.items():
        weight = weights[inhabited]
        block: dict[str, Any] = {"population": round(float(weight.sum()), 0), "measures": {}}
        for measure in relative:
            base = distribution_summary(frame.loc[inhabited, f"{measure}_base"].to_numpy(), weight)
            scen = distribution_summary(frame.loc[inhabited, f"{measure}_scen"].to_numpy(), weight)
            summary_rows.append(
                {
                    "scenario": scenario.name,
                    "res": res,
                    "weights": weighting,
                    "measure": measure,
                    "population": base["population"],
                    "mean_base": base["mean"],
                    "mean_scen": scen["mean"],
                    "median_base": base["median"],
                    "median_scen": scen["median"],
                    "gini_base": base["gini"],
                    "gini_scen": scen["gini"],
                    "gini_delta": scen["gini"] - base["gini"],
                }
            )
            block["measures"][measure] = {
                "mean": [_num(base["mean"], 3), _num(scen["mean"], 3)],
                "median": [_num(base["median"], 3), _num(scen["median"], 3)],
                "gini": [_num(base["gini"], 4), _num(scen["gini"], 4)],
            }

        delta = frame.loc[inhabited, gain_column].to_numpy()
        gains: dict[str, Any] = {}
        for cutoff_jobs in cutoffs:
            residents = float(weight[delta > cutoff_jobs].sum())
            share = residents / float(weight.sum()) if weight.sum() > 0 else float("nan")
            gains[f"{cutoff_jobs:g}"] = {
                "residents": round(residents, 0),
                "share": _num(share, 4),
            }
            gains_rows.append(
                {
                    "scenario": scenario.name,
                    "res": res,
                    "weights": weighting,
                    "measure": gain_column.removesuffix("_delta"),
                    "cutoff_jobs": float(cutoff_jobs),
                    "residents_gaining": residents,
                    "share_gaining": share,
                }
            )
        block["gains"] = gains
        entry["results"][weighting] = block
        headline_numbers[weighting] = {
            "gini": block["measures"][headline]["gini"],
            "gains": gains,
            "population": block["population"],
        }

    entry["headline_measure"] = headline
    entry["headline"] = _headline_sentence(
        scenario, headline_mode, headline_threshold, cutoffs, headline_numbers
    )
    entry["caveats"] = _scenario_caveats(scenario)
    return entry


def _range_text(values: list[float], fmt: str) -> str:
    low, high = min(values), max(values)
    if fmt.format(low) == fmt.format(high):
        return fmt.format(low)
    return f"between {fmt.format(low)} and {fmt.format(high)}"


def _headline_sentence(
    scenario: Scenario,
    mode: str,
    threshold: int,
    cutoffs: tuple[float, ...],
    numbers: dict[str, dict],
) -> str:
    """One sentence per scenario, quoting both weightings as a range.

    The quoted cutoff is the middle one of ``gain_cutoffs_jobs`` — small
    cutoffs count everyone marginally touched, the largest only the new
    line's closest neighbours.
    """
    cutoff = sorted(cutoffs)[len(cutoffs) // 2]
    key = f"{cutoff:g}"
    residents = [block["gains"][key]["residents"] for block in numbers.values()]
    shares = [(block["gains"][key]["share"] or 0.0) * 100 for block in numbers.values()]
    gini_from = [block["gini"][0] for block in numbers.values()]
    gini_to = [block["gini"][1] for block in numbers.values()]

    def gini_text(values: list[float | None]) -> str:
        if any(value is None for value in values):
            return "n/a (nobody reaches any jobs)"
        return _range_text(values, "{:.3f}")

    basis = "raster and calibrated weightings" if len(numbers) > 1 else "raster weighting only"
    return (
        f"{scenario.title}: {_range_text(residents, '{:,.0f}')} residents "
        f"({_range_text(shares, '{:.1f}')}% of the inhabited city) gain more than "
        f"{cutoff:,.0f} jobs reachable within {threshold} min by {mode}; "
        f"the {mode} access Gini moves from {gini_text(gini_from)} "
        f"to {gini_text(gini_to)} ({basis})."
    )


def _scenario_caveats(scenario: Scenario) -> list[str]:
    caveats = []
    if scenario.note:
        caveats.append(scenario.note)
    if "transit" in scenario.modes:
        caveats.append(
            "Deltas are measured against the approximate OSM lower-bound transit "
            "baseline (D-044): where the baseline misses real routes, part of the "
            "gain shown here already exists on the street."
        )
        new_lines = [line for line in scenario.lines if line.extends_route_ref is None]
        if new_lines:
            caveats.append(
                "Assumed operating parameters of the new line(s): "
                + "; ".join(
                    f"{line.name}: {line.speed_kmh:g} km/h, headway {line.headway_min:g} min"
                    for line in new_lines
                )
            )
    if "bike" in scenario.modes:
        caveats.append(
            "Bike corridors raise the FLAT riding speed on matched street edges only; "
            "grades, detours and the rest of the network are unchanged."
        )
    caveats.append(
        "The job surface is the fixed Phase 3 estimate: no relocation of jobs or "
        "residents in response to the new infrastructure is modelled."
    )
    return caveats


def _global_caveats(any_transit: bool) -> list[str]:
    caveats = [
        "Scenario alignments are ILLUSTRATIVE drawings anchored to publicly "
        "announced projects, not official designs; redraw the GeoJSON files "
        "to study other alignments.",
        "A scenario can only improve modelled times (arcs are added, never "
        "removed): construction disruption, service reorganisation and induced "
        "demand are out of scope.",
    ]
    if any_transit:
        caveats.append(
            "Transit remains the approximate lower-bound layer of Phase 4: assumed "
            "speeds and headways, OSM route relations only (DECISIONS D-044)."
        )
    return caveats
