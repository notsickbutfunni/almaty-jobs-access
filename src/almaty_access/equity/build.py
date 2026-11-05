"""Phase 5 orchestrator: who is underserved, and where.

Reads the Phase 4 accessibility tables and the resident counts, restricts
every statistic to INHABITED city hexes (equity is about people) and runs
the whole analysis under two population weightings (DECISIONS D-058):

``raster``
    Kontur Population as prepared in Phase 1.
``calibrated``
    The same raster rescaled so district totals follow the district
    shares of employed residents in the statistics file — the raster is
    skewed between districts, and that skew moves every
    population-weighted figure.
    Measures that contain population themselves (2SFCA: residents are
    its demand term) are RECOMPUTED with the calibrated residents, not
    merely re-weighted.

Outputs, all in data/processed/:

``equity_r{res}.parquet`` / ``equity_calibrated_r{res}.parquet``
    Per-hex layers for maps under each weighting: mode gap, LISA cluster
    and p-value, desert flag and bivariate class of every headline
    measure.
``equity_summary`` / ``_lorenz`` / ``_districts`` / ``_deserts`` /
``_nearest.parquet``
    Long tables with a ``weights`` column: distribution statistics,
    Lorenz curves, district ranking, ranked desert hexes, shares of
    residents within walking thresholds of the nearest facility.
``equity_report.json``
    Parameters, caveats, one result block per weighting and resolution,
    a raster-vs-calibrated comparison, and the one-sentence result the
    README asks for — stated as a range where the weightings disagree.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

import h3
import numpy as np
import pandas as pd
from scipy import stats

from almaty_access import schemas
from almaty_access.access.build import ACCESS_FILE_TEMPLATE, TT_FILE_TEMPLATE
from almaty_access.access.build import REPORT_FILE as ACCESS_REPORT_FILE
from almaty_access.access.measures import two_step_fca
from almaty_access.access.opportunities import (
    SFCA_TYPES,
    load_opportunity_points,
    opportunity_table,
)
from almaty_access.config import Config
from almaty_access.download.stats import FILE_NAME as STATS_FILE
from almaty_access.equity.deserts import access_deserts, bivariate_classes
from almaty_access.equity.inequality import distribution_summary, lorenz_curve, weighted_quantile
from almaty_access.equity.spatial import (
    MIN_HEXES_FOR_LISA,
    fdr_summary,
    global_moran,
    hex_weights,
    local_moran,
    monte_carlo_margin,
)
from almaty_access.equity.tables import (
    CITY_SCOPE,
    below_cutoff_shares,
    center_gradient,
    district_table,
    mode_gap,
    nearest_tables,
)
from almaty_access.equity.weights import (
    RESIDENT_INDICATOR,
    ResidentShares,
    apply_factors,
    calibrated_weights,
    district_resident_shares,
)
from almaty_access.features.build import FEATURES_FILE_TEMPLATE
from almaty_access.io import ensure_built, read_geoparquet
from almaty_access.model.build import JOBS_FILE_TEMPLATE

logger = logging.getLogger(__name__)

RASTER, CALIBRATED = schemas.EQUITY_WEIGHTINGS

EQUITY_FILE_TEMPLATE = "equity_r{res}.parquet"
EQUITY_CALIBRATED_FILE_TEMPLATE = "equity_calibrated_r{res}.parquet"
HEX_FILE_TEMPLATES = {RASTER: EQUITY_FILE_TEMPLATE, CALIBRATED: EQUITY_CALIBRATED_FILE_TEMPLATE}
SUMMARY_FILE = "equity_summary.parquet"
LORENZ_FILE = "equity_lorenz.parquet"
DISTRICTS_FILE = "equity_districts.parquet"
DESERTS_FILE = "equity_deserts.parquet"
NEAREST_FILE = "equity_nearest.parquet"
REPORT_FILE = "equity_report.json"

#: A transit/car ratio below this means the car reaches over four times
#: as many jobs: the report's "car-dependent" share.
CAR_DEPENDENT_RATIO = 0.25

#: Walking threshold quoted in the companion sentences when configured.
COMPANION_THRESHOLD_MIN = 15

#: A fine-resolution hex is a dasymetric POINT MASS when it holds at
#: least this share of the residents of its coarse parent ...
POINT_MASS_SHARE = 0.9
#: ... the parent is properly populated ...
POINT_MASS_MIN_PARENT = 1000.0
#: ... and at least this many of the parent's children are built up, so
#: the residents plainly do not all live in one of them.
POINT_MASS_MIN_BUILT_SIBLINGS = 3

#: A measure is called zero-inflated in the caveats above this share of
#: zero hexes (the statement is generated from the data, never assumed).
ZERO_INFLATED_SHARE = 0.5

#: Stored 2SFCA values are rounded to 4 decimals in Phase 4; a
#: recomputation on the same residents must agree within this.
SFCA_RECOMPUTE_ATOL = 1e-3

_JOB_MEASURE = re.compile(r"^accpct_jobs_(walk|bike|transit|car)_(\d+)$")
_SFCA_MEASURE = re.compile(r"^sfca_(\w+)_walk$")
_MODE_PHRASE = {
    "walk": "on foot",
    "bike": "by bike",
    "transit": "by transit (approximate network, a lower bound)",
    "car": "by car (free-flow, an upper bound)",
}


@dataclass
class _Analysis:
    """Everything one (weighting, resolution) run produces."""

    weighting: str
    res: int
    domain: pd.DataFrame
    hexes: pd.DataFrame
    block: dict
    summary: list[dict] = field(default_factory=list)
    lorenz: list[pd.DataFrame] = field(default_factory=list)
    districts: dict[str, pd.DataFrame] = field(default_factory=dict)
    deserts: dict[str, pd.DataFrame] = field(default_factory=dict)
    nearest: pd.DataFrame | None = None
    lisa: dict[str, pd.DataFrame] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _number(value: float | None, digits: int = 4) -> float | None:
    """Round a finite number; NaN / None become None (an infinity stays)."""
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return value if math.isinf(value) else round(value, digits)


def _jsonable(value: Any) -> Any:
    """Make a report strictly JSON-serializable.

    NaN becomes null; an infinity becomes the string ``"inf"`` — it has
    a meaning here (a Palma ratio whose bottom 40% hold nothing) that a
    null would erase.
    """
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return number
    return value


def _share_text(raw: float, calibrated: float | None) -> str:
    """'62%' — or 'Between 51% and 62%' when the weightings print differently."""
    if calibrated is None:
        return f"{raw:.0%}"
    low, high = sorted((raw, calibrated))
    if f"{low:.0%}" == f"{high:.0%}":
        return f"{raw:.0%}"
    return f"Between {low:.0%} and {high:.0%}"


def _resolutions(config: Config) -> tuple[int, int]:
    return (config.h3.analysis_resolution, config.h3.fine_resolution)


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _load_table(config: Config, res: int) -> pd.DataFrame:
    """Access table joined with residents, distance to the centre and flags."""
    processed = config.paths.processed
    access = pd.read_parquet(processed / ACCESS_FILE_TEMPLATE.format(res=res))
    features = pd.read_parquet(
        processed / FEATURES_FILE_TEMPLATE.format(res=res),
        columns=["h3", "pop_residents", "cent_dist_center_m", "bf_gfa_m2"],
    )
    table = access.merge(features, on="h3", how="left", validate="one_to_one")
    table["population"] = table["pop_residents"].fillna(0.0)
    table["center_km"] = table["cent_dist_center_m"] / 1000.0
    table["inhabited"] = table["in_city"].to_numpy(dtype=bool) & (
        table["population"].to_numpy() >= config.equity.min_population
    )
    table["point_mass"] = _point_mass(config, table, res)
    return table.drop(columns=["pop_residents", "cent_dist_center_m", "bf_gfa_m2"])


def _point_mass(config: Config, table: pd.DataFrame, res: int) -> np.ndarray:
    """Fine hexes that swallowed their coarse parent's residents.

    Phase 1 splits each coarse population hex over its children by
    residential-TAGGED floor area. Where a private-house area carries
    almost no such tags, one child receives everyone while its built-up
    siblings stay empty. Such a hex always passes a "dense" test and
    tops any ranking by residents, so it is flagged (never dropped).
    """
    parent_res = config.h3.analysis_resolution
    if res <= parent_res:
        return np.zeros(len(table), dtype=bool)
    parents = pd.Series([h3.cell_to_parent(cell, parent_res) for cell in table["h3"]])
    population = table["population"].reset_index(drop=True)
    built = table["bf_gfa_m2"].fillna(0.0).reset_index(drop=True) > 0
    parent_population = population.groupby(parents).transform("sum")
    built_siblings = built.groupby(parents).transform("sum")
    flag = (
        (population >= POINT_MASS_SHARE * parent_population)
        & (parent_population >= POINT_MASS_MIN_PARENT)
        & (built_siblings >= POINT_MASS_MIN_BUILT_SIBLINGS)
        & table["in_city"].reset_index(drop=True).astype(bool)
    )
    return flag.to_numpy()


def _analysed_measures(
    config: Config, table: pd.DataFrame, res: int, headline: list[str]
) -> list[str]:
    """Benefit columns summarised at this resolution.

    The analysis resolution keeps job measures (plus anything the config
    explicitly lists); service counts and 2SFCA are granularity noise on
    coarse hexes (D-049) and are summarised at the fine resolution only.
    """
    benefit = [c for c in table.columns if c.startswith(schemas.EQUITY_BENEFIT_PREFIXES)]
    if res == config.h3.analysis_resolution:
        return [c for c in benefit if "_jobs_" in c or c in headline]
    return benefit


def _sfca_columns(measures: list[str]) -> list[str]:
    return [
        m for m in measures if (match := _SFCA_MEASURE.match(m)) and match.group(1) in SFCA_TYPES
    ]


def _recompute_sfca(
    config: Config,
    res: int,
    table: pd.DataFrame,
    calibrated_population: np.ndarray,
    columns: list[str],
) -> dict[str, np.ndarray]:
    """2SFCA with calibrated residents as the demand term.

    Residents enter 2SFCA twice — as the weight of a statistic and as
    the demand competing for each facility. Swapping only the weight
    would produce figures that are neither raw nor calibrated (their
    mean would move although supply and residents did not). So the
    measure itself is rebuilt from the Phase 4 walk matrix. As a guard,
    the same code is first run on the raster residents and must
    reproduce the stored access columns.
    """
    walk = pd.read_parquet(config.paths.processed / TT_FILE_TEMPLATE.format(mode="walk", res=res))
    points = load_opportunity_points(config)
    opportunities = opportunity_table(config, points, res, table["h3"])
    cells = pd.Index(table["h3"], name="h3")
    catchment = float(config.access.sfca_catchment_min)
    raster = pd.Series(table["population"].to_numpy(), index=cells)
    calibrated = pd.Series(calibrated_population, index=cells)

    def run(service: str, residents: pd.Series) -> np.ndarray:
        values = two_step_fca(walk, opportunities[service], residents, catchment)
        return np.round(values.reindex(cells).fillna(0.0).to_numpy(), 4)

    out: dict[str, np.ndarray] = {}
    for column in columns:
        service = _SFCA_MEASURE.match(column).group(1)
        reproduced = run(service, raster)
        stored = table[column].to_numpy(dtype="float64")
        if not np.allclose(reproduced, stored, atol=SFCA_RECOMPUTE_ATOL):
            worst = float(np.abs(reproduced - stored).max())
            raise ValueError(
                f"cannot reproduce {column} at r{res} from the walk matrix "
                f"(max difference {worst:.4f}): access outputs and the equity "
                "recomputation have drifted apart - rebuild `almaty-access access`"
            )
        out[column] = run(service, calibrated)
    return out


def _stop_pois(config: Config, res: int) -> pd.Series:
    """Mapped transit-stop POIs per hex (Phase 1 POI layer)."""
    pois = read_geoparquet(config.paths.interim / "pois.parquet")
    stops = pois.loc[pois["category"] == "transit_stop"]
    cells = [h3.latlng_to_cell(point.y, point.x, res) for point in stops.geometry]
    return pd.Series(cells, dtype="string").value_counts()


# ---------------------------------------------------------------------------
# Analysis blocks
# ---------------------------------------------------------------------------


def _lisa_block(
    config: Config,
    res: int,
    domain: pd.DataFrame,
    measures: list[str],
    cache: dict,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """LISA per headline measure: per-hex frames (keyed by h3) and report.

    Cluster LOCATIONS do not depend on the weighting unless the measure's
    values do (2SFCA with calibrated demand), so results are cached on
    the values and only the resident shares are recomputed.
    """
    frames: dict[str, pd.DataFrame] = {}
    report: dict[str, dict] = {}
    if len(domain) < MIN_HEXES_FOR_LISA:
        reason = (
            f"only {len(domain)} inhabited hexes (< {MIN_HEXES_FOR_LISA}): "
            "local statistics would be noise"
        )
        return frames, {measure: {"skipped_reason": reason} for measure in measures}

    cells = domain["h3"].tolist()
    if ("weights", res) not in cache:
        cache[("weights", res)] = hex_weights(cells)
    weights, islands = cache[("weights", res)]
    population = domain["population"].to_numpy()
    total = float(population.sum())
    equity = config.equity
    for measure in measures:
        values = domain[measure].to_numpy(dtype="float64")
        if np.ptp(values) == 0:
            report[measure] = {"skipped_reason": "measure is constant over the inhabited hexes"}
            continue
        cached = cache.get((res, measure))
        if cached is None or not np.array_equal(cached[0], values):
            moran = global_moran(
                values, weights, permutations=equity.lisa_permutations, seed=config.random_seed
            )
            local = local_moran(
                values,
                weights,
                islands,
                permutations=equity.lisa_permutations,
                seed=config.random_seed,
                alpha=equity.lisa_alpha,
            )
            local.insert(0, "h3", cells)  # same order as the weights were built in
            cached = (values, moran, local)
            cache[(res, measure)] = cached
        _, moran, local = cached
        frames[measure] = local
        labels = local["label"].to_numpy()
        survives = local["fdr"].to_numpy()
        cold, hot = labels == "LL", labels == "HH"
        report[measure] = {
            "global_moran": moran,
            "zero_share_hexes": _number(float((values == 0).mean())),
            "clusters": {str(k): int(v) for k, v in pd.Series(labels).value_counts().items()},
            # Hexes within Monte Carlo error of alpha: another seed could
            # move the cluster counts by about this much.
            "borderline": monte_carlo_margin(
                local["p"].to_numpy(), equity.lisa_alpha, equity.lisa_permutations
            ),
            # Counts are None where the permutations cannot resolve them.
            "fdr": fdr_summary(local, equity.lisa_alpha, equity.lisa_permutations),
            "cold_spot_population": round(float(population[cold].sum()), 0),
            "cold_spot_pop_share": _number(population[cold].sum() / total),
            "hot_spot_pop_share": _number(population[hot].sum() / total),
            "cold_spot_fdr_pop_share": _number(population[cold & survives].sum() / total),
        }
    return frames, report


def _transit_equals_walk(domain: pd.DataFrame, threshold: int) -> np.ndarray:
    return np.isclose(
        domain[f"acc_jobs_transit_{threshold}"],
        domain[f"acc_jobs_walk_{threshold}"],
        rtol=1e-9,
        atol=1e-6,
    )


def _mode_gap_block(config: Config, domain: pd.DataFrame) -> dict | None:
    """Transit / car summary over residents, per threshold."""
    thresholds = config.travel.thresholds_min
    needed = [f"acc_jobs_{mode}_{t}" for mode in ("transit", "car", "walk") for t in thresholds]
    if any(column not in domain.columns for column in needed):
        return None
    gaps = mode_gap(domain, thresholds)
    population = domain["population"].to_numpy()
    total = population.sum()
    out: dict[str, dict] = {}
    for threshold, column in zip(thresholds, gaps.columns, strict=True):
        ratio = gaps[column].to_numpy()
        defined = np.isfinite(ratio)
        weight = population[defined]
        same = _transit_equals_walk(domain, threshold)
        out[column] = {
            "median": _number(weighted_quantile(ratio[defined], weight, 0.5)),
            "pop_share_car_dependent": _number(
                weight[ratio[defined] < CAR_DEPENDENT_RATIO].sum() / total
            ),
            "pop_share_undefined": _number(population[~defined].sum() / total),
            # Where transit adds nothing the "transit" ratio is a walk ratio.
            "pop_share_transit_equals_walk": _number(population[same].sum() / total),
        }
    return out


def _transit_coverage(
    config: Config, domain: pd.DataFrame, headline_measure: str, stop_pois: pd.Series
) -> dict | None:
    """How much of the "transit" layer is really transit?

    The network is a lower bound (D-044); this block makes the gap
    measurable instead of leaving it to a label: where modelled transit
    adds nothing over walking, and how many residents live in hexes
    that have mapped stops but no modelled route.
    """
    thresholds = config.travel.thresholds_min
    needed = [f"acc_jobs_{mode}_{t}" for mode in ("transit", "walk") for t in thresholds]
    if any(column not in domain.columns for column in needed):
        return None
    population = domain["population"].to_numpy()
    total = population.sum()
    routes = domain["transit_routes_cnt"].to_numpy() > 0
    has_stop = domain["h3"].map(stop_pois).fillna(0).to_numpy() > 0
    block: dict = {
        "pop_share_transit_equals_walk": {
            str(t): _number(population[_transit_equals_walk(domain, t)].sum() / total)
            for t in thresholds
        },
        "pop_share_in_hexes_with_modelled_routes": _number(population[routes].sum() / total),
        "pop_share_in_hexes_with_stop_pois_but_no_route": _number(
            population[has_stop & ~routes].sum() / total
        ),
    }
    match = _JOB_MEASURE.match(headline_measure)
    if match and match.group(1) == "transit":
        minutes = int(match.group(2))
        equal = _transit_equals_walk(domain, minutes)
        districts = domain["district"].astype("string").fillna("").to_numpy(dtype=str)
        block["by_district_transit_equals_walk"] = {
            name: _number(
                population[equal & (districts == name)].sum() / population[districts == name].sum()
            )
            for name in sorted(set(districts) - {""})
        }
        twin = f"accpct_jobs_walk_{minutes}"
        if twin in domain.columns:
            cutoff = config.equity.headline_cutoff_pct
            block["headline_walk_only"] = {
                "measure": twin,
                "pop_share": _number(population[domain[twin].to_numpy() < cutoff].sum() / total),
            }
    return block


def _analyse(
    config: Config,
    res: int,
    table: pd.DataFrame,
    weighting: str,
    measures: list[str],
    headline: list[str],
    lisa_cache: dict,
    stop_pois: pd.Series | None,
) -> _Analysis:
    """All outputs of one resolution under one weighting.

    ``table["population"]`` holds the residents of THIS weighting; the
    inhabited domain is the same set of hexes under both.
    """
    equity = config.equity
    thresholds = config.travel.thresholds_min
    domain = table.loc[table["inhabited"]].reset_index(drop=True)
    base = table[list(schemas.EQUITY_ID_COLUMNS)].copy()
    if all(f"acc_jobs_{m}_{t}" in table.columns for m in ("transit", "car") for t in thresholds):
        base = pd.concat([base, mode_gap(table, thresholds)], axis=1)
    block: dict = {
        "population": round(float(domain["population"].sum()), 0),
        "inhabited_hexes": len(domain),
    }
    result = _Analysis(weighting=weighting, res=res, domain=domain, hexes=base, block=block)
    if not len(domain):
        logger.warning("r%d: no inhabited hexes, equity outputs are empty", res)
        return result

    population = domain["population"].to_numpy()
    for measure in measures:
        row = distribution_summary(domain[measure].to_numpy(), population)
        result.summary.append({"res": res, "weights": weighting, "measure": measure, **row})

    result.lisa, block["lisa"] = _lisa_block(config, res, domain, headline, lisa_cache)
    layers = pd.DataFrame({"h3": domain["h3"].to_numpy()})
    block["measures"], block["deserts"], block["below_cutoffs"] = {}, {}, {}
    routes = domain["transit_routes_cnt"].to_numpy() > 0
    has_stop = None if stop_pois is None else domain["h3"].map(stop_pois).fillna(0).to_numpy() > 0
    for measure in headline:
        values = domain[measure].to_numpy()
        stats_row = distribution_summary(values, population)
        block["measures"][measure] = {
            key: _number(stats_row[key]) for key in ("mean", "median", "gini", "palma")
        }
        grid, shares = lorenz_curve(values, population)
        result.lorenz.append(
            pd.DataFrame(
                {
                    "res": res,
                    "weights": weighting,
                    "measure": measure,
                    "pop_share": grid,
                    "access_share": shares,
                }
            )
        )
        result.districts[measure] = district_table(domain, measure)
        deserts, info = access_deserts(
            domain,
            measure,
            pop_quantile=equity.desert_pop_quantile,
            access_quantile=equity.desert_access_quantile,
        )
        is_desert = domain["h3"].isin(deserts["h3"]).to_numpy()
        match = _JOB_MEASURE.match(measure)
        if match and match.group(1) == "transit" and has_stop is not None:
            # A "transit desert" without a modelled route is a walking
            # value: say how many of them there are.
            info["hexes_with_modelled_routes"] = int((is_desert & routes).sum())
            info["hexes_with_stop_pois"] = int((is_desert & has_stop).sum())
        block["deserts"][measure] = info
        result.deserts[measure] = deserts
        if measure.startswith("accpct_"):
            block["below_cutoffs"][measure] = below_cutoff_shares(
                domain, measure, equity.job_share_cutoffs_pct
            )

        local = result.lisa.get(measure)
        if local is None:
            layers[f"lisa_{measure}"] = pd.Series(pd.NA, index=layers.index, dtype="string")
            layers[f"lisap_{measure}"] = np.nan
        else:
            by_cell = local.set_index("h3")
            layers[f"lisa_{measure}"] = by_cell["label"].reindex(layers["h3"]).to_numpy()
            layers[f"lisap_{measure}"] = np.round(by_cell["p"].reindex(layers["h3"]).to_numpy(), 4)
        layers[f"desert_{measure}"] = is_desert
        layers[f"biv_{measure}"] = bivariate_classes(
            domain["population"], domain[measure]
        ).to_numpy()

    hexes = base.merge(layers, on="h3", how="left", validate="one_to_one")
    for column in layers.columns:
        if column.startswith("desert_"):
            hexes[column] = hexes[column].fillna(False).astype(bool)
        elif column.startswith(("lisa_", "biv_")):
            hexes[column] = hexes[column].astype("string")
    result.hexes = hexes

    gap = _mode_gap_block(config, domain)
    if gap is not None:
        block["mode_gap"] = gap
    if stop_pois is not None and headline:
        coverage = _transit_coverage(config, domain, headline[0], stop_pois)
        if coverage is not None:
            block["transit_coverage"] = coverage

    nearest, nearest_summary = nearest_tables(domain, equity.nearest_thresholds_min)
    result.nearest = nearest
    block["nearest"] = nearest_summary
    block["center_gradient"] = center_gradient(domain, headline, equity.center_distance_bands_km)
    block["center_spearman"] = {
        measure: _number(stats.spearmanr(domain["center_km"], domain[measure]).statistic)
        for measure in headline
        if len(domain) > 2 and np.ptp(domain[measure].to_numpy()) > 0
    }
    return result


# ---------------------------------------------------------------------------
# Raster vs calibrated, the headline, the caveats
# ---------------------------------------------------------------------------


def _comparison(raster: _Analysis, calibrated: _Analysis, headline: list[str]) -> dict:
    """What the choice of population weights changes, measure by measure."""
    out: dict[str, dict] = {}
    for measure in headline:
        entry: dict = {}
        for key in ("mean", "median", "gini", "palma"):
            entry[key] = {
                RASTER: raster.block["measures"][measure][key],
                CALIBRATED: calibrated.block["measures"][measure][key],
            }
        if measure in raster.block["below_cutoffs"]:
            entry["below_cutoffs"] = {
                cut: {RASTER: share, CALIBRATED: calibrated.block["below_cutoffs"][measure][cut]}
                for cut, share in raster.block["below_cutoffs"][measure].items()
            }

        first, second = raster.deserts[measure], calibrated.deserts[measure]
        a, b = set(first["h3"]), set(second["h3"])
        entry["deserts"] = {
            RASTER: len(a),
            CALIBRATED: len(b),
            "both": len(a & b),
            "raster_only": len(a - b),
            "calibrated_only": len(b - a),
            "same_top_hex": bool(len(first) and len(second))
            and first["h3"].iloc[0] == second["h3"].iloc[0],
        }

        if measure in raster.lisa and measure in calibrated.lisa:
            hot_a = set(raster.lisa[measure].query("label == 'HH'")["h3"])
            hot_b = set(calibrated.lisa[measure].query("label == 'HH'")["h3"])
            entry["hot_spot_hexes"] = {
                RASTER: len(hot_a),
                CALIBRATED: len(hot_b),
                "both": len(hot_a & hot_b),
            }

        ranks_a = raster.districts[measure].set_index("district")["rank"]
        ranks_b = calibrated.districts[measure].set_index("district")["rank"]
        entry["district_ranks_changed"] = {
            str(name): {RASTER: int(ranks_a[name]), CALIBRATED: int(ranks_b[name])}
            for name in ranks_a.index
            if name in ranks_b.index and ranks_a[name] != ranks_b[name]
        }
        out[measure] = entry
    return out


def _headline(
    config: Config,
    raster: _Analysis,
    calibrated: _Analysis | None,
    measure: str,
) -> dict:
    """The README's one-sentence result, with the numbers behind it.

    The share is computed on both weightings and stated as a range when
    they print differently. "Concentrated in" names only districts that
    are among the three with the most affected residents under BOTH
    weightings — the third place swaps with the weights, and a sentence
    hedged on its percentage must not be falsely firm about its places.
    """
    cutoff = config.equity.headline_cutoff_pct

    def affected(analysis: _Analysis) -> tuple[float, pd.Series, dict]:
        # Values AND weights of the same run: a measure that contains
        # residents (2SFCA) differs between the weightings.
        domain = analysis.domain
        weights = domain["population"].to_numpy()
        below = domain[measure].to_numpy() < cutoff
        districts = domain["district"].astype("string").fillna("").to_numpy(dtype=str)
        by_district = pd.Series(weights[below], index=districts[below]).groupby(level=0).sum()
        by_district = by_district.drop(labels=[""], errors="ignore")
        rates = {
            name: _number(
                weights[below & (districts == name)].sum() / weights[districts == name].sum()
            )
            for name in sorted(set(districts) - {""})
        }
        return (
            float(weights[below].sum() / weights.sum()),
            by_district.sort_values(ascending=False, kind="stable"),
            rates,
        )

    share, top_raster, rates = affected(raster)
    share_calibrated, top_calibrated, rates_calibrated = (
        (None, None, None) if calibrated is None else affected(calibrated)
    )
    if top_calibrated is None:
        named = list(top_raster.index[:3])
    else:
        named = [name for name in top_calibrated.index[:3] if name in set(top_raster.index[:3])]

    quantity = _share_text(share, share_calibrated)
    hedge = (
        " - depending on how the population raster is calibrated -"
        if quantity.startswith("Between")
        else ""
    )
    where = f", concentrated in {_join([n.capitalize() for n in named])}" if named else ""
    match = _JOB_MEASURE.match(measure)
    if match:
        mode, minutes = match.group(1), int(match.group(2))
        sentence = (
            f"{quantity} of residents{hedge} can reach fewer than {cutoff:g}% of city jobs "
            f"within {minutes} min {_MODE_PHRASE[mode]}{where}."
        )
    else:
        sentence = f"{quantity} of residents{hedge} have {measure} below {cutoff:g}{where}."
    return {
        "sentence": sentence,
        "measure": measure,
        "res": raster.res,
        "cutoff": cutoff,
        "pop_share": _number(share),
        "pop_share_calibrated": _number(share_calibrated),
        "districts_named": named,
        # Affected residents per district under each weighting (the raster
        # total is itself an undercount: read these as shares, not counts).
        "top_districts_raster": {k: round(float(v), 0) for k, v in top_raster.head(3).items()},
        "top_districts_calibrated": (
            None
            if top_calibrated is None
            else {k: round(float(v), 0) for k, v in top_calibrated.head(3).items()}
        ),
        # Share of each district's own residents below the cutoff. For a
        # measure without residents in it the two are identical:
        # calibration rescales whole districts.
        "district_rates": rates,
        "district_rates_calibrated": rates_calibrated,
    }


def _companions(config: Config, raster: _Analysis, calibrated: _Analysis | None) -> list[str]:
    """Walk-based sentences that do not depend on the transit layer."""
    thresholds = config.equity.nearest_thresholds_min
    threshold = COMPANION_THRESHOLD_MIN if COMPANION_THRESHOLD_MIN in thresholds else thresholds[0]

    def city_rows(analysis: _Analysis) -> pd.Series:
        table = analysis.nearest
        rows = table.loc[(table["threshold_min"] == threshold) & (table["district"] == CITY_SCOPE)]
        return rows.set_index("facility")["pop_share_within"]

    if raster.nearest is None:
        return []
    raw = city_rows(raster)
    other = None if calibrated is None or calibrated.nearest is None else city_rows(calibrated)
    lines = []
    for facility, share in raw.items():
        quantity = _share_text(float(share), None if other is None else float(other[facility]))
        note = "; raster vs calibrated residents" if quantity.startswith("Between") else ""
        lines.append(
            f"{quantity} of residents live within a {threshold}-minute walk of the "
            f"nearest {facility} (r{raster.res}{note})."
        )
    return lines


def _caveats(
    config: Config,
    analyses: dict[tuple[str, int], _Analysis],
    calibration: ResidentShares | None,
    factors_fine: dict[str, float],
) -> list[str]:
    """Everything a reader must know before quoting a number."""
    analysis_res, fine_res = _resolutions(config)
    raster = analyses[(RASTER, analysis_res)]
    access_report = json.loads((config.paths.processed / ACCESS_REPORT_FILE).read_text("utf-8"))
    routes = access_report["transit_network"]["routes_kept"]
    served = int((raster.domain["transit_routes_cnt"] > 0).sum()) if len(raster.domain) else 0
    lines = [
        "Transit is approximate and a LOWER BOUND: OSM route relations with assumed speeds "
        f"and headways; {routes} routes with ordered stops are modelled and {served} of "
        f"{len(raster.domain)} inhabited r{analysis_res} hexes host a modelled stop "
        "(DECISIONS D-044, D-047).",
        "Car times are free-flow (no congestion): an UPPER BOUND on car access (D-046).",
        "The mode gap divides a lower bound (transit) by an upper bound (car): it "
        "OVERSTATES car dependence and should be read as a ranking, not a level.",
        "Jobs per hex are a model estimate (Phase 3 dasymetric surface), not a census.",
        "Service counts and 2SFCA are hex-aggregated: they are analysed at the fine "
        "resolution only (D-049).",
        "Deserts: population above its hex quantile AND access at or below the "
        "population-weighted access quantile (D-055). The README-literal rule (both "
        "quantiles across hexes) is counted beside every result as n_hexes_literal_rule; "
        "a measure without a meaningful low tail (free-flow car) reports no deserts.",
        "LISA p-values are tie-aware two-sided pseudo p-values from seeded conditional "
        "permutations, so cluster counts are Monte Carlo estimates: `borderline` gives, per "
        "measure, the hexes within two standard errors of alpha that another seed could "
        "flip (on zero-heavy measures they flip in blocks). One test per hex also means "
        "multiple testing - the FDR counts are the conservative reading, and null where "
        "the permutation count cannot resolve them.",
    ]
    other = analyses.get((CALIBRATED, analysis_res))
    coverage = raster.block.get("transit_coverage")
    if coverage:
        shares = coverage["pop_share_transit_equals_walk"]
        twin = None if other is None else other.block["transit_coverage"]
        listed = ", ".join(
            _share_text(
                float(value),
                None if twin is None else float(twin["pop_share_transit_equals_walk"][t]),
            ).replace("Between ", "")
            + f" at {t} min"
            for t, value in shares.items()
        )
        lines.append(
            "For many residents the modelled transit adds nothing over walking (share of "
            f"residents with transit = walk job access, across the weightings: {listed}); "
            "transit rankings and transit deserts partly measure where OSM lacks stop members."
        )
    fine = analyses.get((RASTER, fine_res))
    if fine is not None and len(fine.domain):
        zero_shares = [
            entry["zero_share_hexes"]
            for name, entry in fine.block.get("lisa", {}).items()
            if _SFCA_MEASURE.match(name) and entry.get("zero_share_hexes") is not None
        ]
        if zero_shares and max(zero_shares) > ZERO_INFLATED_SHARE:
            lines.append(
                f"2SFCA is zero-inflated ({min(zero_shares):.0%}-{max(zero_shares):.0%} of "
                f"inhabited r{fine_res} hexes reach no facility within the catchment): its "
                "LISA can flag provision hot spots, but a zero among zeros is not "
                "statistically surprising and does not become a cold spot."
            )
        point = fine.domain.loc[fine.domain["point_mass"]]
        if len(point):
            lines.append(
                f"{len(point)} r{fine_res} hexes ({point['population'].sum():,.0f} raster "
                "residents) hold nearly all residents of their coarse parent although the "
                "neighbouring children are built up: an artifact of the residential-floor-"
                "area split (flag `point_mass`). They top rankings by residents - read "
                "fine-resolution ranks with that flag."
            )
    if calibration is None:
        lines.append(
            "Residents per hex come from Kontur Population, a modelled raster; no "
            "residence-based series was available to check its district split."
        )
        return lines
    kind = "SYNTHETIC test numbers" if calibration.synthetic else f"source: {calibration.source}"
    skew = ""
    if len(factors_fine) > 1:
        most = min(factors_fine, key=factors_fine.get)
        least = max(factors_fine, key=factors_fine.get)
        skew = (
            f" it gives {most.capitalize()} {1 / factors_fine[most]:.2f}x and "
            f"{least.capitalize()} {1 / factors_fine[least]:.2f}x their share "
            f"(r{fine_res} grid)"
        )
    lines.append(
        "Residents per hex come from Kontur Population, a modelled raster that is skewed "
        f"between districts: against the district shares of employed residents ({kind})"
        f"{skew}. Every population-weighted result is therefore produced twice - `raster` "
        "and `calibrated` - and 2SFCA is recomputed with calibrated demand; `comparison` "
        "lists what changes. Absolute resident counts inherit the raster's undercount: "
        "prefer shares."
    )
    return lines


# ---------------------------------------------------------------------------
# The cached step
# ---------------------------------------------------------------------------


def build_equity(config: Config, *, force: bool = False) -> None:
    """Equity tables, per-hex layers and the report (cached step)."""
    processed = config.paths.processed
    resolutions = _resolutions(config)
    calibration = district_resident_shares(config)
    weightings = (RASTER, CALIBRATED) if calibration is not None else (RASTER,)
    outputs = [
        *(
            processed / HEX_FILE_TEMPLATES[weighting].format(res=res)
            for weighting in weightings
            for res in resolutions
        ),
        processed / SUMMARY_FILE,
        processed / LORENZ_FILE,
        processed / DISTRICTS_FILE,
        processed / DESERTS_FILE,
        processed / NEAREST_FILE,
        processed / REPORT_FILE,
    ]
    inputs = [
        *(processed / ACCESS_FILE_TEMPLATE.format(res=res) for res in resolutions),
        *(processed / FEATURES_FILE_TEMPLATE.format(res=res) for res in resolutions),
        *(processed / JOBS_FILE_TEMPLATE.format(res=res) for res in resolutions),
        *(processed / TT_FILE_TEMPLATE.format(mode="walk", res=res) for res in resolutions),
        processed / ACCESS_REPORT_FILE,
        config.paths.interim / "pois.parquet",
        config.paths.raw / "osm_landuse.parquet",
        config.source_path,
    ]
    stats_path = config.paths.raw / STATS_FILE
    if stats_path.exists():  # optional: without it the calibrated run is skipped
        inputs.append(stats_path)
    # Per-hex files of a calibrated run that no longer applies (the
    # residence series disappeared) must not survive next to a report
    # that says the calibration was skipped.
    orphans = [
        path
        for res in resolutions
        if calibration is None
        and (path := processed / HEX_FILE_TEMPLATES[CALIBRATED].format(res=res)).exists()
    ]

    def build() -> None:
        for path in orphans:
            path.unlink()
        equity = config.equity
        analysis_res, fine_res = resolutions
        tables = {res: _load_table(config, res) for res in resolutions}
        headline_by_res = {
            analysis_res: list(equity.headline_analysis),
            fine_res: list(equity.headline_fine),
        }
        # Fail on a misspelled headline measure BEFORE writing anything.
        for res, level in zip(resolutions, ("analysis", "fine"), strict=True):
            missing = [m for m in headline_by_res[res] if m not in tables[res].columns]
            if missing:
                raise ValueError(
                    f"equity.headline_measures.{level} names columns that "
                    f"access_r{res}.parquet does not have: {missing}"
                )

        analyses: dict[tuple[str, int], _Analysis] = {}
        lisa_cache: dict = {}
        factors_by_res: dict[int, dict[str, float]] = {}
        shares_by_res: dict[int, dict] = {}
        first_threshold = config.travel.thresholds_min[0]
        for res in resolutions:
            table = tables[res]
            headline = headline_by_res[res]
            measures = _analysed_measures(config, table, res, headline)
            has_transit = f"acc_jobs_transit_{first_threshold}" in table.columns
            stop_pois = _stop_pois(config, res) if has_transit else None
            analyses[(RASTER, res)] = _analyse(
                config, res, table, RASTER, measures, headline, lisa_cache, stop_pois
            )
            domain = analyses[(RASTER, res)].domain
            if calibration is None or not len(domain):
                continue
            _, factors = calibrated_weights(
                domain["population"], domain["district"], calibration.shares
            )
            factors_by_res[res] = factors
            raster_totals = domain.groupby("district")["population"].sum()
            target = calibration.shares.reindex(raster_totals.index)
            shares_by_res[res] = {
                str(name): {
                    RASTER: _number(raster_totals[name] / raster_totals.sum()),
                    "target": _number(target[name] / target.sum()),
                }
                for name in raster_totals.index
            }
            calibrated_table = table.copy()
            calibrated_table["population"] = apply_factors(
                table["population"], table["district"].where(table["in_city"]), factors
            )
            sfca = _sfca_columns(measures)
            if sfca:
                recomputed = _recompute_sfca(
                    config, res, table, calibrated_table["population"].to_numpy(), sfca
                )
                for column, values in recomputed.items():
                    calibrated_table[column] = values
            analysis = _analyse(
                config,
                res,
                calibrated_table,
                CALIBRATED,
                measures,
                headline,
                lisa_cache,
                stop_pois,
            )
            if sfca:
                # The calibrated file carries the measure values that differ.
                analysis.hexes = analysis.hexes.merge(
                    calibrated_table[["h3", *sfca]], on="h3", validate="one_to_one"
                )
            analyses[(CALIBRATED, res)] = analysis

        # ---- per-hex files and long tables --------------------------------
        for weighting in weightings:
            for res in resolutions:
                analysis = analyses.get((weighting, res)) or analyses[(RASTER, res)]
                analysis.hexes.to_parquet(
                    processed / HEX_FILE_TEMPLATES[weighting].format(res=res), index=False
                )

        def stack(parts: list[pd.DataFrame], columns: tuple[str, ...]) -> pd.DataFrame:
            parts = [part for part in parts if len(part)]
            if not parts:
                return pd.DataFrame(columns=list(columns))
            return pd.concat(parts, ignore_index=True)[list(columns)]

        def tagged(frame: pd.DataFrame, analysis: _Analysis, measure: str | None) -> pd.DataFrame:
            frame = frame.copy()
            if measure is not None:
                frame.insert(0, "measure", measure)
            frame.insert(0, "weights", analysis.weighting)
            frame.insert(0, "res", analysis.res)
            return frame

        every = list(analyses.values())
        pd.DataFrame(
            [row for a in every for row in a.summary],
            columns=list(schemas.PROCESSED_EQUITY_SUMMARY),
        ).to_parquet(processed / SUMMARY_FILE, index=False)
        stack(
            [frame for a in every for frame in a.lorenz], schemas.PROCESSED_EQUITY_LORENZ
        ).to_parquet(processed / LORENZ_FILE, index=False)
        stack(
            [tagged(frame, a, m) for a in every for m, frame in a.districts.items()],
            schemas.PROCESSED_EQUITY_DISTRICTS,
        ).to_parquet(processed / DISTRICTS_FILE, index=False)
        stack(
            [tagged(frame, a, m) for a in every for m, frame in a.deserts.items()],
            schemas.PROCESSED_EQUITY_DESERTS,
        ).to_parquet(processed / DESERTS_FILE, index=False)
        stack(
            [tagged(a.nearest, a, None) for a in every if a.nearest is not None],
            schemas.PROCESSED_EQUITY_NEAREST,
        ).to_parquet(processed / NEAREST_FILE, index=False)

        # ---- report --------------------------------------------------------
        report: dict = {
            "params": {
                "min_population": equity.min_population,
                "headline_measures": {f"r{res}": headline_by_res[res] for res in resolutions},
                "lisa_permutations": equity.lisa_permutations,
                "lisa_alpha": equity.lisa_alpha,
                "lisa_p_value": "tie-aware two-sided (twice the smaller tail)",
                "desert_pop_quantile": equity.desert_pop_quantile,
                "desert_access_quantile": equity.desert_access_quantile,
                "job_share_cutoffs_pct": list(equity.job_share_cutoffs_pct),
                "headline_cutoff_pct": equity.headline_cutoff_pct,
                "nearest_thresholds_min": list(equity.nearest_thresholds_min),
                "center_distance_bands_km": list(equity.center_distance_bands_km),
                "car_dependent_ratio": CAR_DEPENDENT_RATIO,
                "random_seed": config.random_seed,
            },
            "weightings": {RASTER: "Kontur Population as prepared in Phase 1"},
        }
        if calibration is None:
            report["calibration"] = {
                "skipped_reason": f"no usable '{RESIDENT_INDICATOR}' series in {STATS_FILE}"
            }
        else:
            report["weightings"][CALIBRATED] = calibration.method
            report["calibration"] = {
                "indicator": RESIDENT_INDICATOR,
                "year": calibration.year,
                "source": calibration.source,
                "synthetic": calibration.synthetic,
                # target share / raster share, per grid (coarse hexes are
                # assigned to districts whole, so the grids differ a little).
                "district_factors": {
                    f"r{res}": {k: round(v, 4) for k, v in factors.items()}
                    for res, factors in factors_by_res.items()
                },
                "district_shares": {f"r{res}": v for res, v in shares_by_res.items()},
            }

        report["population"] = {}
        for res in resolutions:
            table = tables[res]
            raster = analyses[(RASTER, res)]
            city = float(table.loc[table["in_city"], "population"].sum())
            point = raster.domain.loc[raster.domain["point_mass"]]
            report["population"][f"r{res}"] = {
                "city": round(city, 0),
                "inhabited_hexes": len(raster.domain),
                "inhabited_population": raster.block["population"],
                "inhabited_pop_share": _number(
                    raster.block["population"] / city if city > 0 else math.nan, 6
                ),
                "point_mass_hexes": len(point),
                "point_mass_population": round(float(point["population"].sum()), 0),
            }

        report["results"] = {
            weighting: {
                f"r{res}": analyses[(weighting, res)].block
                for res in resolutions
                if (weighting, res) in analyses
            }
            for weighting in weightings
        }
        if calibration is not None:
            report["comparison"] = {
                f"r{res}": _comparison(
                    analyses[(RASTER, res)], analyses[(CALIBRATED, res)], headline_by_res[res]
                )
                for res in resolutions
                if (CALIBRATED, res) in analyses
            }

        raster_analysis = analyses[(RASTER, analysis_res)]
        if len(raster_analysis.domain):
            report["headline"] = _headline(
                config,
                raster_analysis,
                analyses.get((CALIBRATED, analysis_res)),
                equity.headline_analysis[0],
            )
        raster_fine = analyses[(RASTER, fine_res)]
        report["companions"] = (
            _companions(config, raster_fine, analyses.get((CALIBRATED, fine_res)))
            if len(raster_fine.domain)
            else []
        )
        report["caveats"] = _caveats(
            config, analyses, calibration, factors_by_res.get(fine_res, {})
        )

        (processed / REPORT_FILE).write_text(
            json.dumps(_jsonable(report), indent=2, allow_nan=False), encoding="utf-8"
        )
        for (weighting, res), analysis in analyses.items():
            logger.info("equity r%d [%s]: %d inhabited hexes", res, weighting, len(analysis.domain))
        if "headline" in report:
            logger.info("headline: %s", report["headline"]["sentence"])

    ensure_built(outputs, inputs, build, force=force or bool(orphans))
