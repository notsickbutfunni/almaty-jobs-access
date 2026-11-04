"""Assemble the Phase 3 job surfaces (``almaty-access model``).

Order of operations:

1. official targets (annual means) + resolved years,
2. Model A: GFA-coefficient weights → district-rescaled baseline,
3. Model B: district-level forest → per-hex weight surface per tree →
   each tree rescaled per district → P10/P50/P90 across trees (the
   central P50 surface is itself rescaled, so it conserves totals
   exactly; the quantile bands conserve only approximately),
4. leave-one-district-out comparison of both models + permutation
   importance (the README's done-when evidence),
5. ``jobs_r{8,9}.parquet`` + ``model_report.json``.

Buffer-ring hexes get 0 jobs: official totals exist only for the 8
districts. With fewer than ``MIN_DISTRICTS_FOR_MODEL`` districts in
the grid (the sample covers one), the forest is skipped with a
recorded reason and the rf columns carry the baseline, keeping the
schema stable for CI and Phase 4.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from almaty_access.config import Config
from almaty_access.download.stats import FILE_NAME as STATS_FILE
from almaty_access.features.build import FEATURES_FILE_TEMPLATE
from almaty_access.io import ensure_built
from almaty_access.model.aggregate import apply_log1p, district_design, hex_design
from almaty_access.model.baseline import baseline_weights, rescale_to_totals
from almaty_access.model.forest import (
    fit_forest,
    lodo_table,
    metric_table,
    training_table,
    tree_weights,
)
from almaty_access.model.target import load_district_targets, resolve_years

logger = logging.getLogger(__name__)

JOBS_FILE_TEMPLATE = "jobs_r{res}.parquet"
MODEL_REPORT_FILE = "model_report.json"

#: Below this many districts with hexes the forest cannot generalise
#: (and LODO is meaningless), so Model B is skipped with a reason.
MIN_DISTRICTS_FOR_MODEL = 4


def _rescaled_tree_surfaces(
    weights: np.ndarray, districts: pd.Series, totals: dict[str, float]
) -> np.ndarray:
    """Rescale every tree's weight surface per district (vectorized).

    Every hex must belong to a district in ``totals`` (the caller
    passes per-resolution totals precisely so this holds); a stray
    district raises a clear error instead of a NaN-index crash. A
    district whose weights sum to zero in some tree (possible once the
    empty-hex mask zeroes weights) falls back to an equal split over
    its hexes — the same "we know nothing here" prior as Model A's.
    """
    names = list(totals)
    mapped = districts.map({name: i for i, name in enumerate(names)})
    if mapped.isna().any():
        stray = sorted(districts.loc[mapped.isna()].unique())
        raise ValueError(f"hexes belong to districts without totals: {stray}")
    codes = mapped.to_numpy(dtype=int)
    indicator = np.zeros((len(codes), len(names)))
    indicator[np.arange(len(codes)), codes] = 1.0
    sums = weights @ indicator  # (n_trees, n_districts)
    totals_row = np.array([totals[name] for name in names])
    hex_counts = indicator.sum(axis=0)
    with np.errstate(divide="ignore"):
        scale = np.where(sums > 0, totals_row / np.where(sums > 0, sums, 1.0), 0.0)
    rescaled = weights * scale[:, codes]
    # Equal split for (tree, district) cells whose weights vanished.
    zero_cells = sums <= 0
    if zero_cells.any():
        equal = (totals_row / hex_counts)[None, :].repeat(weights.shape[0], axis=0)
        for tree_index, district_index in zip(*np.nonzero(zero_cells), strict=True):
            rescaled[tree_index, codes == district_index] = equal[tree_index, district_index]
    return rescaled


def _job_capable(features: pd.DataFrame) -> pd.Series:
    """WorldPop-style mask: hexes that can plausibly hold payroll jobs.

    exp(prediction) is strictly positive, so without a mask the forest
    smears jobs over bare mountain hexes (the review measured 10.6% of
    all city jobs on hexes with zero GFA, zero POIs and zero
    population — 38.6% of Medeu's total). A hex with no floor space
    AND no POI gets weight 0; the official total then concentrates on
    the district's built-up hexes.
    """
    rows = features.set_index("h3")
    return (rows["bf_gfa_m2"] > 0) | (rows["poi_cnt_total"] > 0)


def build_model(config: Config, *, force: bool = False) -> list[Path]:
    """Build job surfaces at both resolutions with make-style caching."""
    processed = config.paths.processed
    resolutions = (config.h3.analysis_resolution, config.h3.fine_resolution)
    outputs = [
        *(processed / JOBS_FILE_TEMPLATE.format(res=res) for res in resolutions),
        processed / MODEL_REPORT_FILE,
    ]
    inputs = [
        *(processed / FEATURES_FILE_TEMPLATE.format(res=res) for res in resolutions),
        config.paths.raw / STATS_FILE,
        config.source_path,
    ]

    def build() -> None:
        features = {
            res: pd.read_parquet(processed / FEATURES_FILE_TEMPLATE.format(res=res))
            for res in resolutions
        }
        analysis_res = config.h3.analysis_resolution
        targets = load_district_targets(config)
        target_year, training_years = resolve_years(targets, config)
        official = (
            targets.loc[targets["year"] == target_year].set_index("district")["jobs"].astype(float)
        )

        # District sets differ per resolution at grid edges (a border
        # sliver can hold r9 hexes but no r8 hex), so totals are
        # resolved per resolution — the review caught the r8 set being
        # applied to the r9 surface.
        present = {
            res: sorted(features[res].loc[features[res]["in_city"], "district"].unique())
            for res in resolutions
        }
        totals = {
            res: {district: float(official[district]) for district in present[res]}
            for res in resolutions
        }
        without_hexes = {
            res: sorted(set(official.index) - set(present[res])) for res in resolutions
        }
        for res in resolutions:
            if without_hexes[res]:
                logger.warning(
                    "r%d: districts without hexes (totals stay unallocated there): %s",
                    res,
                    ", ".join(without_hexes[res]),
                )

        # ---- Model A ----------------------------------------------------
        weights_a = {res: baseline_weights(features[res], config) for res in resolutions}
        district_of = {res: features[res].set_index("h3")["district"] for res in resolutions}
        jobs_a = {
            res: rescale_to_totals(weights_a[res], district_of[res], totals[res])
            for res in resolutions
        }

        # ---- Model B ----------------------------------------------------
        rf_skipped = ""
        n_training_districts = len(present[analysis_res])
        if n_training_districts < MIN_DISTRICTS_FOR_MODEL:
            rf_skipped = (
                f"only {n_training_districts} district(s) at the analysis resolution "
                f"(< {MIN_DISTRICTS_FOR_MODEL}); a district-level forest cannot "
                "generalise and LODO is meaningless - rf columns carry the baseline"
            )
            logger.warning("random forest skipped: %s", rf_skipped)

        jobs_rf = {res: jobs_a[res].copy() for res in resolutions}
        jobs_p10 = {res: jobs_a[res].copy() for res in resolutions}
        jobs_p90 = {res: jobs_a[res].copy() for res in resolutions}
        lodo: pd.DataFrame | None = None
        importances: list[dict[str, float | str]] = []

        if not rf_skipped:
            prefixes = config.features.log1p_prefixes
            district_logged = apply_log1p(district_design(features[analysis_res], config), prefixes)
            X, y, _ = training_table(district_logged, targets, training_years)
            forest = fit_forest(X, y, config)

            q_low, q_mid, q_high = config.model.quantiles
            for res in resolutions:
                design = apply_log1p(hex_design(features[res]), prefixes)[list(X.columns)]
                capable = _job_capable(features[res]).loc[design.index].to_numpy()
                surfaces = _rescaled_tree_surfaces(
                    tree_weights(forest, design) * capable,
                    district_of[res].loc[design.index],
                    totals[res],
                )
                p10, p90 = np.quantile(surfaces, [q_low, q_high], axis=0)
                # Central estimate: rescale the median surface once more
                # so the result conserves district totals exactly (the
                # per-hex median of individually conserving surfaces is
                # itself only approximately conserving).
                median_surface = pd.Series(np.quantile(surfaces, q_mid, axis=0), index=design.index)
                central = rescale_to_totals(
                    median_surface, district_of[res].loc[design.index], totals[res]
                )
                jobs_rf[res].loc[design.index] = central
                jobs_p10[res].loc[design.index] = p10
                jobs_p90[res].loc[design.index] = p90
                logger.info(
                    "r%d: %d of %d city hexes job-capable (mask zeroes the rest)",
                    res,
                    int(capable.sum()),
                    len(capable),
                )

            baseline_unscaled = (
                weights_a[analysis_res]
                .groupby(district_of[analysis_res])
                .sum()
                .reindex(present[analysis_res])
            )
            lodo = lodo_table(
                district_logged, targets, target_year, training_years, baseline_unscaled, config
            )
            permutation = permutation_importance(
                forest, X, y, n_repeats=25, random_state=config.random_seed
            )
            order = np.argsort(permutation.importances_mean)[::-1][:15]
            importances = [
                {"feature": str(X.columns[i]), "importance": float(permutation.importances_mean[i])}
                for i in order
            ]

        # ---- outputs -----------------------------------------------------
        processed.mkdir(parents=True, exist_ok=True)
        for res in resolutions:
            frame = features[res][["h3", "res", "district", "in_city"]].copy()
            index = pd.Index(frame["h3"])
            frame["jobs_baseline"] = jobs_a[res].reindex(index).fillna(0.0).to_numpy()
            frame["jobs_rf"] = jobs_rf[res].reindex(index).fillna(0.0).to_numpy()
            frame["jobs_rf_p10"] = jobs_p10[res].reindex(index).fillna(0.0).to_numpy()
            frame["jobs_rf_p90"] = jobs_p90[res].reindex(index).fillna(0.0).to_numpy()
            frame.to_parquet(processed / JOBS_FILE_TEMPLATE.format(res=res), index=False)
            logger.info(
                "jobs r%d: %.0f baseline / %.0f rf jobs over %d hexes",
                res,
                frame["jobs_baseline"].sum(),
                frame["jobs_rf"].sum(),
                int((frame["jobs_rf"] > 0).sum()),
            )

        # Plausibility (README §3.3): jobs per resident, the top job
        # hexes, the industrial jobs-high/population-low contrast, and
        # whether hospital/university anchor hexes rank high.
        analysis = features[analysis_res].set_index("h3")
        analysis_totals = totals[analysis_res]
        ratios = {}
        for district in present[analysis_res]:
            mask = analysis["district"].eq(district).fillna(False)
            residents = float(analysis.loc[mask, "pop_residents"].sum())
            ratios[district] = {
                "jobs": round(analysis_totals[district]),
                "residents": round(residents),
                "jobs_per_resident": round(analysis_totals[district] / residents, 3)
                if residents > 0
                else None,
            }
        top = (
            pd.DataFrame(
                {
                    "h3": jobs_rf[analysis_res].index,
                    "district": district_of[analysis_res]
                    .loc[jobs_rf[analysis_res].index]
                    .to_numpy(),
                    "jobs_rf": jobs_rf[analysis_res].to_numpy(),
                }
            )
            .nlargest(20, "jobs_rf")
            .round(0)
        )

        city = analysis.loc[analysis["in_city"]].copy()
        city["jobs_rf"] = jobs_rf[analysis_res].reindex(city.index)
        industrial_top = city.nlargest(min(20, len(city)), "bf_gfa_industrial_m2")
        median_jobs = float(city["jobs_rf"].median())
        anchor_mask = pd.Series(False, index=city.index)
        for column in ("poi_cnt_health_hospital", "poi_cnt_education_university"):
            if column in city.columns:
                anchor_mask |= city[column] > 0
        anchors = city.loc[anchor_mask]
        plausibility = {
            # Industrial zones should hold many jobs and few residents.
            "industrial_top20_mean_jobs": round(float(industrial_top["jobs_rf"].mean()), 1),
            "industrial_top20_mean_residents": round(
                float(industrial_top["pop_residents"].mean()), 1
            ),
            "city_mean_jobs": round(float(city["jobs_rf"].mean()), 1),
            "city_mean_residents": round(float(city["pop_residents"].mean()), 1),
            # Hospitals and universities should sit above the median hex.
            "anchor_hex_count": len(anchors),
            "anchor_share_above_median_jobs": round(
                float((anchors["jobs_rf"] > median_jobs).mean()), 3
            )
            if len(anchors)
            else None,
        }

        metrics_a = metrics_b = None
        if lodo is not None:
            actual = lodo["actual"].to_numpy()
            metrics_a = metric_table(actual, lodo["pred_baseline"].to_numpy())
            metrics_b = metric_table(actual, lodo["pred_rf"].to_numpy())

        report = {
            "target_indicator": config.model.target_indicator,
            "target_year": target_year,
            "training_years": training_years,
            "districts_modeled": {f"r{res}": present[res] for res in resolutions},
            "districts_without_hexes": {f"r{res}": without_hexes[res] for res in resolutions},
            "official_totals": {k: round(float(v)) for k, v in official.items()},
            "rf_skipped_reason": rf_skipped,
            "lodo": lodo.round(1).to_dict("records") if lodo is not None else [],
            "lodo_metrics_baseline": metrics_a,
            "lodo_metrics_rf": metrics_b,
            "rf_beats_baseline": (
                bool(metrics_b["mape"] < metrics_a["mape"]) if metrics_a and metrics_b else None
            ),
            "permutation_importance_top15": importances,
            "jobs_per_resident": ratios,
            "plausibility": plausibility,
            "top20_hexes_rf": top.to_dict("records"),
            "quantiles": list(config.model.quantiles),
            "baseline_jobs_per_100m2": config.model.baseline_jobs_per_100m2,
            "random_seed": config.random_seed,
        }
        (processed / MODEL_REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")

    ensure_built(outputs, inputs, build, force=force)
    return outputs
