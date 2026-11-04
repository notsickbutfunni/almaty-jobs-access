"""Model B: the WorldPop-style random-forest dasymetric model.

Trained at DISTRICT level — the only level with ground truth — on
log(job density): 8 districts x the configured years (features are
static 2026 OSM, so extra years stabilise the target rather than add
independent points, and validation must hold out whole districts).
The fitted forest then predicts log density PER HEX (same units thanks
to :mod:`almaty_access.model.aggregate`), giving the weight surface
that :func:`almaty_access.model.baseline.rescale_to_totals` pins to
the official totals. Uncertainty comes from the forest itself: each
tree yields a full surface, each is rescaled, and per-hex quantiles
across trees give P10/P50/P90.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from almaty_access.config import Config

logger = logging.getLogger(__name__)


def training_table(
    district_design: pd.DataFrame,
    targets: pd.DataFrame,
    training_years: list[int],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(X, y, groups): one row per district x year.

    X repeats each district's static feature row; y is log(job density)
    for that year; groups holds the district key for group-aware
    validation (all years of a district leave together).
    """
    feature_cols = [column for column in district_design.columns if column != "area_km2"]
    rows: list[pd.Series] = []
    y: list[float] = []
    groups: list[str] = []
    yearly = targets.loc[targets["year"].isin(training_years)]
    for district, area_km2 in district_design["area_km2"].items():
        for _, row in yearly.loc[yearly["district"] == district].iterrows():
            rows.append(district_design.loc[district, feature_cols])
            y.append(float(np.log(row["jobs"] / area_km2)))
            groups.append(str(district))
    X = pd.DataFrame(rows).reset_index(drop=True)
    return X, pd.Series(y, name="log_density"), pd.Series(groups, name="district")


def fit_forest(X: pd.DataFrame, y: pd.Series, config: Config) -> RandomForestRegressor:
    params = config.model.random_forest
    forest = RandomForestRegressor(
        n_estimators=params.n_estimators,
        min_samples_leaf=params.min_samples_leaf,
        max_features=params.max_features,
        random_state=config.random_seed,
        n_jobs=-1,
    )
    forest.fit(X, y)
    return forest


def tree_weights(forest: RandomForestRegressor, design: pd.DataFrame) -> np.ndarray:
    """(n_trees, n_units) weight surface per tree: exp(log density)."""
    X = design.to_numpy()
    return np.exp(np.stack([tree.predict(X) for tree in forest.estimators_]))


def metric_table(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae": float(np.abs(error).mean()),
        "mape": float((np.abs(error) / actual).mean()),
        "r2": float(r2_score(actual, predicted)),
    }


def lodo_table(
    district_design_logged: pd.DataFrame,
    targets: pd.DataFrame,
    target_year: int,
    training_years: list[int],
    baseline_totals_unscaled: pd.Series,
    config: Config,
) -> pd.DataFrame:
    """Leave-one-district-out totals for both models.

    For each district: Model B retrains on the other districts' years
    and predicts the held-out total from its (never seen) feature row;
    Model A scales its unscaled GFA total by the training districts'
    official-to-unscaled ratio. Honest by construction: nothing about
    the held-out district's target reaches either model.
    """
    actual = targets.loc[targets["year"] == target_year].set_index("district")["jobs"].astype(float)
    rows = []
    for district in district_design_logged.index:
        train_design = district_design_logged.drop(index=district)
        X, y, _ = training_table(train_design, targets, training_years)
        forest = fit_forest(X, y, config)
        feature_cols = [c for c in district_design_logged.columns if c != "area_km2"]
        held_out = district_design_logged.loc[[district], feature_cols]
        density = float(np.exp(forest.predict(held_out)[0]))
        rf_total = density * float(district_design_logged.loc[district, "area_km2"])

        others = [d for d in district_design_logged.index if d != district]
        scale = float(actual.loc[others].sum() / baseline_totals_unscaled.loc[others].sum())
        baseline_total = float(baseline_totals_unscaled.loc[district] * scale)

        rows.append(
            {
                "district": district,
                "actual": float(actual.loc[district]),
                "pred_baseline": baseline_total,
                "pred_rf": rf_total,
            }
        )
    table = pd.DataFrame(rows)
    logger.info(
        "LODO MAPE: baseline %.1f%%, random forest %.1f%%",
        100 * metric_table(table["actual"].to_numpy(), table["pred_baseline"].to_numpy())["mape"],
        100 * metric_table(table["actual"].to_numpy(), table["pred_rf"].to_numpy())["mape"],
    )
    return table
