"""Official training targets: per-district payroll jobs from Taldau.

``payroll_jobs_located`` counts payroll employees at the district where
the enterprise reports — jobs located IN the district, which is exactly
the quantity being disaggregated (NOT ``employed_residents``, who are
counted where they live; the two differ by commuting). Quarterly values
are averaged into an annual figure to smooth seasonality.
"""

from __future__ import annotations

import logging

import pandas as pd

from almaty_access.config import Config
from almaty_access.download.stats import FILE_NAME as STATS_FILE

logger = logging.getLogger(__name__)


def load_district_targets(config: Config) -> pd.DataFrame:
    """Annual jobs per district: columns ``district``, ``year``, ``jobs``.

    Fails loudly when the indicator is missing or any configured
    district has no data — a silently shrunken training set would
    corrupt every rescaled surface downstream.
    """
    path = config.paths.raw / STATS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `almaty-access download --only stats` first"
        )
    stats = pd.read_csv(path)
    rows = stats.loc[stats["indicator"] == config.model.target_indicator]
    if rows.empty:
        available = sorted(stats["indicator"].unique())
        raise ValueError(
            f"{path}: no rows with indicator '{config.model.target_indicator}' "
            f"(available: {available})"
        )
    annual = rows.groupby(["district", "year"], as_index=False).agg(
        jobs=("value", "mean"),
        periods=("month", "nunique"),  # quarters behind the annual mean
    )
    expected = {district.key for district in config.districts}
    missing = expected - set(annual["district"])
    if missing:
        raise ValueError(f"{path}: districts without target data: {sorted(missing)}")
    if (annual["jobs"] <= 0).any():
        bad = annual.loc[annual["jobs"] <= 0, ["district", "year"]].to_dict("records")
        raise ValueError(f"{path}: non-positive job totals: {bad}")
    return annual.loc[annual["district"].isin(expected)].reset_index(drop=True)


def resolve_years(targets: pd.DataFrame, config: Config) -> tuple[int, list[int]]:
    """(target year, training years) actually available in the data.

    The config asks for specific years; the CSV decides what exists
    (the synthetic sample holds a single year). Only years where EVERY
    district reports are usable — an unbalanced panel would weight
    districts unequally.
    """
    n_districts = len(config.districts)
    complete = {
        int(year)
        for year, group in targets.groupby("year")
        if group["district"].nunique() == n_districts
    }
    if not complete:
        raise ValueError("district_stats.csv has no year covering all districts")
    # A year must also carry as many periods as the dataset normally
    # has: the live Taldau series is quarterly, and the current year's
    # lone Q1 would otherwise masquerade as an annual mean. The rule
    # adapts to the data (an annual-only series has a mode of 1).
    periods = targets.groupby(["district", "year"])["periods"].first()
    full_periods = int(periods.max())
    short_years = {int(year) for (_, year), count in periods.items() if int(count) < full_periods}
    if short_years & complete:
        logger.info(
            "excluding partially reported year(s) %s (fewer than %d periods)",
            sorted(short_years & complete),
            full_periods,
        )
        complete -= short_years
    if not complete:
        raise ValueError("district_stats.csv has no fully reported year")

    target_year = config.model.target_year
    if target_year not in complete:
        fallback = max(complete)
        logger.warning(
            "target year %d not in the data; using latest complete year %d",
            target_year,
            fallback,
        )
        target_year = fallback

    training_years = sorted(set(config.model.training_years) & complete)
    if not training_years:
        logger.warning(
            "none of the configured training years %s are in the data; using [%d]",
            list(config.model.training_years),
            target_year,
        )
        training_years = [target_year]
    return target_year, training_years
