"""Population-weight sensitivity: is the resident raster itself biased?

Every equity statistic weights hexes by residents, and the residents
come from Kontur Population — a modelled raster. Checked against the one
residence-based series the pipeline already holds (employed residents
per district; for the city, the Taldau labour-force survey), the raster
turns out to be skewed between districts: in Almaty it gives the dense
high-rise districts well under their share (Auezov about 0.4 of it) and
the low-rise periphery more. Those dense districts are also the best
served, so the skew inflates every inequality figure.

Phase 5 does not rewrite the population layer (that would change the
outputs of Phases 2-4 and is a decision for the project owner). It runs
the whole equity analysis TWICE instead: on the raster as it is, and on
weights rescaled so district totals follow the district shares of that
series. Employed residents are not total population, but their split
across districts is a far better guide than a raster known to be off by
more than a factor of two.

The module quotes the statistics file's ``source`` and never calls it
official: the only status the project can vouch for is its own
synthetic sample numbers, and those are labelled as such.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from almaty_access.config import Config
from almaty_access.download.stats import FILE_NAME as STATS_FILE
from almaty_access.download.stats import SYNTHETIC_SOURCE

logger = logging.getLogger(__name__)

#: The residence-based indicator used as the calibration target.
RESIDENT_INDICATOR = "employed_residents"


@dataclass(frozen=True)
class ResidentShares:
    """District shares of employed residents and where they came from."""

    shares: pd.Series
    """Share per configured district, summing to 1."""
    year: int
    source: str
    """The ``source`` column of the rows used, verbatim (joined when
    several; "unknown" when the file has no such column). The code
    quotes it and never certifies it: whether a hand-collected file is
    official is for its owner to say."""
    synthetic: bool
    """True for the sample's fake numbers — the one status the project
    itself writes and can therefore vouch for."""

    @property
    def method(self) -> str:
        """One-sentence description for reports, honest about the source."""
        basis = (
            f"the district shares of employed residents in {STATS_FILE} "
            f"({self.year}; source: {self.source})"
        )
        if self.synthetic:
            basis += " - SYNTHETIC test numbers, not real statistics"
        return (
            f"hex residents rescaled so that district totals follow {basis}; the city "
            "total and the within-district distribution are unchanged"
        )


def district_resident_shares(config: Config) -> ResidentShares | None:
    """Share of employed residents per district, latest fully covered year.

    Returns None when the statistics file or the indicator is absent, or
    no year covers every configured district — the calibrated run is
    then skipped and the report says why.
    """
    path = config.paths.raw / STATS_FILE
    if not path.exists():
        return None
    stats = pd.read_csv(path)
    rows = stats.loc[stats["indicator"] == RESIDENT_INDICATOR]
    if rows.empty:
        logger.warning("%s has no '%s' rows: calibrated run skipped", path, RESIDENT_INDICATOR)
        return None
    expected = [district.key for district in config.districts]
    annual = rows.groupby(["year", "district"])["value"].mean().unstack("district")
    annual = annual.reindex(columns=expected).dropna()
    annual = annual.loc[(annual > 0).all(axis=1)]
    if annual.empty:
        logger.warning(
            "no year of '%s' covers all districts: calibrated run skipped", RESIDENT_INDICATOR
        )
        return None
    year = int(annual.index.max())
    values = annual.loc[year]
    sources: list[str] = []
    if "source" in rows.columns:  # an optional column of the statistics file
        sources = sorted({str(s) for s in rows.loc[rows["year"] == year, "source"].dropna()})
    return ResidentShares(
        shares=values / values.sum(),
        year=year,
        source="; ".join(sources) if sources else "unknown",
        synthetic=SYNTHETIC_SOURCE in sources,
    )


def calibrated_weights(
    population: pd.Series, district: pd.Series, shares: pd.Series
) -> tuple[np.ndarray, dict[str, float]]:
    """Hex weights whose district totals follow the given shares.

    Each district's hexes are multiplied by one factor, target share
    over raster share (both renormalized over the districts actually
    present, so a partial grid stays consistent). The total is
    preserved; a hex without a district keeps its raster value. Returns
    the weights and the per-district factors.
    """
    names = district.astype("string").fillna("").to_numpy(dtype=str)
    values = population.to_numpy(dtype="float64")
    present = sorted(set(names) - {""})
    raster = pd.Series({name: values[names == name].sum() for name in present})
    target = shares.reindex(present)
    usable = target.notna() & (raster > 0)
    factors = pd.Series(1.0, index=present)
    if usable.any():
        target_share = target[usable] / target[usable].sum()
        raster_share = raster[usable] / raster[usable].sum()
        factors[usable] = target_share / raster_share
    weights = values * pd.Series(names).map(factors).fillna(1.0).to_numpy()
    return weights, {str(name): float(factor) for name, factor in factors.items()}


def apply_factors(
    population: pd.Series, district: pd.Series, factors: dict[str, float]
) -> np.ndarray:
    """Multiply each hex by its district's factor (1.0 where unknown).

    Used to carry the factors derived on the inhabited domain to EVERY
    hex — the 2SFCA demand term needs residents everywhere, buffer ring
    included.
    """
    names = district.astype("string").fillna("").to_numpy(dtype=str)
    multipliers = pd.Series(names).map(factors).fillna(1.0).to_numpy()
    return population.to_numpy(dtype="float64") * multipliers
