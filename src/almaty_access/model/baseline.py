"""Model A: the floor-area heuristic, and the shared rescaling step.

jobs_hex ∝ Σ_f GFA_f(hex) · coefficient_f — a transparent weight
surface. The coefficients (config) only shape the surface WITHIN each
district: :func:`rescale_to_totals` then forces every district's hex
sum to equal the official total (the mass-preserving / pycnophylactic
constraint both models share). Model A is the benchmark Model B must
beat on leave-one-district-out error (README §7 Phase 3).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd

from almaty_access.config import Config
from almaty_access.schemas import BUILDING_FUNCTIONS

logger = logging.getLogger(__name__)


def baseline_weights(features: pd.DataFrame, config: Config) -> pd.Series:
    """Unscaled job weights per hex from raw GFA columns (h3-indexed)."""
    rows = features.set_index("h3")
    weights = pd.Series(0.0, index=rows.index, name="weight")
    for function in BUILDING_FUNCTIONS:
        coefficient = config.model.baseline_jobs_per_100m2[function]
        weights += rows[f"bf_gfa_{function}_m2"].astype(float) * (coefficient / 100.0)
    return weights


def rescale_to_totals(
    weights: pd.Series, district: pd.Series, totals: Mapping[str, float]
) -> pd.Series:
    """district-wise jobs: total x weight / Σ weights (mass-preserving).

    Hexes outside every district (buffer ring) get 0. A district whose
    weights sum to 0 still holds official jobs, so they are split
    equally over its hexes — loudly, because it means the surface has
    no information there.
    """
    jobs = pd.Series(0.0, index=weights.index, name="jobs")
    for name, total in totals.items():
        # .eq + fillna: buffer hexes carry district = NA, and a mask
        # holding NA cannot be used for boolean indexing.
        mask = district.eq(name).fillna(False).to_numpy()
        if not mask.any():
            continue
        district_weights = weights.loc[mask]
        weight_sum = float(district_weights.sum())
        if weight_sum > 0:
            jobs.loc[mask] = district_weights / weight_sum * total
        else:
            logger.warning(
                "district %s: zero total weight; splitting %d jobs equally over %d hexes",
                name,
                int(total),
                int(mask.sum()),
            )
            jobs.loc[mask] = total / mask.sum()
    return jobs
