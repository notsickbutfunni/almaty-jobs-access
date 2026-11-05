"""Spatial autocorrelation with PySAL: global Moran's I and LISA.

Global Moran's I answers "is accessibility spatially clustered at all?"
(for a cumulative accessibility surface the answer is yes by
construction — neighbouring hexes share most of what they can reach).
The informative tool is the LOCAL statistic (LISA, Anselin 1995): for
every hex it tests whether the hex and its neighbours are jointly high
(HH, a hot spot) or jointly low (LL, a cold spot) beyond what a random
reshuffling of the other hexes' values would produce.

Neighbours are the six H3 ring-1 hexes that are also inhabited; weights
are row-standardized (each hex compares itself with the MEAN of its
neighbours). esda supplies the statistic and the seeded conditional
permutations; the two-sided pseudo p-value is computed HERE from those
permutations, because ties break every p-value esda offers:

- Many accessibility measures are zero-inflated (most hexes reach no
  school within the catchment), so a zero hex among zero neighbours
  TIES with every permutation that also draws all-zero neighbours.
- esda's ``'two-sided'`` mirrors the observed percentile into the
  opposite tail and counts everything beyond it — the tie mass lands
  inside that count and inflates every p-value (kindergartens: none
  below 0.084 at 999 permutations or 0.090 at 9,999, so not a single
  significant cell).
- esda's one-sided tails compare floats with ``>=``; the observed
  statistic and its tied permutations differ in the last bits, the
  ties are NOT counted, and a zero hex with two zero neighbours gets
  p = 0.001 at 999 permutations although a third of them tie with it
  (tie-inclusive one-sided tail 0.33).

``tie_aware_p`` therefore counts both tails with a float tolerance and
doubles the smaller one — the textbook conservative two-sided
permutation p-value. On the tie-free walk measure it agrees closely
with esda's ``'two-sided'`` (340 HH / 781 LL against 343 / 800 at 999
permutations; 341 / 803 against 341 / 809 at 9,999). esda's remaining
un-directed option, ``'folded'``, does not fall into either tie trap
but folds the reference distribution about its mean — and that
distribution is skewed, so one tail becomes unreachable (397 HH and
not one walking cold spot at 999 permutations).

Pseudo p-values are estimates, so counts of "significant" cells move
with the seed: ``monte_carlo_margin`` reports how many cells sit within
Monte Carlo error of alpha.
"""

from __future__ import annotations

from collections.abc import Sequence

import h3
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from libpysal.weights import W

#: Fewer inhabited hexes than this and local statistics are noise: the
#: build records a skip reason instead (the 3-hex sample grid at r8).
MIN_HEXES_FOR_LISA = 20

#: PySAL quadrant codes of the Moran scatterplot.
QUADRANT_LABELS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}

NOT_SIGNIFICANT = "ns"
ISLAND = "island"


def hex_weights(cells: Sequence[str]) -> tuple[W, np.ndarray]:
    """Row-standardized contiguity weights over a set of H3 cells.

    Returns ``(w, islands)``; ``islands`` flags cells none of whose six
    neighbours is in the set. Islands have no neighbourhood to compare
    with, so they can never form a cluster.
    """
    cell_list = list(cells)
    cell_set = set(cell_list)
    neighbors = {cell: [n for n in h3.grid_ring(cell, 1) if n in cell_set] for cell in cell_list}
    islands = np.array([not neighbors[cell] for cell in cell_list], dtype=bool)
    weights = W(neighbors, id_order=cell_list, silence_warnings=True)
    weights.transform = "r"
    return weights, islands


def global_moran(values: np.ndarray, weights: W, *, permutations: int, seed: int) -> dict:
    """Global Moran's I with a permutation pseudo p-value (one-sided).

    esda's global statistic draws its permutations from numpy's GLOBAL
    random state and offers no seed argument, so the state is seeded
    here to make reports reproducible.
    """
    np.random.seed(seed)
    result = Moran(np.asarray(values, dtype="float64"), weights, permutations=permutations)
    return {
        "I": round(float(result.I), 4),
        "expected_I": round(float(result.EI), 4),
        "p_sim": round(float(result.p_sim), 4),
        "z_sim": round(float(result.z_sim), 2),
    }


def benjamini_hochberg(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate mask (NaN never passes).

    With m tests, sort p ascending and find the largest k with
    p_(k) <= k / m * alpha; everything up to p_(k) is a discovery. It
    keeps the EXPECTED share of false positives among the reported
    clusters below alpha — the multiple-testing question LISA raises
    when it runs one test per hex.
    """
    p = np.asarray(p_values, dtype="float64")
    valid = np.isfinite(p)
    mask = np.zeros(len(p), dtype=bool)
    m = int(valid.sum())
    if m == 0:
        return mask
    ordered = np.sort(p[valid])
    below = ordered <= (np.arange(1, m + 1) / m) * alpha
    if not below.any():
        return mask
    threshold = ordered[np.flatnonzero(below).max()]
    mask[valid] = p[valid] <= threshold
    return mask


def fdr_resolution(p_values: np.ndarray, alpha: float, permutations: int) -> dict:
    """Can the permutation count resolve FDR control at all?

    The smallest two-sided pseudo p-value is ``2 / (permutations + 1)``.
    Benjamini-Hochberg over m tests admits k cells at that floor only if
    ``floor <= k / m * alpha``, i.e. when at least ``n_needed`` cells sit
    there. A zero FDR count with fewer floor-level cells than that is a
    limit of the permutation count, not evidence that nothing survives.
    """
    p = np.asarray(p_values, dtype="float64")
    valid = np.isfinite(p)
    tests = int(valid.sum())
    floor = 2.0 / (permutations + 1)
    return {
        "tests": tests,
        "min_p": floor,
        "n_at_floor": int((p[valid] <= floor * (1.0 + 1e-9)).sum()),
        "n_needed": int(np.ceil(floor * tests / alpha)) if tests else 0,
    }


def fdr_summary(local: pd.DataFrame, alpha: float, permutations: int) -> dict:
    """FDR survivors of a ``local_moran`` frame - or None when unresolved.

    Returns the number of cells surviving Benjamini-Hochberg control,
    split into hot and cold spots, together with ``fdr_resolution``.
    When nothing survives AND there are floor-level cells but fewer of
    them than BH needs, the counts are None: "this permutation count
    cannot tell" is a different statement from "nothing survives".
    """
    labels = local["label"].to_numpy()
    survives = local["fdr"].to_numpy()
    resolution = fdr_resolution(local["p"].to_numpy(), alpha, permutations)
    unresolved = not survives.any() and 0 < resolution["n_at_floor"] < resolution["n_needed"]
    return {
        "significant": None if unresolved else int(survives.sum()),
        "hot_spots": None if unresolved else int(((labels == "HH") & survives).sum()),
        "cold_spots": None if unresolved else int(((labels == "LL") & survives).sum()),
        **resolution,
    }


def monte_carlo_margin(p_values: np.ndarray, alpha: float, permutations: int) -> dict:
    """How many cells sit within Monte Carlo error of the alpha line?

    A pseudo p-value is an ESTIMATE: it doubles a tail share counted
    over a finite number of permutations, so near alpha its standard
    error is about ``2 * sqrt(t * (1 - t) / (permutations + 1))`` with
    ``t = alpha / 2``. A cell whose p-value lies within two standard
    errors of alpha could land on either side of the line under another
    seed. On zero-inflated measures such cells come in blocks (the
    reference distribution is lumpy, so a whole range of neighbourhoods
    shares one p-value), and a cluster COUNT can move by dozens between
    seeds. Returns the band and the number of cells inside it on each
    side of alpha.
    """
    p = np.asarray(p_values, dtype="float64")
    p = p[np.isfinite(p)]
    tail = alpha / 2.0
    error = 2.0 * float(np.sqrt(tail * (1.0 - tail) / (permutations + 1)))
    low, high = max(0.0, alpha - 2.0 * error), alpha + 2.0 * error
    inside = (p >= low) & (p <= high)
    return {
        "p_band": [round(low, 4), round(high, 4)],
        "significant": int((inside & (p < alpha)).sum()),
        "not_significant": int((inside & (p >= alpha)).sum()),
    }


#: Relative float tolerance when comparing a simulated statistic with
#: the observed one: identical configurations computed along different
#: code paths differ in the last bits and must still count as ties.
TIE_RTOL = 1e-9


def tie_aware_p(observed: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    """Two-sided pseudo p-values that count ties as "as extreme".

    ``observed`` has shape (n,), ``simulated`` (n, permutations). Each
    tail is ``(count + 1) / (permutations + 1)`` with ties included up
    to a float tolerance; the p-value doubles the smaller tail, capped
    at 1.
    """
    observed = np.asarray(observed, dtype="float64")[:, None]
    simulated = np.asarray(simulated, dtype="float64")
    tolerance = TIE_RTOL * (1.0 + np.abs(observed))
    permutations = simulated.shape[1]
    upper = ((simulated >= observed - tolerance).sum(axis=1) + 1) / (permutations + 1)
    lower = ((simulated <= observed + tolerance).sum(axis=1) + 1) / (permutations + 1)
    return np.minimum(1.0, 2.0 * np.minimum(upper, lower))


def local_moran(
    values: np.ndarray,
    weights: W,
    islands: np.ndarray,
    *,
    permutations: int,
    seed: int,
    alpha: float,
) -> pd.DataFrame:
    """LISA cluster label and pseudo p-value per cell.

    Columns: ``label`` (HH / LL / HL / LH where p < alpha, ``ns``
    otherwise, ``island`` for cells without neighbours), ``p`` (the
    tie-aware two-sided pseudo p-value; NaN for islands) and ``fdr``
    (True where the cell also survives Benjamini-Hochberg control at
    the same alpha).
    """
    # Islands make esda divide 0 by 0 in a z-score this module never
    # reads; the errstate keeps that noise out of the pipeline log.
    with np.errstate(invalid="ignore", divide="ignore"):
        result = Moran_Local(
            np.asarray(values, dtype="float64"),
            weights,
            permutations=permutations,
            seed=seed,
            # Named only to silence esda's deprecation notice: its own
            # p_sim is not used (see the module docstring).
            alternative="two-sided",
            keep_simulations=True,
        )
    p = tie_aware_p(result.Is, result.rlisas)
    p[islands] = np.nan
    labels = np.array([QUADRANT_LABELS[int(q)] for q in result.q], dtype=object)
    significant = np.isfinite(p) & (p < alpha)
    labels[~significant] = NOT_SIGNIFICANT
    labels[islands] = ISLAND
    return pd.DataFrame(
        {
            "label": labels,
            "p": p,
            "fdr": benjamini_hochberg(p, alpha) & significant,
        }
    )
