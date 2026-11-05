"""Population-weighted inequality measures: Lorenz, Gini, Palma.

The unit of observation is a hex, but the unit of concern is a person:
every resident of a hex is assigned that hex's accessibility, and every
statistic weights hexes by their population. Sorting residents from the
worst to the best served and accumulating both people and accessibility
gives the Lorenz curve; because everyone inside a hex shares one value,
the curve is exactly piecewise linear with one vertex per hex — so the
Gini coefficient and the Palma ratio below are exact, not approximated
on a grid.
"""

from __future__ import annotations

import numpy as np

#: Palma ratio shares (the definition, not tunables): the access held by
#: the best-served 10% of residents over that held by the worst 40%.
PALMA_TOP = 0.10
PALMA_BOTTOM = 0.40

#: Relative slack when a cumulative weight is compared with a quantile
#: target (floating-point sums rarely land exactly on a boundary).
QUANTILE_RTOL = 1e-12


def _sorted_valid(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Finite values with positive weights, sorted ascending by value."""
    v = np.asarray(values, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    if v.shape != w.shape:
        raise ValueError(f"values and weights differ in shape: {v.shape} vs {w.shape}")
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[keep], w[keep]
    if (v < 0).any():
        raise ValueError("inequality measures need non-negative values")
    order = np.argsort(v, kind="stable")
    return v[order], w[order]


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Population-weighted quantile: the smallest value x such that at
    least a share ``q`` of the total weight has a value <= x.

    This is the inverse of the weighted empirical CDF — for integer
    weights it equals ``numpy.quantile(repeated, q, method="inverted_cdf")``.
    Returns NaN when nothing has positive weight.
    """
    if not 0 <= q <= 1:
        raise ValueError(f"quantile must be in [0, 1], got {q}")
    v, w = _sorted_valid(values, weights)
    if not len(v):
        return float("nan")
    # Compare on the raw cumulative weights with a relative tolerance:
    # a target that exceeds the running sum only by rounding (five
    # weights of 0.3 at q = 0.2: 0.2 * 1.5 evaluates to
    # 0.30000000000000004, a hair above the first cumulative weight
    # 0.3) must count as reached, or the answer skips to the next value.
    cumulative = np.cumsum(w)
    target = q * cumulative[-1] * (1.0 - QUANTILE_RTOL)
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(v[min(index, len(v) - 1)])


def _lorenz_vertices(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Exact Lorenz vertices (population share, access share), or None
    when the distribution is empty or sums to zero."""
    v, w = _sorted_valid(values, weights)
    if not len(v):
        return None
    mass = v * w
    total = mass.sum()
    if total <= 0:
        return None
    population = np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    access = np.concatenate([[0.0], np.cumsum(mass) / total])
    population[-1] = 1.0  # remove floating-point drift at the end point
    access[-1] = 1.0
    return population, access


def lorenz_curve(
    values: np.ndarray, weights: np.ndarray, n_points: int = 101
) -> tuple[np.ndarray, np.ndarray]:
    """Lorenz curve sampled on a regular population grid.

    Returns ``(pop_share, access_share)``: the bottom ``pop_share`` of
    residents (worst served first) hold ``access_share`` of all
    accessibility. Sampling is exact — the curve is linear between hex
    vertices. An empty or all-zero distribution returns NaN shares.
    """
    grid = np.linspace(0.0, 1.0, n_points)
    vertices = _lorenz_vertices(values, weights)
    if vertices is None:
        return grid, np.full(n_points, np.nan)
    population, access = vertices
    return grid, np.interp(grid, population, access)


def gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Population-weighted Gini coefficient in [0, 1).

    Twice the area between the Lorenz curve and the diagonal of perfect
    equality: G = 1 - sum((P_k - P_{k-1}) * (L_k + L_{k-1})). 0 means
    every resident has the same accessibility; values near 1 mean a few
    residents hold nearly all of it. NaN when the total is zero.
    """
    vertices = _lorenz_vertices(values, weights)
    if vertices is None:
        return float("nan")
    population, access = vertices
    area = float(np.sum(np.diff(population) * (access[1:] + access[:-1])))
    return max(0.0, 1.0 - area)


def palma(values: np.ndarray, weights: np.ndarray) -> float:
    """Palma ratio: access share of the best-served 10% of residents
    divided by the share of the worst-served 40%.

    Under perfect equality it is 0.25 (10% / 40%). ``inf`` when the
    bottom 40% hold nothing at all; NaN when the total is zero.
    """
    vertices = _lorenz_vertices(values, weights)
    if vertices is None:
        return float("nan")
    population, access = vertices
    bottom = float(np.interp(PALMA_BOTTOM, population, access))
    top = 1.0 - float(np.interp(1.0 - PALMA_TOP, population, access))
    if bottom <= 0:
        return float("inf")
    return top / bottom


def bottom_share_mean(values: np.ndarray, weights: np.ndarray, share: float) -> float:
    """Mean value of the worst-served ``share`` of residents.

    Read off the Lorenz curve: the bottom ``share`` of residents hold
    L(share) of everything, so their mean is L(share) / share times the
    overall mean. A hex straddling the cut contributes its residents
    proportionally. Returns 0.0 when the total is zero, NaN when there
    is nothing to average.
    """
    v, w = _sorted_valid(values, weights)
    if not len(v):
        return float("nan")
    vertices = _lorenz_vertices(v, w)
    if vertices is None:
        return 0.0
    population, access = vertices
    overall = float((v * w).sum() / w.sum())
    return float(np.interp(share, population, access)) / share * overall


def distribution_summary(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """All summary statistics of one measure over one set of hexes."""
    v, w = _sorted_valid(values, weights)
    if not len(v):
        nan = float("nan")
        return {
            "n_hexes": 0,
            "population": 0.0,
            "mean": nan,
            "median": nan,
            "p10": nan,
            "p90": nan,
            "gini": nan,
            "palma": nan,
            "pop_share_zero": nan,
        }
    total_weight = float(w.sum())
    return {
        "n_hexes": len(v),
        "population": total_weight,
        "mean": float((v * w).sum() / total_weight),
        "median": weighted_quantile(v, w, 0.5),
        "p10": weighted_quantile(v, w, 0.1),
        "p90": weighted_quantile(v, w, 0.9),
        "gini": gini(v, w),
        "palma": palma(v, w),
        "pop_share_zero": float(w[v == 0].sum() / total_weight),
    }
