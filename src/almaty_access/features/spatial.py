"""Spatial autocorrelation: Moran's I on the H3 grid.

Moran's I asks "do similar values sit next to each other?":

    I = (n / S0) · (Σᵢⱼ wᵢⱼ zᵢ zⱼ) / (Σᵢ zᵢ²),   z = x - x̄

with wᵢⱼ = 1 for H3 neighbours (row-standardized) and S0 = Σ wᵢⱼ.
I ≈ +1 means strong clustering, 0 random, negative checkerboard.
Significance comes from a permutation test: shuffle the values, keep
the neighbour structure, and see how extreme the observed I is. A tiny
implementation on purpose — no extra dependency, and every step is
explainable in an interview.
"""

from __future__ import annotations

from dataclasses import dataclass

import h3
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MoranResult:
    """Observed statistic and its pseudo p-value."""

    i: float
    p_sim: float
    n_perm: int


def _neighbour_pairs(index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Row-aligned (i, j) arrays of H3 ring-1 neighbour pairs in the index."""
    position = {cell: i for i, cell in enumerate(index)}
    left: list[int] = []
    right: list[int] = []
    for cell, i in position.items():
        for neighbour in h3.grid_ring(cell, 1):
            j = position.get(neighbour)
            if j is not None:
                left.append(i)
                right.append(j)
    return np.array(left, dtype=int), np.array(right, dtype=int)


def morans_i(values: pd.Series, *, n_perm: int = 199, seed: int = 0) -> MoranResult:
    """Global Moran's I of an H3-indexed series (row-standardized weights)."""
    x = values.to_numpy(dtype=float)
    if len(x) < 3 or np.isnan(x).any():
        raise ValueError("morans_i needs >= 3 cells and no NaNs")
    if np.allclose(x, x[0]):
        return MoranResult(i=float("nan"), p_sim=float("nan"), n_perm=n_perm)

    left, right = _neighbour_pairs(values.index)
    if len(left) == 0:
        raise ValueError("no neighbouring cells in the index; is it an H3 grid?")
    # Row-standardize: each cell's neighbour weights sum to 1.
    degree = np.bincount(left, minlength=len(x)).astype(float)
    weights = 1.0 / degree[left]
    s0 = weights.sum()

    def statistic(sample: np.ndarray) -> float:
        z = sample - sample.mean()
        return float(len(z) / s0 * (weights * z[left] * z[right]).sum() / (z**2).sum())

    observed = statistic(x)
    rng = np.random.default_rng(seed)
    permuted = np.array([statistic(rng.permutation(x)) for _ in range(n_perm)])
    # Two-sided pseudo p-value: how often is a shuffle at least as extreme?
    extreme = int((np.abs(permuted) >= abs(observed)).sum())
    return MoranResult(i=observed, p_sim=(extreme + 1) / (n_perm + 1), n_perm=n_perm)
