"""Neighbourhood context: k-ring sums and means of key features.

Job clusters spill over hex borders (README §7 Phase 2), so for each
configured feature we add the sum and mean over the H3 disk of radius
k (the hex and its k rings of neighbours). Disks are intersected with
the grid; because the grid extends 5 km beyond the city, disks of city
hexes are complete and only buffer-edge hexes see truncated disks.
"""

from __future__ import annotations

from collections.abc import Sequence

import h3
import numpy as np
import pandas as pd


def kring_features(
    features: pd.DataFrame, columns: Sequence[str], ks: Sequence[int]
) -> pd.DataFrame:
    """``ctx_k{k}_{sum,mean}_{column}`` over H3 disks, aligned to features.

    ``features`` must be indexed by H3 cell id and hold every column in
    ``columns``; missing columns raise (a config typo must not produce
    a silently empty context block).
    """
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise ValueError(f"context_columns not in the feature table: {missing}")

    cells = features.index.to_numpy()
    position = {cell: i for i, cell in enumerate(cells)}
    values = features[list(columns)].to_numpy(dtype=float)

    out = pd.DataFrame(index=features.index)
    for k in ks:
        sums = np.zeros((len(cells), len(columns)))
        counts = np.zeros(len(cells))
        for i, cell in enumerate(cells):
            neighbours = [
                position[neighbour] for neighbour in h3.grid_disk(cell, k) if neighbour in position
            ]
            sums[i] = values[neighbours].sum(axis=0)
            counts[i] = len(neighbours)
        for j, column in enumerate(columns):
            out[f"ctx_k{k}_sum_{column}"] = sums[:, j]
            out[f"ctx_k{k}_mean_{column}"] = sums[:, j] / counts
    return out
